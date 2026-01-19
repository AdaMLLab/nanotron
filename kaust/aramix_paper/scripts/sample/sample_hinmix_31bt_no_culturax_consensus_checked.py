#!/usr/bin/env python3
"""
Script to sample ~31B tokens from the HinMix MinHash deduped dataset,
filtering CulturaX documents that don't appear in the consensus dataset.

Filtering logic:
- KEEP: Documents from non-CulturaX sources
- KEEP: CulturaX documents that appear in hinmix-consensus (have duplicates in other sources)
- DROP: CulturaX documents that don't appear in hinmix-consensus (unique to CulturaX)

This approach keeps CulturaX content that is corroborated by other sources.

Dataset structure notes:
- minhash_deduped: has `metadata` dict with `source` key
- consensus: has `sources` column (list of sources where doc appears)
"""

import argparse
import glob
import hashlib
import os
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer


# Source name mapping: dataset source -> canonical name
SOURCE_MAPPING = {
    "c4": "C4",
    "C4": "C4",
    "mc4": "C4",
    "mC4": "C4",
    "culturax": "CulturaX",
    "CulturaX": "CulturaX",
    "uonlp/CulturaX": "CulturaX",
    "fineweb-2": "FineWeb-2",
    "FineWeb-2": "FineWeb-2",
    "fineweb2": "FineWeb-2",
    "HuggingFaceFW/fineweb-2": "FineWeb-2",
    "hplt-2": "HPLT-2",
    "HPLT-2": "HPLT-2",
    "hplt2": "HPLT-2",
    "HPLT/HPLT2.0_cleaned": "HPLT-2",
    "sangraha": "Sangraha",
    "Sangraha": "Sangraha",
    "ai4bharat/sangraha": "Sangraha",
}

# Target total tokens (31B with a small buffer)
TARGET_TOKENS = 33_000_000_000


def normalize_source(source: str) -> str:
    """Normalize source name to canonical form."""
    if source in SOURCE_MAPPING:
        return SOURCE_MAPPING[source]
    source_lower = source.lower()
    for key, val in SOURCE_MAPPING.items():
        if key.lower() == source_lower:
            return val
    return source


def get_source_from_filepath(filepath: str) -> str:
    """
    Extract source from parquet file path.

    The directory structure is: <base>/minhash_deduped/<source>/<filename>.parquet
    So we extract the parent directory name of the parquet file.
    """
    parent_dir = os.path.basename(os.path.dirname(filepath))
    return parent_dir


def text_hash(text: str) -> str:
    """Create a hash of text for efficient lookup."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def build_consensus_text_set(consensus_dir: str) -> Set[str]:
    """
    Build a set of text hashes from the consensus dataset.
    These are documents that appear in multiple sources.
    """
    print("Building consensus text hash set...")

    parquet_files = sorted(glob.glob(os.path.join(consensus_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {consensus_dir}")

    print(f"Found {len(parquet_files)} consensus parquet files")

    consensus_hashes: Set[str] = set()

    for pf_path in tqdm(parquet_files, desc="Loading consensus"):
        table = pq.read_table(pf_path, columns=["text"])
        texts = table.column("text").to_pylist()

        for text in texts:
            consensus_hashes.add(text_hash(text))

    print(f"Loaded {len(consensus_hashes):,} unique text hashes from consensus")
    return consensus_hashes


def is_culturax(source: str) -> bool:
    """Check if source is CulturaX."""
    canonical = normalize_source(source)
    return canonical == "CulturaX"


def estimate_tokens_per_source(
    data_dir: str,
    consensus_hashes: Set[str],
    tokenizer,
    sample_size: int = 1000
) -> Tuple[Dict[str, float], Dict[str, int], int, int]:
    """
    Estimate average tokens per document for each source by sampling.
    Also counts eligible documents (non-CulturaX or CulturaX in consensus).

    Returns:
        avg_tokens_per_doc: Dict mapping source -> average tokens per document
        doc_counts: Dict mapping source -> eligible document count
        total_culturax_docs: Total CulturaX documents
        kept_culturax_docs: CulturaX documents kept (in consensus)
    """
    print("Estimating tokens per document for each source...")

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} parquet files")

    docs_by_source: Dict[str, List[str]] = defaultdict(list)
    doc_counts: Dict[str, int] = defaultdict(int)
    total_culturax_docs = 0
    kept_culturax_docs = 0
    dropped_culturax_docs = 0

    print("Counting documents per source (with CulturaX consensus check)...")
    for pf_path in tqdm(parquet_files, desc="Counting"):
        # Extract source from file path
        src = get_source_from_filepath(pf_path)
        canonical = normalize_source(src)

        table = pq.read_table(pf_path, columns=["text"])
        texts = table.column("text").to_pylist()

        if canonical == "CulturaX":
            for text in texts:
                total_culturax_docs += 1
                # Check if in consensus
                if text_hash(text) in consensus_hashes:
                    kept_culturax_docs += 1
                    doc_counts[canonical] += 1
                else:
                    dropped_culturax_docs += 1
        else:
            doc_counts[canonical] += len(texts)

    print(f"\nCulturaX statistics:")
    print(f"  Total CulturaX documents: {total_culturax_docs:,}")
    if total_culturax_docs > 0:
        print(f"  Kept (in consensus): {kept_culturax_docs:,} ({100*kept_culturax_docs/total_culturax_docs:.1f}%)")
        print(f"  Dropped (unique): {dropped_culturax_docs:,} ({100*dropped_culturax_docs/total_culturax_docs:.1f}%)")
    else:
        print("  No CulturaX documents found in dataset")

    print("\nDocument counts by source (eligible only):")
    for src, count in sorted(doc_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count:,}")

    # Second pass: sample documents for token estimation
    print(f"\nSampling {sample_size} documents per source for token estimation...")
    sample_needs: Dict[str, int] = {src: sample_size for src in doc_counts.keys()}

    for pf_path in tqdm(parquet_files, desc="Sampling"):
        if all(n <= 0 for n in sample_needs.values()):
            break

        # Extract source from file path
        src = get_source_from_filepath(pf_path)
        canonical = normalize_source(src)

        if sample_needs.get(canonical, 0) <= 0:
            continue

        table = pq.read_table(pf_path, columns=["text"])
        texts = table.column("text").to_pylist()

        # For CulturaX, only sample from consensus-approved docs
        if canonical == "CulturaX":
            eligible_texts = [t for t in texts if text_hash(t) in consensus_hashes]
        else:
            eligible_texts = texts

        if len(eligible_texts) > 0:
            n_to_sample = min(sample_needs[canonical], len(eligible_texts))
            sampled = random.sample(eligible_texts, n_to_sample)
            docs_by_source[canonical].extend(sampled)
            sample_needs[canonical] -= n_to_sample

    # Tokenize samples
    avg_tokens_per_doc: Dict[str, float] = {}

    print("\nTokenizing samples to estimate average tokens...")
    for src, docs in tqdm(docs_by_source.items(), desc="Tokenizing"):
        if not docs:
            avg_tokens_per_doc[src] = 0
            continue

        total_tokens = 0
        for doc in docs[:sample_size]:
            tokens = tokenizer.encode(doc, add_special_tokens=False)
            total_tokens += len(tokens)

        avg_tokens_per_doc[src] = total_tokens / len(docs[:sample_size])
        print(f"  {src}: avg {avg_tokens_per_doc[src]:.1f} tokens/doc ({len(docs[:sample_size])} samples)")

    return avg_tokens_per_doc, dict(doc_counts), total_culturax_docs, kept_culturax_docs


def calculate_sampling_plan(
    doc_counts: Dict[str, int],
    avg_tokens: Dict[str, float],
    target_tokens: int
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Calculate how many documents to sample from each source proportionally.

    Returns:
        docs_to_sample: Dict mapping source -> number of documents to sample
        source_ratios: Dict mapping source -> percentage ratio
    """
    print("\nCalculating sampling plan (proportional to eligible data)...")

    available_sources = set(doc_counts.keys())

    # Calculate total eligible documents and proportional ratios
    total_docs = sum(doc_counts.values())
    source_ratios = {src: 100 * count / total_docs for src, count in doc_counts.items()}

    print("\nSource ratios (proportional to eligible data):")
    for src, ratio in sorted(source_ratios.items(), key=lambda x: -x[1]):
        print(f"  {src}: {ratio:.2f}%")

    # Calculate target tokens per source
    target_tokens_per_source = {
        src: target_tokens * (ratio / 100)
        for src, ratio in source_ratios.items()
    }

    # Calculate documents needed per source
    docs_to_sample: Dict[str, int] = {}

    print("\nSampling plan:")
    total_estimated_tokens = 0

    for src in available_sources:
        if avg_tokens.get(src, 0) == 0:
            docs_to_sample[src] = 0
            continue

        target_src_tokens = target_tokens_per_source.get(src, 0)
        needed_docs = int(target_src_tokens / avg_tokens[src])

        available_docs = doc_counts[src]
        actual_docs = min(needed_docs, available_docs)
        docs_to_sample[src] = actual_docs

        estimated_tokens = actual_docs * avg_tokens[src]
        total_estimated_tokens += estimated_tokens

        status = "" if actual_docs == needed_docs else f" (capped from {needed_docs:,})"
        print(f"  {src}: {actual_docs:,} docs (~{estimated_tokens/1e9:.2f}B tokens){status}")

    print(f"\nTotal estimated tokens: {total_estimated_tokens/1e9:.2f}B")

    return docs_to_sample, source_ratios


def sample_and_save(
    data_dir: str,
    output_dir: str,
    sampling_plan: Dict[str, int],
    consensus_hashes: Set[str],
    tokenizer,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Sample documents according to plan and save to output directory.
    CulturaX documents are only included if they appear in consensus.

    Returns:
        source_stats: Statistics about sampled documents per source
    """
    print(f"\nSampling documents and saving to {output_dir}...")

    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))

    # First pass: collect all eligible indices per source
    print("Collecting eligible document indices per source...")
    source_indices: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    for pf_path in tqdm(parquet_files, desc="Indexing"):
        # Extract source from file path
        src = get_source_from_filepath(pf_path)
        canonical = normalize_source(src)

        table = pq.read_table(pf_path, columns=["text"])
        texts = table.column("text").to_pylist()

        for idx, text in enumerate(texts):
            # For CulturaX, only include if in consensus
            if canonical == "CulturaX":
                if text_hash(text) not in consensus_hashes:
                    continue

            source_indices[canonical].append((pf_path, idx))

    # Randomly sample indices for each source
    print("Randomly sampling indices...")
    sampled_indices: Dict[str, dict] = {}

    for src, indices in source_indices.items():
        n_to_sample = sampling_plan.get(src, 0)
        if n_to_sample > 0 and indices:
            sampled = random.sample(indices, min(n_to_sample, len(indices)))
            sampled_by_file: Dict[str, set] = defaultdict(set)
            for pf_path, idx in sampled:
                sampled_by_file[pf_path].add(idx)
            sampled_indices[src] = sampled_by_file
            print(f"  {src}: sampled {len(sampled):,} documents")

    # Flatten sampled indices by file
    file_indices: Dict[str, set] = defaultdict(set)
    for src, by_file in sampled_indices.items():
        for pf_path, indices in by_file.items():
            file_indices[pf_path].update(indices)

    # Read and collect sampled documents
    print("Reading sampled documents...")
    all_sampled_rows = []

    for pf_path in tqdm(parquet_files, desc="Reading"):
        if pf_path not in file_indices:
            continue

        # Extract source from file path
        src = get_source_from_filepath(pf_path)
        canonical_source = normalize_source(src)

        indices_to_keep = file_indices[pf_path]
        table = pq.read_table(pf_path)

        for idx in sorted(indices_to_keep):
            metadata = table.column("metadata")[idx].as_py()
            row = {
                "text": table.column("text")[idx].as_py(),
                "id": table.column("id")[idx].as_py(),
                "metadata": metadata,
                "_source": canonical_source,  # Store source for statistics
            }
            all_sampled_rows.append(row)

    print(f"\nTotal sampled documents: {len(all_sampled_rows):,}")

    # Shuffle
    print("Shuffling documents...")
    random.shuffle(all_sampled_rows)

    # Estimate final token count
    print("Estimating final token count...")
    sample_for_counting = random.sample(all_sampled_rows, min(5000, len(all_sampled_rows)))
    sample_tokens = sum(
        len(tokenizer.encode(r["text"], add_special_tokens=False))
        for r in tqdm(sample_for_counting, desc="Counting tokens")
    )
    avg_tokens = sample_tokens / len(sample_for_counting)
    estimated_total = avg_tokens * len(all_sampled_rows)
    print(f"Estimated total tokens: {estimated_total/1e9:.2f}B (avg {avg_tokens:.1f} tokens/doc)")

    # Save to parquet files
    print("Saving to parquet files...")
    rows_per_file = 500_000

    output_data_dir = os.path.join(output_dir, "data", "sampled")
    os.makedirs(output_data_dir, exist_ok=True)

    n_files = (len(all_sampled_rows) + rows_per_file - 1) // rows_per_file

    for file_idx in tqdm(range(n_files), desc="Writing files"):
        start_idx = file_idx * rows_per_file
        end_idx = min((file_idx + 1) * rows_per_file, len(all_sampled_rows))

        chunk = all_sampled_rows[start_idx:end_idx]

        table = pa.table({
            "text": [r["text"] for r in chunk],
            "id": [r["id"] for r in chunk],
            "metadata": [r["metadata"] for r in chunk],
        })

        output_path = os.path.join(
            output_data_dir,
            f"train-{file_idx:05d}-of-{n_files:05d}.parquet"
        )
        pq.write_table(table, output_path)

    # Calculate source statistics
    print("\nCalculating source statistics...")
    source_stats: Dict[str, Dict] = defaultdict(lambda: {"docs": 0, "tokens_estimate": 0})

    for row in all_sampled_rows:
        canonical = row["_source"]
        source_stats[canonical]["docs"] += 1

    # Estimate tokens per source
    for src in source_stats:
        source_stats[src]["tokens_estimate"] = int(source_stats[src]["docs"] * avg_tokens)

    # Write summary
    summary_path = os.path.join(output_dir, "sampling_summary.txt")
    with open(summary_path, "w") as f:
        f.write("HinMix 31B Token Sample Summary (CulturaX Consensus-Checked)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total documents: {len(all_sampled_rows):,}\n")
        f.write(f"Estimated total tokens: {estimated_total/1e9:.2f}B\n\n")
        f.write("Filtering: CulturaX docs kept only if they appear in consensus\n")
        f.write("(i.e., they have duplicates in other sources)\n\n")
        f.write("Source breakdown:\n")

        total_docs = sum(s["docs"] for s in source_stats.values())
        for src, stats in sorted(source_stats.items(), key=lambda x: -x[1]["docs"]):
            pct = 100 * stats["docs"] / total_docs
            f.write(f"  {src}: {stats['docs']:,} docs ({pct:.1f}%)\n")

    print(f"Summary written to {summary_path}")
    print("\nDone!")

    return dict(source_stats)


def main():
    parser = argparse.ArgumentParser(
        description="Sample ~31B tokens from HinMix dataset (CulturaX consensus-checked)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="~/data/pretraining_data/hinmix-minhash-deduped",
        help="Input dataset directory"
    )
    parser.add_argument(
        "--consensus-dir",
        type=str,
        default="~/data/pretraining_data/hinmix-consensus",
        help="Consensus dataset directory (for CulturaX checking)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/data/hinmix-minhash-no_culturax_consensus_checked-31bt",
        help="Output directory for sampled dataset"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="google/gemma-2b",
        help="Path or name of tokenizer"
    )
    parser.add_argument(
        "--target-tokens",
        type=float,
        default=33.0,
        help="Target tokens in billions (default: 33B for ~31B final)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of documents to sample per source for token estimation"
    )

    args = parser.parse_args()

    # Expand user paths
    input_dir = os.path.expanduser(args.input_dir)
    consensus_dir = os.path.expanduser(args.consensus_dir)
    output_dir = os.path.expanduser(args.output_dir)

    target_tokens = int(args.target_tokens * 1e9)

    print("=" * 70)
    print("HinMix 31B Token Sampler (CulturaX Consensus-Checked)")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Consensus directory: {consensus_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Target tokens: {target_tokens/1e9:.1f}B")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    print("\nFiltering logic:")
    print("  - KEEP: Non-CulturaX documents")
    print("  - KEEP: CulturaX documents that appear in consensus")
    print("  - DROP: CulturaX documents NOT in consensus (unique to CulturaX)")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # Build consensus hash set
    consensus_hashes = build_consensus_text_set(consensus_dir)

    # Estimate tokens per source
    avg_tokens, doc_counts, total_culturax, kept_culturax = estimate_tokens_per_source(
        input_dir,
        consensus_hashes,
        tokenizer,
        sample_size=args.sample_size
    )

    # Calculate sampling plan
    sampling_plan, source_ratios = calculate_sampling_plan(
        doc_counts,
        avg_tokens,
        target_tokens
    )

    # Sample and save
    source_stats = sample_and_save(
        input_dir,
        output_dir,
        sampling_plan,
        consensus_hashes,
        tokenizer,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
