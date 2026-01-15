#!/usr/bin/env python3
"""
Script to sample ~30B tokens from the Aramix MinHash deduped dataset,
filtering ArabicWeb24 documents that don't appear in the consensus dataset.

Filtering logic:
- KEEP: Documents from non-ArabicWeb24 sources
- KEEP: ArabicWeb24 documents that appear in aramix-consensus (have duplicates in other sources)
- DROP: ArabicWeb24 documents that don't appear in aramix-consensus (unique to ArabicWeb24)

This approach keeps ArabicWeb24 content that is corroborated by other sources.
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
    "uonlp/CulturaX": "CulturaX",
    "CulturaX": "CulturaX",
    "lightonai/ArabicWeb24": "ArabicWeb24",
    "ArabicWeb24": "ArabicWeb24",
    "allenai/c4": "C4",
    "C4": "C4",
    "ClusterlabAi/101_billion_arabic_words_dataset": "101B Arabic Words",
    "101B Arabic Words": "101B Arabic Words",
    # HPLT variants
    "HPLT/HPLT2.0_cleaned": "HPLT 2.0",
    "hplt/HPLT": "HPLT 2.0",
    "HPLT": "HPLT 2.0",
    "HPLT 2.0": "HPLT 2.0",
    # FineWeb-2 variants
    "HuggingFaceFW/fineweb-2": "FineWeb-2",
    "fineweb-2": "FineWeb-2",
    "FineWeb-2": "FineWeb-2",
    # FinePDFs variants
    "HuggingFaceFW/finepdfs": "FinePDFs",
    "FinePDFs": "FinePDFs",
    "finepdfs": "FinePDFs",
}

# Target ratios (percentages) - EXCLUDING unique ArabicWeb24
# Original ratios from MinHash column, redistributed after removing ArabicWeb24's unique portion
# We keep ArabicWeb24 docs that appear in consensus, so actual ratios will depend on data
TARGET_RATIOS = {
    "CulturaX": 29.55,
    "HPLT 2.0": 24.31,
    "FineWeb-2": 19.33,
    "C4": 15.84,
    "101B Arabic Words": 6.61,
    "FinePDFs": 4.36,
}

# Target total tokens (30B with a small buffer)
TARGET_TOKENS = 31_500_000_000


def normalize_source(source: str) -> str:
    """Normalize source name to canonical form."""
    if source in SOURCE_MAPPING:
        return SOURCE_MAPPING[source]
    source_lower = source.lower()
    for key, val in SOURCE_MAPPING.items():
        if key.lower() == source_lower:
            return val
    return source


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


def is_arabicweb24(source: str) -> bool:
    """Check if source is ArabicWeb24."""
    canonical = normalize_source(source)
    return canonical == "ArabicWeb24"


def estimate_tokens_per_source(
    data_dir: str,
    consensus_hashes: Set[str],
    tokenizer,
    sample_size: int = 1000
) -> Tuple[Dict[str, float], Dict[str, int], int, int]:
    """
    Estimate average tokens per document for each source by sampling.
    Also counts eligible documents (non-AW24 or AW24 in consensus).

    Returns:
        avg_tokens_per_doc: Dict mapping source -> average tokens per document
        doc_counts: Dict mapping source -> eligible document count
        total_aw24_docs: Total ArabicWeb24 documents
        kept_aw24_docs: ArabicWeb24 documents kept (in consensus)
    """
    print("Estimating tokens per document for each source...")

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} parquet files")

    docs_by_source: Dict[str, List[str]] = defaultdict(list)
    doc_counts: Dict[str, int] = defaultdict(int)
    total_aw24_docs = 0
    kept_aw24_docs = 0
    dropped_aw24_docs = 0

    print("Counting documents per source (with ArabicWeb24 consensus check)...")
    for pf_path in tqdm(parquet_files, desc="Counting"):
        table = pq.read_table(pf_path, columns=["source", "text"])
        sources = table.column("source").to_pylist()
        texts = table.column("text").to_pylist()

        for src, text in zip(sources, texts):
            canonical = normalize_source(src)

            if canonical == "ArabicWeb24":
                total_aw24_docs += 1
                # Check if in consensus
                if text_hash(text) in consensus_hashes:
                    kept_aw24_docs += 1
                    doc_counts[canonical] += 1
                else:
                    dropped_aw24_docs += 1
                    continue  # Skip this document
            else:
                doc_counts[canonical] += 1

    print(f"\nArabicWeb24 statistics:")
    print(f"  Total AW24 documents: {total_aw24_docs:,}")
    print(f"  Kept (in consensus): {kept_aw24_docs:,} ({100*kept_aw24_docs/total_aw24_docs:.1f}%)")
    print(f"  Dropped (unique): {dropped_aw24_docs:,} ({100*dropped_aw24_docs/total_aw24_docs:.1f}%)")

    print("\nDocument counts by source (eligible only):")
    for src, count in sorted(doc_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count:,}")

    # Second pass: sample documents for token estimation
    print(f"\nSampling {sample_size} documents per source for token estimation...")
    sample_needs: Dict[str, int] = {src: sample_size for src in doc_counts.keys()}

    for pf_path in tqdm(parquet_files, desc="Sampling"):
        if all(n <= 0 for n in sample_needs.values()):
            break

        table = pq.read_table(pf_path, columns=["text", "source"])
        df = table.to_pandas()

        for src, group in df.groupby("source"):
            canonical = normalize_source(src)

            if sample_needs.get(canonical, 0) > 0:
                # For ArabicWeb24, only sample from consensus-approved docs
                if canonical == "ArabicWeb24":
                    eligible_mask = group["text"].apply(lambda t: text_hash(t) in consensus_hashes)
                    eligible_group = group[eligible_mask]
                else:
                    eligible_group = group

                if len(eligible_group) > 0:
                    n_to_sample = min(sample_needs[canonical], len(eligible_group))
                    sampled = eligible_group.sample(n=n_to_sample, random_state=42)
                    docs_by_source[canonical].extend(sampled["text"].tolist())
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

    return avg_tokens_per_doc, dict(doc_counts), total_aw24_docs, kept_aw24_docs


def calculate_sampling_plan(
    doc_counts: Dict[str, int],
    avg_tokens: Dict[str, float],
    target_tokens: int,
    target_ratios: Dict[str, float]
) -> Dict[str, int]:
    """
    Calculate how many documents to sample from each source.
    """
    print("\nCalculating sampling plan...")

    available_sources = set(doc_counts.keys())

    # Check if ArabicWeb24 is in available sources (kept docs)
    # If so, we need to decide how to handle its ratio
    # For now, we'll just sample proportionally based on available docs

    # Use target ratios for non-AW24 sources, and include AW24 kept docs proportionally
    available_ratios = {k: v for k, v in target_ratios.items() if k in available_sources}

    # If ArabicWeb24 kept docs exist, add them with a proportion based on their count
    if "ArabicWeb24" in available_sources and "ArabicWeb24" not in available_ratios:
        # Calculate what percentage AW24 represents of total eligible docs
        total_eligible = sum(doc_counts.values())
        aw24_pct = 100 * doc_counts["ArabicWeb24"] / total_eligible
        available_ratios["ArabicWeb24"] = aw24_pct
        print(f"Adding ArabicWeb24 (consensus-approved) with {aw24_pct:.2f}% of eligible docs")

    missing_sources = set(target_ratios.keys()) - available_sources
    if missing_sources:
        print(f"Warning: These sources are not available: {missing_sources}")

    # Renormalize ratios
    total_ratio = sum(available_ratios.values())
    normalized_ratios = {k: v / total_ratio * 100 for k, v in available_ratios.items()}

    print("\nNormalized target ratios:")
    for src, ratio in sorted(normalized_ratios.items(), key=lambda x: -x[1]):
        print(f"  {src}: {ratio:.2f}%")

    # Calculate target tokens per source
    target_tokens_per_source = {
        src: target_tokens * (ratio / 100)
        for src, ratio in normalized_ratios.items()
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

    return docs_to_sample


def sample_and_save(
    data_dir: str,
    output_dir: str,
    sampling_plan: Dict[str, int],
    consensus_hashes: Set[str],
    tokenizer,
    seed: int = 42
) -> None:
    """
    Sample documents according to plan and save to output directory.
    ArabicWeb24 documents are only included if they appear in consensus.
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
        table = pq.read_table(pf_path, columns=["source", "text"])
        sources = table.column("source").to_pylist()
        texts = table.column("text").to_pylist()

        for idx, (src, text) in enumerate(zip(sources, texts)):
            canonical = normalize_source(src)

            # For ArabicWeb24, only include if in consensus
            if canonical == "ArabicWeb24":
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

        indices_to_keep = file_indices[pf_path]
        table = pq.read_table(pf_path)

        for idx in sorted(indices_to_keep):
            row = {
                "text": table.column("text")[idx].as_py(),
                "id": table.column("id")[idx].as_py(),
                "source": table.column("source")[idx].as_py(),
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
            "source": [r["source"] for r in chunk],
        })

        output_path = os.path.join(
            output_data_dir,
            f"train-{file_idx:05d}-of-{n_files:05d}.parquet"
        )
        pq.write_table(table, output_path)

    # Write summary
    print("\nWriting summary statistics...")
    source_stats: Dict[str, Dict] = defaultdict(lambda: {"docs": 0})

    for row in all_sampled_rows:
        canonical = normalize_source(row["source"])
        source_stats[canonical]["docs"] += 1

    summary_path = os.path.join(output_dir, "sampling_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Aramix 30B Token Sample Summary (ArabicWeb24 Consensus-Checked)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total documents: {len(all_sampled_rows):,}\n")
        f.write(f"Estimated total tokens: {estimated_total/1e9:.2f}B\n\n")
        f.write("Filtering: ArabicWeb24 docs kept only if they appear in consensus\n")
        f.write("(i.e., they have duplicates in other sources)\n\n")
        f.write("Source breakdown:\n")

        total_docs = sum(s["docs"] for s in source_stats.values())
        for src, stats in sorted(source_stats.items(), key=lambda x: -x[1]["docs"]):
            pct = 100 * stats["docs"] / total_docs
            f.write(f"  {src}: {stats['docs']:,} docs ({pct:.1f}%)\n")

    print(f"Summary written to {summary_path}")
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Sample ~30B tokens from Aramix dataset (ArabicWeb24 consensus-checked)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="~/data/pretraining_data/aramix-minhash-deduped",
        help="Input dataset directory"
    )
    parser.add_argument(
        "--consensus-dir",
        type=str,
        default="~/data/aramix_paper/aramix-consensus",
        help="Consensus dataset directory (for ArabicWeb24 checking)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/data/aramix_paper/aramix-minhash-no_aw24_consensus_checked-30bt",
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
        default=31.5,
        help="Target tokens in billions (default: 31.5B)"
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

    target_tokens = int(args.target_tokens * 1e9)

    print("=" * 70)
    print("Aramix 30B Token Sampler (ArabicWeb24 Consensus-Checked)")
    print("=" * 70)
    print(f"Input directory: {args.input_dir}")
    print(f"Consensus directory: {args.consensus_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Target tokens: {target_tokens/1e9:.1f}B")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    print("\nFiltering logic:")
    print("  - KEEP: Non-ArabicWeb24 documents")
    print("  - KEEP: ArabicWeb24 documents that appear in consensus")
    print("  - DROP: ArabicWeb24 documents NOT in consensus (unique to AW24)")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # Build consensus hash set
    consensus_hashes = build_consensus_text_set(args.consensus_dir)

    # Estimate tokens per source
    avg_tokens, doc_counts, total_aw24, kept_aw24 = estimate_tokens_per_source(
        args.input_dir,
        consensus_hashes,
        tokenizer,
        sample_size=args.sample_size
    )

    # Calculate sampling plan
    sampling_plan = calculate_sampling_plan(
        doc_counts,
        avg_tokens,
        target_tokens,
        TARGET_RATIOS
    )

    # Sample and save
    sample_and_save(
        args.input_dir,
        args.output_dir,
        sampling_plan,
        consensus_hashes,
        tokenizer,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
