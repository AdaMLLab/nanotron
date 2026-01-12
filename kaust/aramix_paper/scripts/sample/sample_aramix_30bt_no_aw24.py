#!/usr/bin/env python3
"""
Script to sample ~30B tokens from the Aramix MinHash deduped dataset
while maintaining the specified source ratios.

IMPORTANT: This script EXCLUDES ArabicWeb24 and redistributes its proportion
to the remaining sources.

Target ratios (after excluding ArabicWeb24 and redistributing):
- CulturaX: 29.55%
- HPLT 2.0: 24.31%
- FineWeb-2: 19.33%
- C4: 15.84%
- 101B Arabic Words: 6.61%
- FinePDFs: 4.36%
"""

import argparse
import glob
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

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

# Sources to EXCLUDE from sampling
EXCLUDED_SOURCES = {"ArabicWeb24"}

# Target ratios (percentages) - EXCLUDING ArabicWeb24
# Original ratios from MinHash column, redistributed after removing ArabicWeb24 (19.9%)
# Calculation: new_ratio = original_ratio / 80.2 * 100
TARGET_RATIOS = {
    "CulturaX": 29.55,       # was 23.7%
    "HPLT 2.0": 24.31,       # was 19.5%
    "FineWeb-2": 19.33,      # was 15.5%
    "C4": 15.84,             # was 12.7%
    "101B Arabic Words": 6.61,  # was 5.3%
    "FinePDFs": 4.36,        # was 3.5%
}

# Target total tokens (30B with a small buffer)
TARGET_TOKENS = 31_500_000_000  # 31.5B to ensure we get just over 30B


def normalize_source(source: str) -> str:
    """Normalize source name to canonical form."""
    if source in SOURCE_MAPPING:
        return SOURCE_MAPPING[source]
    # Try case-insensitive match
    source_lower = source.lower()
    for key, val in SOURCE_MAPPING.items():
        if key.lower() == source_lower:
            return val
    # Return as-is if no mapping found
    return source


def is_excluded_source(source: str) -> bool:
    """Check if a source should be excluded."""
    canonical = normalize_source(source)
    return canonical in EXCLUDED_SOURCES


def estimate_tokens_per_source(
    data_dir: str,
    tokenizer,
    sample_size: int = 1000
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Estimate average tokens per document for each source by sampling.

    Returns:
        avg_tokens_per_doc: Dict mapping source -> average tokens per document
        doc_counts: Dict mapping source -> total document count
    """
    print("Estimating tokens per document for each source...")

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} parquet files")

    # Collect documents by source
    docs_by_source: Dict[str, List[str]] = defaultdict(list)
    doc_counts: Dict[str, int] = defaultdict(int)

    # First pass: count total documents per source
    print("Counting documents per source...")
    excluded_count = 0
    for pf_path in tqdm(parquet_files, desc="Counting"):
        table = pq.read_table(pf_path, columns=["source"])
        sources = table.column("source").to_pylist()
        for src in sources:
            canonical = normalize_source(src)
            if canonical in EXCLUDED_SOURCES:
                excluded_count += 1
                continue
            doc_counts[canonical] += 1

    print(f"\nExcluded {excluded_count:,} documents from excluded sources (ArabicWeb24)")
    print("\nDocument counts by source (included only):")
    for src, count in sorted(doc_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count:,}")

    # Second pass: sample documents for token estimation
    print(f"\nSampling {sample_size} documents per source for token estimation...")

    # Calculate sampling rate per source
    sample_needs: Dict[str, int] = {src: sample_size for src in doc_counts.keys()}

    for pf_path in tqdm(parquet_files, desc="Sampling"):
        if all(n <= 0 for n in sample_needs.values()):
            break

        table = pq.read_table(pf_path, columns=["text", "source"])
        df = table.to_pandas()

        for src, group in df.groupby("source"):
            canonical = normalize_source(src)
            if canonical in EXCLUDED_SOURCES:
                continue
            if sample_needs.get(canonical, 0) > 0:
                n_to_sample = min(sample_needs[canonical], len(group))
                sampled = group.sample(n=n_to_sample, random_state=42)
                docs_by_source[canonical].extend(sampled["text"].tolist())
                sample_needs[canonical] -= n_to_sample

    # Tokenize samples to estimate average tokens per document
    avg_tokens_per_doc: Dict[str, float] = {}

    print("\nTokenizing samples to estimate average tokens...")
    for src, docs in tqdm(docs_by_source.items(), desc="Tokenizing"):
        if not docs:
            avg_tokens_per_doc[src] = 0
            continue

        total_tokens = 0
        for doc in docs[:sample_size]:  # Limit to sample_size
            tokens = tokenizer.encode(doc, add_special_tokens=False)
            total_tokens += len(tokens)

        avg_tokens_per_doc[src] = total_tokens / len(docs[:sample_size])
        print(f"  {src}: avg {avg_tokens_per_doc[src]:.1f} tokens/doc ({len(docs[:sample_size])} samples)")

    return avg_tokens_per_doc, dict(doc_counts)


def calculate_sampling_plan(
    doc_counts: Dict[str, int],
    avg_tokens: Dict[str, float],
    target_tokens: int,
    target_ratios: Dict[str, float]
) -> Dict[str, int]:
    """
    Calculate how many documents to sample from each source.

    Returns:
        Dict mapping source -> number of documents to sample
    """
    print("\nCalculating sampling plan...")

    # Normalize target ratios to only include available sources
    available_sources = set(doc_counts.keys())
    available_ratios = {k: v for k, v in target_ratios.items() if k in available_sources}

    # Check for missing sources
    missing_sources = set(target_ratios.keys()) - available_sources
    if missing_sources:
        print(f"Warning: These sources are not available in the dataset: {missing_sources}")
        print("Redistributing their ratios proportionally...")

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

        # Cap at available documents
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
    tokenizer,
    seed: int = 42
) -> None:
    """
    Randomly sample documents according to the plan and save to output directory.
    """
    print(f"\nSampling documents and saving to {output_dir}...")

    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))

    # First pass: collect all indices per source
    print("Collecting document indices per source...")
    source_indices: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    for pf_path in tqdm(parquet_files, desc="Indexing"):
        table = pq.read_table(pf_path, columns=["source"])
        sources = table.column("source").to_pylist()

        for idx, src in enumerate(sources):
            canonical = normalize_source(src)
            if canonical in EXCLUDED_SOURCES:
                continue
            source_indices[canonical].append((pf_path, idx))

    # Randomly sample indices for each source
    print("Randomly sampling indices...")
    sampled_indices: Dict[str, set] = {}

    for src, indices in source_indices.items():
        n_to_sample = sampling_plan.get(src, 0)
        if n_to_sample > 0 and indices:
            sampled = random.sample(indices, min(n_to_sample, len(indices)))
            # Group by file
            sampled_by_file: Dict[str, set] = defaultdict(set)
            for pf_path, idx in sampled:
                sampled_by_file[pf_path].add(idx)
            sampled_indices[src] = sampled_by_file
            print(f"  {src}: sampled {len(sampled):,} documents")

    # Flatten sampled indices by file for efficient reading
    file_indices: Dict[str, set] = defaultdict(set)
    for src, by_file in sampled_indices.items():
        for pf_path, indices in by_file.items():
            file_indices[pf_path].update(indices)

    # Read and collect sampled documents
    print("Reading sampled documents...")
    all_sampled_rows = []
    actual_token_count = 0

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

    # Shuffle the sampled documents
    print("Shuffling documents...")
    random.shuffle(all_sampled_rows)

    # Count actual tokens (sample-based estimation)
    print("Estimating final token count...")
    sample_for_counting = random.sample(all_sampled_rows, min(5000, len(all_sampled_rows)))
    sample_tokens = sum(len(tokenizer.encode(r["text"], add_special_tokens=False)) for r in tqdm(sample_for_counting, desc="Counting tokens"))
    avg_tokens = sample_tokens / len(sample_for_counting)
    estimated_total = avg_tokens * len(all_sampled_rows)
    print(f"Estimated total tokens: {estimated_total/1e9:.2f}B (avg {avg_tokens:.1f} tokens/doc)")

    # Save to parquet files (split into manageable chunks)
    print("Saving to parquet files...")
    rows_per_file = 500_000  # 500k rows per file

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

    # Write summary statistics
    print("\nWriting summary statistics...")
    source_stats: Dict[str, Dict] = defaultdict(lambda: {"docs": 0, "tokens_estimate": 0})

    for row in all_sampled_rows:
        canonical = normalize_source(row["source"])
        source_stats[canonical]["docs"] += 1

    # Estimate tokens per source
    for src in source_stats:
        source_stats[src]["tokens_estimate"] = int(source_stats[src]["docs"] * avg_tokens)

    summary_path = os.path.join(output_dir, "sampling_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Aramix 30B Token Sample Summary (No ArabicWeb24)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total documents: {len(all_sampled_rows):,}\n")
        f.write(f"Estimated total tokens: {estimated_total/1e9:.2f}B\n\n")
        f.write("EXCLUDED SOURCES: ArabicWeb24\n\n")
        f.write("Source breakdown:\n")

        total_docs = sum(s["docs"] for s in source_stats.values())
        for src, stats in sorted(source_stats.items(), key=lambda x: -x[1]["docs"]):
            pct = 100 * stats["docs"] / total_docs
            f.write(f"  {src}: {stats['docs']:,} docs ({pct:.1f}%)\n")

    print(f"Summary written to {summary_path}")
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Sample ~30B tokens from Aramix dataset (excluding ArabicWeb24) with target source ratios"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/alrashsm/data/pretraining_data/aramix-minhash-deduped",
        help="Input dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/alrashsm/data/aramix_paper/aramix-minhash-no_aw24-30bt/",
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

    print("=" * 60)
    print("Aramix 30B Token Sampler (No ArabicWeb24)")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Target tokens: {target_tokens/1e9:.1f}B")
    print(f"Random seed: {args.seed}")
    print(f"EXCLUDED SOURCES: {EXCLUDED_SOURCES}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # Estimate tokens per source
    avg_tokens, doc_counts = estimate_tokens_per_source(
        args.input_dir,
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
        tokenizer,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
