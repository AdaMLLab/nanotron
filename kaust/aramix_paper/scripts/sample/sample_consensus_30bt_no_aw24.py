#!/usr/bin/env python3
"""
Script to sample ~30B tokens from the Aramix Consensus dataset
while filtering out documents that only have ArabicWeb24 + one other source.

Filtering rules:
- DROP: Documents with exactly 2 sources where one is 'arabicweb24'
- KEEP: Documents with 'arabicweb24' + 2 or more other sources (3+ total)
- KEEP: Documents without 'arabicweb24'
"""

import argparse
import glob
import os
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer


# Target total tokens (30B with a small buffer)
TARGET_TOKENS = 31_500_000_000  # 31.5B to ensure we get just over 30B

# ArabicWeb24 identifier (lowercase as seen in data)
ARABICWEB24 = "arabicweb24"


def should_keep_document(sources: List[str]) -> bool:
    """
    Determine if a document should be kept based on its sources.

    Rules:
    - DROP: Exactly 2 sources where one is 'arabicweb24'
    - KEEP: 'arabicweb24' + 2 or more other sources (3+ total)
    - KEEP: No 'arabicweb24' in sources
    """
    if ARABICWEB24 not in sources:
        # No arabicweb24 -> keep
        return True

    # Has arabicweb24
    if len(sources) >= 3:
        # 3+ sources means 2+ non-arabicweb24 sources -> keep
        return True

    # Has arabicweb24 and only 2 sources total -> drop
    return False


def count_documents_and_estimate_tokens(
    data_dir: str,
    tokenizer,
    sample_size: int = 5000
) -> Tuple[int, int, float, Dict[str, int]]:
    """
    Count total and eligible documents, estimate tokens per document.

    Returns:
        total_docs: Total document count
        eligible_docs: Documents that pass the filter
        avg_tokens_per_doc: Average tokens per document (from sampling)
        source_counts: Dict mapping source -> count of documents containing that source
    """
    print("Counting documents and estimating tokens...")

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} parquet files")

    total_docs = 0
    eligible_docs = 0
    dropped_docs = 0
    source_counts: Dict[str, int] = defaultdict(int)

    # Sample documents for token estimation
    sampled_texts: List[str] = []

    print("Counting documents and checking eligibility...")
    for pf_path in tqdm(parquet_files, desc="Counting"):
        table = pq.read_table(pf_path)
        sources_col = table.column("source").to_pylist()
        texts_col = table.column("text").to_pylist()

        for i, sources in enumerate(sources_col):
            total_docs += 1

            if should_keep_document(sources):
                eligible_docs += 1

                # Count sources for eligible documents
                for src in sources:
                    source_counts[src] += 1

                # Sample for token estimation
                if len(sampled_texts) < sample_size:
                    sampled_texts.append(texts_col[i])
                elif random.random() < sample_size / eligible_docs:
                    # Reservoir sampling to get uniform sample
                    idx = random.randint(0, sample_size - 1)
                    sampled_texts[idx] = texts_col[i]
            else:
                dropped_docs += 1

    print(f"\nDocument counts:")
    print(f"  Total documents: {total_docs:,}")
    print(f"  Eligible documents: {eligible_docs:,}")
    print(f"  Dropped documents: {dropped_docs:,} ({100*dropped_docs/total_docs:.1f}%)")

    print(f"\nSource distribution (in eligible documents):")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / eligible_docs
        print(f"  {src}: {count:,} ({pct:.1f}%)")

    # Estimate tokens per document
    print(f"\nTokenizing {len(sampled_texts)} samples to estimate tokens per document...")
    total_tokens = 0
    for text in tqdm(sampled_texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)

    avg_tokens = total_tokens / len(sampled_texts) if sampled_texts else 0
    print(f"Average tokens per document: {avg_tokens:.1f}")

    estimated_total = avg_tokens * eligible_docs
    print(f"Estimated total tokens in eligible docs: {estimated_total/1e9:.2f}B")

    return total_docs, eligible_docs, avg_tokens, dict(source_counts)


def calculate_sampling_rate(
    eligible_docs: int,
    avg_tokens: float,
    target_tokens: int
) -> float:
    """
    Calculate what fraction of documents to sample.
    """
    estimated_total = eligible_docs * avg_tokens

    if estimated_total <= target_tokens:
        print(f"Warning: Eligible documents ({estimated_total/1e9:.2f}B tokens) "
              f"are less than target ({target_tokens/1e9:.2f}B). Using all documents.")
        return 1.0

    sampling_rate = target_tokens / estimated_total
    print(f"\nSampling rate: {sampling_rate:.4f} ({sampling_rate*100:.2f}%)")
    print(f"Expected documents to sample: {int(eligible_docs * sampling_rate):,}")

    return sampling_rate


def sample_and_save(
    data_dir: str,
    output_dir: str,
    sampling_rate: float,
    tokenizer,
    seed: int = 42
) -> None:
    """
    Filter and sample documents, then save to output directory.
    """
    print(f"\nSampling documents and saving to {output_dir}...")

    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))

    # Collect all eligible document indices
    print("Collecting eligible document indices...")
    eligible_indices: List[Tuple[str, int]] = []

    for pf_path in tqdm(parquet_files, desc="Indexing"):
        table = pq.read_table(pf_path, columns=["source"])
        sources_col = table.column("source").to_pylist()

        for idx, sources in enumerate(sources_col):
            if should_keep_document(sources):
                eligible_indices.append((pf_path, idx))

    print(f"Total eligible documents: {len(eligible_indices):,}")

    # Sample indices
    n_to_sample = int(len(eligible_indices) * sampling_rate)
    print(f"Sampling {n_to_sample:,} documents...")

    sampled_indices = random.sample(eligible_indices, min(n_to_sample, len(eligible_indices)))

    # Group by file for efficient reading
    indices_by_file: Dict[str, Set[int]] = defaultdict(set)
    for pf_path, idx in sampled_indices:
        indices_by_file[pf_path].add(idx)

    # Read and collect sampled documents
    print("Reading sampled documents...")
    all_sampled_rows = []

    for pf_path in tqdm(parquet_files, desc="Reading"):
        if pf_path not in indices_by_file:
            continue

        indices_to_keep = indices_by_file[pf_path]
        table = pq.read_table(pf_path)

        for idx in sorted(indices_to_keep):
            row = {
                "text": table.column("text")[idx].as_py(),
                "source": table.column("source")[idx].as_py(),
            }
            # Include 'duplicated' column if it exists
            if "duplicated" in table.column_names:
                row["duplicated"] = table.column("duplicated")[idx].as_py()
            all_sampled_rows.append(row)

    print(f"\nTotal sampled documents: {len(all_sampled_rows):,}")

    # Shuffle the sampled documents
    print("Shuffling documents...")
    random.shuffle(all_sampled_rows)

    # Count actual tokens (sample-based estimation)
    print("Estimating final token count...")
    sample_for_counting = random.sample(all_sampled_rows, min(5000, len(all_sampled_rows)))
    sample_tokens = sum(
        len(tokenizer.encode(r["text"], add_special_tokens=False))
        for r in tqdm(sample_for_counting, desc="Counting tokens")
    )
    avg_tokens = sample_tokens / len(sample_for_counting)
    estimated_total = avg_tokens * len(all_sampled_rows)
    print(f"Estimated total tokens: {estimated_total/1e9:.2f}B (avg {avg_tokens:.1f} tokens/doc)")

    # Save to parquet files (split into manageable chunks)
    print("Saving to parquet files...")
    rows_per_file = 500_000  # 500k rows per file

    output_data_dir = os.path.join(output_dir, "data")
    os.makedirs(output_data_dir, exist_ok=True)

    n_files = (len(all_sampled_rows) + rows_per_file - 1) // rows_per_file

    for file_idx in tqdm(range(n_files), desc="Writing files"):
        start_idx = file_idx * rows_per_file
        end_idx = min((file_idx + 1) * rows_per_file, len(all_sampled_rows))

        chunk = all_sampled_rows[start_idx:end_idx]

        # Build table columns
        table_data = {
            "text": [r["text"] for r in chunk],
            "source": [r["source"] for r in chunk],
        }
        if "duplicated" in chunk[0]:
            table_data["duplicated"] = [r["duplicated"] for r in chunk]

        table = pa.table(table_data)

        output_path = os.path.join(
            output_data_dir,
            f"train-{file_idx:05d}-of-{n_files:05d}.parquet"
        )
        pq.write_table(table, output_path)

    # Write summary statistics
    print("\nWriting summary statistics...")
    source_stats: Dict[str, int] = defaultdict(int)

    for row in all_sampled_rows:
        for src in row["source"]:
            source_stats[src] += 1

    summary_path = os.path.join(output_dir, "sampling_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Aramix Consensus 30B Token Sample Summary (No ArabicWeb24-only pairs)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total documents: {len(all_sampled_rows):,}\n")
        f.write(f"Estimated total tokens: {estimated_total/1e9:.2f}B\n\n")
        f.write("Filtering rule: Dropped docs with exactly 2 sources where one is 'arabicweb24'\n\n")
        f.write("Source distribution (documents containing each source):\n")

        for src, count in sorted(source_stats.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(all_sampled_rows)
            f.write(f"  {src}: {count:,} ({pct:.1f}%)\n")

    print(f"Summary written to {summary_path}")
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Sample ~30B tokens from Aramix Consensus dataset (filtering ArabicWeb24-only pairs)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="~/data/aramix_paper/aramix-consensus",
        help="Input dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/data/aramix_paper/aramix-consensus-no_aw24-30bt",
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
        default=5000,
        help="Number of documents to sample for token estimation"
    )

    args = parser.parse_args()

    target_tokens = int(args.target_tokens * 1e9)

    print("=" * 70)
    print("Aramix Consensus 30B Token Sampler (No ArabicWeb24-only pairs)")
    print("=" * 70)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Target tokens: {target_tokens/1e9:.1f}B")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    print("\nFiltering rule:")
    print("  - DROP: Documents with exactly 2 sources where one is 'arabicweb24'")
    print("  - KEEP: Documents with 'arabicweb24' + 2+ other sources")
    print("  - KEEP: Documents without 'arabicweb24'")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # Count documents and estimate tokens
    total_docs, eligible_docs, avg_tokens, source_counts = count_documents_and_estimate_tokens(
        args.input_dir,
        tokenizer,
        sample_size=args.sample_size
    )

    # Calculate sampling rate
    sampling_rate = calculate_sampling_rate(
        eligible_docs,
        avg_tokens,
        target_tokens
    )

    # Sample and save
    sample_and_save(
        args.input_dir,
        args.output_dir,
        sampling_rate,
        tokenizer,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
