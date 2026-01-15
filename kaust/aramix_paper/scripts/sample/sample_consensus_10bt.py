#!/usr/bin/env python3
"""
Script to sample ~10B tokens from the Aramix Consensus dataset
while maintaining the specified duplicated ratios.

Target ratios (based on document counts for duplicated 2-4 only):
- duplicated=2: 60,466,799 docs (46.6%)
- duplicated=3: 43,200,076 docs (33.3%)
- duplicated=4: 26,015,836 docs (20.1%)

Documents with duplicated=5,6,7 are filtered out.
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


# Target document counts for each duplicated value (used to compute ratios)
TARGET_DOC_COUNTS = {
    2: 60_466_799,
    3: 43_200_076,
    4: 26_015_836,
}

# Compute target ratios as percentages
_total_docs = sum(TARGET_DOC_COUNTS.values())
TARGET_RATIOS = {k: 100 * v / _total_docs for k, v in TARGET_DOC_COUNTS.items()}

# Valid duplicated values to include (filter out 5, 6, 7)
VALID_DUPLICATED_VALUES = {2, 3, 4}

# Target total tokens (10B with a small buffer)
TARGET_TOKENS = 10_500_000_000  # 10.5B to ensure we get just over 10B


def estimate_tokens_per_duplicated(
    data_dir: str,
    tokenizer,
    sample_size: int = 1000
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    Estimate average tokens per document for each duplicated value by sampling.

    Returns:
        avg_tokens_per_doc: Dict mapping duplicated value -> average tokens per document
        doc_counts: Dict mapping duplicated value -> total document count
    """
    print("Estimating tokens per document for each duplicated value...")

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} parquet files")

    # Collect documents by duplicated value
    docs_by_duplicated: Dict[int, List[str]] = defaultdict(list)
    doc_counts: Dict[int, int] = defaultdict(int)

    # First pass: count total documents per duplicated value
    print("Counting documents per duplicated value...")
    for pf_path in tqdm(parquet_files, desc="Counting"):
        table = pq.read_table(pf_path, columns=["duplicated"])
        duplicated_vals = table.column("duplicated").to_pylist()
        for dup in duplicated_vals:
            if dup in VALID_DUPLICATED_VALUES:
                doc_counts[dup] += 1

    print("\nDocument counts by duplicated value (filtered to 2-4):")
    for dup, count in sorted(doc_counts.items()):
        print(f"  duplicated={dup}: {count:,}")

    # Second pass: sample documents for token estimation
    print(f"\nSampling {sample_size} documents per duplicated value for token estimation...")

    # Calculate sampling rate per duplicated value
    sample_needs: Dict[int, int] = {dup: sample_size for dup in VALID_DUPLICATED_VALUES}

    for pf_path in tqdm(parquet_files, desc="Sampling"):
        if all(n <= 0 for n in sample_needs.values()):
            break

        table = pq.read_table(pf_path, columns=["text", "duplicated"])
        df = table.to_pandas()

        for dup, group in df.groupby("duplicated"):
            if dup in VALID_DUPLICATED_VALUES and sample_needs.get(dup, 0) > 0:
                n_to_sample = min(sample_needs[dup], len(group))
                sampled = group.sample(n=n_to_sample, random_state=42)
                docs_by_duplicated[dup].extend(sampled["text"].tolist())
                sample_needs[dup] -= n_to_sample

    # Tokenize samples to estimate average tokens per document
    avg_tokens_per_doc: Dict[int, float] = {}

    print("\nTokenizing samples to estimate average tokens...")
    for dup, docs in tqdm(sorted(docs_by_duplicated.items()), desc="Tokenizing"):
        if not docs:
            avg_tokens_per_doc[dup] = 0
            continue

        total_tokens = 0
        for doc in docs[:sample_size]:  # Limit to sample_size
            tokens = tokenizer.encode(doc, add_special_tokens=False)
            total_tokens += len(tokens)

        avg_tokens_per_doc[dup] = total_tokens / len(docs[:sample_size])
        print(f"  duplicated={dup}: avg {avg_tokens_per_doc[dup]:.1f} tokens/doc ({len(docs[:sample_size])} samples)")

    return avg_tokens_per_doc, dict(doc_counts)


def calculate_sampling_plan(
    doc_counts: Dict[int, int],
    avg_tokens: Dict[int, float],
    target_tokens: int,
    target_ratios: Dict[int, float]
) -> Dict[int, int]:
    """
    Calculate how many documents to sample from each duplicated value.

    Returns:
        Dict mapping duplicated value -> number of documents to sample
    """
    print("\nCalculating sampling plan...")

    # Normalize target ratios to only include available duplicated values
    available_values = set(doc_counts.keys())
    available_ratios = {k: v for k, v in target_ratios.items() if k in available_values}

    # Check for missing values
    missing_values = set(target_ratios.keys()) - available_values
    if missing_values:
        print(f"Warning: These duplicated values are not available: {missing_values}")
        print("Redistributing their ratios proportionally...")

    # Renormalize ratios
    total_ratio = sum(available_ratios.values())
    normalized_ratios = {k: v / total_ratio * 100 for k, v in available_ratios.items()}

    print("\nNormalized target ratios:")
    for dup, ratio in sorted(normalized_ratios.items()):
        print(f"  duplicated={dup}: {ratio:.2f}%")

    # Calculate target tokens per duplicated value
    target_tokens_per_value = {
        dup: target_tokens * (ratio / 100)
        for dup, ratio in normalized_ratios.items()
    }

    # Calculate documents needed per duplicated value
    docs_to_sample: Dict[int, int] = {}

    print("\nSampling plan:")
    total_estimated_tokens = 0

    for dup in sorted(available_values):
        if avg_tokens.get(dup, 0) == 0:
            docs_to_sample[dup] = 0
            continue

        target_dup_tokens = target_tokens_per_value.get(dup, 0)
        needed_docs = int(target_dup_tokens / avg_tokens[dup])

        # Cap at available documents
        available_docs = doc_counts[dup]
        actual_docs = min(needed_docs, available_docs)
        docs_to_sample[dup] = actual_docs

        estimated_tokens = actual_docs * avg_tokens[dup]
        total_estimated_tokens += estimated_tokens

        status = "" if actual_docs == needed_docs else f" (capped from {needed_docs:,})"
        print(f"  duplicated={dup}: {actual_docs:,} docs (~{estimated_tokens/1e9:.2f}B tokens){status}")

    print(f"\nTotal estimated tokens: {total_estimated_tokens/1e9:.2f}B")

    return docs_to_sample


def sample_and_save(
    data_dir: str,
    output_dir: str,
    sampling_plan: Dict[int, int],
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

    # First pass: collect all indices per duplicated value
    print("Collecting document indices per duplicated value...")
    duplicated_indices: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

    for pf_path in tqdm(parquet_files, desc="Indexing"):
        table = pq.read_table(pf_path, columns=["duplicated"])
        duplicated_vals = table.column("duplicated").to_pylist()

        for idx, dup in enumerate(duplicated_vals):
            if dup in VALID_DUPLICATED_VALUES:
                duplicated_indices[dup].append((pf_path, idx))

    # Randomly sample indices for each duplicated value
    print("Randomly sampling indices...")
    sampled_indices: Dict[int, Dict[str, set]] = {}

    for dup, indices in sorted(duplicated_indices.items()):
        n_to_sample = sampling_plan.get(dup, 0)
        if n_to_sample > 0 and indices:
            sampled = random.sample(indices, min(n_to_sample, len(indices)))
            # Group by file
            sampled_by_file: Dict[str, set] = defaultdict(set)
            for pf_path, idx in sampled:
                sampled_by_file[pf_path].add(idx)
            sampled_indices[dup] = sampled_by_file
            print(f"  duplicated={dup}: sampled {len(sampled):,} documents")

    # Flatten sampled indices by file for efficient reading
    file_indices: Dict[str, set] = defaultdict(set)
    for dup, by_file in sampled_indices.items():
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
                "duplicated": table.column("duplicated")[idx].as_py(),
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
            "duplicated": [r["duplicated"] for r in chunk],
            "source": [r["source"] for r in chunk],
        })

        output_path = os.path.join(
            output_data_dir,
            f"train-{file_idx:05d}-of-{n_files:05d}.parquet"
        )
        pq.write_table(table, output_path)

    # Write summary statistics
    print("\nWriting summary statistics...")
    duplicated_stats: Dict[int, Dict] = defaultdict(lambda: {"docs": 0, "tokens_estimate": 0})

    for row in all_sampled_rows:
        dup = row["duplicated"]
        duplicated_stats[dup]["docs"] += 1

    # Estimate tokens per duplicated value
    for dup in duplicated_stats:
        duplicated_stats[dup]["tokens_estimate"] = int(duplicated_stats[dup]["docs"] * avg_tokens)

    summary_path = os.path.join(output_dir, "sampling_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Aramix Consensus 10B Token Sample Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total documents: {len(all_sampled_rows):,}\n")
        f.write(f"Estimated total tokens: {estimated_total/1e9:.2f}B\n\n")
        f.write("Duplicated value breakdown:\n")

        total_docs = sum(s["docs"] for s in duplicated_stats.values())
        for dup, stats in sorted(duplicated_stats.items()):
            pct = 100 * stats["docs"] / total_docs
            f.write(f"  duplicated={dup}: {stats['docs']:,} docs ({pct:.1f}%)\n")

    print(f"Summary written to {summary_path}")
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Sample ~10B tokens from Aramix Consensus dataset with target duplicated ratios"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="~/data/aramix-consensus/",
        help="Input dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/data/aramix-consensus-10bt/",
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
        default=10.5,
        help="Target tokens in billions (default: 10.5B)"
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
        help="Number of documents to sample per duplicated value for token estimation"
    )

    args = parser.parse_args()

    target_tokens = int(args.target_tokens * 1e9)

    print("=" * 60)
    print("Aramix Consensus 10B Token Sampler")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Target tokens: {target_tokens/1e9:.1f}B")
    print(f"Random seed: {args.seed}")
    print(f"Filtering to duplicated values: {sorted(VALID_DUPLICATED_VALUES)}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # Estimate tokens per duplicated value
    avg_tokens, doc_counts = estimate_tokens_per_duplicated(
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
