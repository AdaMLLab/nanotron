#!/usr/bin/env python3
"""
Script to sample ~30B tokens from the FineWeb-Edu Arabic dataset.

This dataset is a single source (educational Arabic web content), so no
ratio balancing is needed - we simply sample enough documents to reach
the target token count.

Dataset info:
- Location: /home/alrashsm/data/fineweb-edu-ar-kaust-gen-ai/ar
- Format: HuggingFace dataset (zip files with jsonl)
- Column: text
"""

import argparse
import os
import random
from typing import List, Dict

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


def estimate_tokens(
    dataset,
    tokenizer,
    sample_size: int = 5000
) -> float:
    """
    Estimate average tokens per document by sampling.

    Returns:
        Average tokens per document
    """
    print(f"Estimating tokens per document (sampling {sample_size} docs)...")

    total_docs = len(dataset)
    sample_indices = random.sample(range(total_docs), min(sample_size, total_docs))

    total_tokens = 0
    for idx in tqdm(sample_indices, desc="Tokenizing samples"):
        text = dataset[idx]["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)

    avg_tokens = total_tokens / len(sample_indices)
    print(f"Average tokens per document: {avg_tokens:.1f}")
    print(f"Total documents in dataset: {total_docs:,}")
    print(f"Estimated total tokens in dataset: {avg_tokens * total_docs / 1e9:.2f}B")

    return avg_tokens


def sample_and_save(
    dataset,
    output_dir: str,
    target_tokens: int,
    avg_tokens: float,
    tokenizer,
    seed: int = 42
) -> None:
    """
    Sample documents to reach target token count and save to parquet files.
    """
    random.seed(seed)

    total_docs = len(dataset)
    docs_needed = int(target_tokens / avg_tokens)

    # Cap at available documents
    if docs_needed >= total_docs:
        print(f"\nTarget requires {docs_needed:,} docs, but only {total_docs:,} available.")
        print(f"Using all available documents.")
        docs_to_sample = total_docs
        sample_indices = list(range(total_docs))
    else:
        print(f"\nSampling {docs_needed:,} documents from {total_docs:,} available.")
        docs_to_sample = docs_needed
        sample_indices = random.sample(range(total_docs), docs_to_sample)

    # Shuffle the indices
    random.shuffle(sample_indices)

    estimated_tokens = docs_to_sample * avg_tokens
    print(f"Estimated tokens after sampling: {estimated_tokens/1e9:.2f}B")

    # Create output directory
    output_data_dir = os.path.join(output_dir, "data")
    os.makedirs(output_data_dir, exist_ok=True)

    # Save to parquet files in chunks
    rows_per_file = 500_000
    n_files = (docs_to_sample + rows_per_file - 1) // rows_per_file

    print(f"\nSaving to {n_files} parquet files...")

    for file_idx in tqdm(range(n_files), desc="Writing files"):
        start_idx = file_idx * rows_per_file
        end_idx = min((file_idx + 1) * rows_per_file, docs_to_sample)

        chunk_indices = sample_indices[start_idx:end_idx]

        texts = []
        ids = []

        for idx in chunk_indices:
            row = dataset[idx]
            texts.append(row["text"])
            ids.append(f"fineweb-edu-ar-{idx}")

        table = pa.table({
            "text": texts,
            "id": ids,
            "source": ["fineweb-edu-ar"] * len(texts),
        })

        output_path = os.path.join(
            output_data_dir,
            f"train-{file_idx:05d}-of-{n_files:05d}.parquet"
        )
        pq.write_table(table, output_path)

    # Verify token count with a sample
    print("\nVerifying token count...")
    verify_sample_size = min(5000, docs_to_sample)
    verify_indices = random.sample(sample_indices, verify_sample_size)
    verify_tokens = 0
    for idx in tqdm(verify_indices, desc="Verifying"):
        text = dataset[idx]["text"]
        verify_tokens += len(tokenizer.encode(text, add_special_tokens=False))

    verified_avg = verify_tokens / verify_sample_size
    verified_total = verified_avg * docs_to_sample

    print(f"Verified average tokens: {verified_avg:.1f}")
    print(f"Verified total tokens: {verified_total/1e9:.2f}B")

    # Write summary
    summary_path = os.path.join(output_dir, "sampling_summary.txt")
    with open(summary_path, "w") as f:
        f.write("FineWeb-Edu Arabic 30B Token Sample Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Source dataset: fineweb-edu-ar-kaust-gen-ai\n")
        f.write(f"Total documents sampled: {docs_to_sample:,}\n")
        f.write(f"Estimated total tokens: {verified_total/1e9:.2f}B\n")
        f.write(f"Average tokens per doc: {verified_avg:.1f}\n\n")
        f.write(f"Output files: {n_files} parquet files\n")
        f.write(f"Output location: {output_data_dir}\n")

    print(f"\nSummary written to {summary_path}")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Sample ~30B tokens from FineWeb-Edu Arabic dataset"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/alrashsm/data/fineweb-edu-ar-kaust-gen-ai/ar",
        help="Input dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/alrashsm/data/fineweb-edu-ar-30bt/",
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
        default=30.5,
        help="Target tokens in billions (default: 30.5B for ~30B target)"
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

    print("=" * 60)
    print("FineWeb-Edu Arabic 30B Token Sampler")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Target tokens: {target_tokens/1e9:.1f}B")
    print(f"Random seed: {args.seed}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(args.input_dir, split="train")
    print(f"Dataset loaded: {len(dataset):,} documents")

    # Set random seed
    random.seed(args.seed)

    # Estimate tokens per document
    avg_tokens = estimate_tokens(dataset, tokenizer, sample_size=args.sample_size)

    # Sample and save
    sample_and_save(
        dataset,
        args.output_dir,
        target_tokens,
        avg_tokens,
        tokenizer,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
