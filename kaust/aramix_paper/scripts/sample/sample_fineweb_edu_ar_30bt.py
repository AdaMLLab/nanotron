#!/usr/bin/env python3
"""Sample ~30B tokens from FineWeb-Edu Arabic (HuggingFace dataset)."""

import argparse
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


def sample_and_save(dataset, output_dir, target_tokens, tokenizer, seed=42, sample_size=5000):
    random.seed(seed)
    total_docs = len(dataset)

    # Estimate avg tokens
    indices = random.sample(range(total_docs), min(sample_size, total_docs))
    total_tok = sum(len(tokenizer.encode(dataset[i]["text"], add_special_tokens=False))
                    for i in tqdm(indices, desc="Estimating tokens"))
    avg_tokens = total_tok / len(indices)
    print(f"Avg tokens/doc: {avg_tokens:.1f}, dataset total: {avg_tokens * total_docs / 1e9:.2f}B")

    docs_needed = min(int(target_tokens / avg_tokens), total_docs)
    sample_indices = random.sample(range(total_docs), docs_needed)
    random.shuffle(sample_indices)
    print(f"Sampling {docs_needed:,} / {total_docs:,} docs (~{docs_needed * avg_tokens / 1e9:.2f}B tokens)")

    output_data_dir = os.path.join(output_dir, "data")
    os.makedirs(output_data_dir, exist_ok=True)
    rows_per_file = 500_000
    n_files = (docs_needed + rows_per_file - 1) // rows_per_file

    for file_idx in tqdm(range(n_files), desc="Writing"):
        chunk = sample_indices[file_idx * rows_per_file : (file_idx + 1) * rows_per_file]
        texts, ids = [], []
        for idx in chunk:
            texts.append(dataset[idx]["text"])
            ids.append(f"fineweb-edu-ar-{idx}")
        table = pa.table({"text": texts, "id": ids, "source": ["fineweb-edu-ar"] * len(texts)})
        pq.write_table(table, os.path.join(output_data_dir, f"train-{file_idx:05d}-of-{n_files:05d}.parquet"))

    print(f"Done: {output_data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="~/data/fineweb-edu-ar-kaust-gen-ai/ar")
    parser.add_argument("--output-dir", default="~/data/fineweb-edu-ar-30bt/")
    parser.add_argument("--target-tokens", type=float, default=30.5, help="Billions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    target = int(args.target_tokens * 1e9)
    dataset = load_dataset(args.input_dir, split="train")
    print(f"Loaded {len(dataset):,} documents")
    sample_and_save(dataset, args.output_dir, target, tokenizer, args.seed, args.sample_size)
