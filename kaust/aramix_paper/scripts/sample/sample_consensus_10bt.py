#!/usr/bin/env python3
"""Sample ~10B tokens from Aramix Consensus, grouped by duplicated value (2/3/4 only)."""

import argparse
import glob
import os
import random
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer
from sampling_utils import save_rows_to_parquet

VALID_DUPLICATED = {2, 3, 4}
TARGET_DOC_COUNTS = {2: 60_466_799, 3: 43_200_076, 4: 26_015_836}
_total = sum(TARGET_DOC_COUNTS.values())
TARGET_RATIOS = {k: 100 * v / _total for k, v in TARGET_DOC_COUNTS.items()}


def estimate_tokens_per_duplicated(data_dir, tokenizer, sample_size=1000):
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    print(f"Found {len(parquet_files)} parquet files")

    doc_counts = defaultdict(int)
    for pf_path in tqdm(parquet_files, desc="Counting"):
        for dup in pq.read_table(pf_path, columns=["duplicated"]).column("duplicated").to_pylist():
            if dup in VALID_DUPLICATED:
                doc_counts[dup] += 1

    for dup, count in sorted(doc_counts.items()):
        print(f"  duplicated={dup}: {count:,}")

    sample_needs = {dup: sample_size for dup in VALID_DUPLICATED}
    docs_by_dup = defaultdict(list)
    for pf_path in tqdm(parquet_files, desc="Sampling"):
        if all(n <= 0 for n in sample_needs.values()):
            break
        df = pq.read_table(pf_path, columns=["text", "duplicated"]).to_pandas()
        for dup, group in df.groupby("duplicated"):
            if dup in VALID_DUPLICATED and sample_needs.get(dup, 0) > 0:
                n = min(sample_needs[dup], len(group))
                docs_by_dup[dup].extend(group.sample(n=n, random_state=42)["text"].tolist())
                sample_needs[dup] -= n

    avg_tokens = {}
    for dup, docs in tqdm(sorted(docs_by_dup.items()), desc="Tokenizing"):
        if not docs:
            avg_tokens[dup] = 0
            continue
        total = sum(len(tokenizer.encode(d, add_special_tokens=False)) for d in docs[:sample_size])
        avg_tokens[dup] = total / len(docs[:sample_size])
        print(f"  duplicated={dup}: avg {avg_tokens[dup]:.1f} tokens/doc")

    return avg_tokens, dict(doc_counts)


def calculate_plan(doc_counts, avg_tokens, target_tokens):
    total_ratio = sum(v for k, v in TARGET_RATIOS.items() if k in doc_counts)
    normalized = {k: v / total_ratio * 100 for k, v in TARGET_RATIOS.items() if k in doc_counts}
    target_per = {dup: target_tokens * (r / 100) for dup, r in normalized.items()}

    plan = {}
    total_est = 0
    for dup in sorted(doc_counts):
        if avg_tokens.get(dup, 0) == 0:
            plan[dup] = 0
            continue
        needed = int(target_per.get(dup, 0) / avg_tokens[dup])
        actual = min(needed, doc_counts[dup])
        plan[dup] = actual
        est = actual * avg_tokens[dup]
        total_est += est
        cap = f" (capped from {needed:,})" if actual < needed else ""
        print(f"  duplicated={dup}: {actual:,} docs (~{est/1e9:.2f}B tokens){cap}")
    print(f"Total estimated: {total_est/1e9:.2f}B")
    return plan


def sample_and_save(data_dir, output_dir, plan, tokenizer, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))

    dup_indices = defaultdict(list)
    for pf_path in tqdm(parquet_files, desc="Indexing"):
        for idx, dup in enumerate(pq.read_table(pf_path, columns=["duplicated"]).column("duplicated").to_pylist()):
            if dup in VALID_DUPLICATED:
                dup_indices[dup].append((pf_path, idx))

    file_indices = defaultdict(set)
    for dup, indices in sorted(dup_indices.items()):
        n = plan.get(dup, 0)
        if n > 0 and indices:
            for pf_path, idx in random.sample(indices, min(n, len(indices))):
                file_indices[pf_path].add(idx)
            print(f"  duplicated={dup}: {min(n, len(indices)):,} docs")

    all_rows = []
    for pf_path in tqdm(parquet_files, desc="Reading"):
        if pf_path not in file_indices:
            continue
        table = pq.read_table(pf_path)
        for idx in sorted(file_indices[pf_path]):
            all_rows.append({
                "text": table.column("text")[idx].as_py(),
                "duplicated": table.column("duplicated")[idx].as_py(),
                "source": table.column("source")[idx].as_py(),
            })

    random.shuffle(all_rows)
    print(f"Sampled {len(all_rows):,} docs")

    sample_n = min(5000, len(all_rows))
    sample_tok = sum(len(tokenizer.encode(r["text"], add_special_tokens=False))
                     for r in tqdm(random.sample(all_rows, sample_n), desc="Counting tokens"))
    print(f"Estimated total: {sample_tok / sample_n * len(all_rows) / 1e9:.2f}B tokens")

    save_rows_to_parquet(all_rows, output_dir, subdir="data/sampled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="~/data/aramix-consensus/")
    parser.add_argument("--output-dir", default="~/data/aramix-consensus-10bt/")
    parser.add_argument("--target-tokens", type=float, default=10.5, help="Billions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    target = int(args.target_tokens * 1e9)

    avg_tokens, doc_counts = estimate_tokens_per_duplicated(args.input_dir, tokenizer, args.sample_size)
    plan = calculate_plan(doc_counts, avg_tokens, target)
    sample_and_save(args.input_dir, args.output_dir, plan, tokenizer, args.seed)
