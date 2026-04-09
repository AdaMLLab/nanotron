"""Download and sanitize HF datasets with inconsistent parquet schemas.

Reads parquet files directly via pyarrow to avoid datasets library cast errors.
Keeps only the text column needed for tokenization, writes to local jsonl.
Then tokenize with: preprocess_data.py ... jsonl --dataset <output_dir> --column text
"""

import os
import json

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

OUTPUT_BASE = os.path.expanduser("~/data/aratrain")

DATASETS = [
    ("SultanR/ultradata-math-qa-ar", "ar_text"),
    ("SultanR/ultradata-math-textbook-exercise-ar", "ar_text"),
]

for dataset_name, column in DATASETS:
    short_name = dataset_name.split("/")[-1]
    output_dir = os.path.join(OUTPUT_BASE, short_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "data.jsonl")

    if os.path.exists(output_file):
        existing_lines = sum(1 for _ in open(output_file))
        if existing_lines > 0:
            print(f"=== Skipping {dataset_name} (already sanitized: {existing_lines} rows) ===")
            continue

    print(f"=== Sanitizing {dataset_name} ===")
    print(f"    Column: {column}")
    print(f"    Output: {output_file}")

    api = HfApi()
    repo_files = api.list_repo_files(dataset_name, repo_type="dataset")
    parquet_files = sorted([f for f in repo_files if f.endswith(".parquet")])
    print(f"    Found {len(parquet_files)} parquet files")

    count = 0
    with open(output_file, "w") as f:
        for i, pf in enumerate(parquet_files):
            path = hf_hub_download(dataset_name, pf, repo_type="dataset")
            table = pq.read_table(path, columns=[column])
            for val in table.column(column).to_pylist():
                if val:
                    f.write(json.dumps({"text": val}) + "\n")
                    count += 1
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(parquet_files)} files, {count} rows...")

    print(f"    Done: {count} rows from {len(parquet_files)} files")
    print()

print("=== All datasets sanitized ===")
print(f"Verify: ls {OUTPUT_BASE}/")
