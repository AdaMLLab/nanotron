"""Filter consensus parquet files by `duplicated >= threshold`.

Usage:
    python filter_parquet.py <src_dir> <dst_dir> <threshold>

Idempotent: skips files that already exist at dst.
"""
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def main(src_dir: str, dst_dir: str, threshold: int) -> None:
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("train-*.parquet"))
    if not files:
        raise SystemExit(f"No train-*.parquet files in {src}")

    total_in, total_out = 0, 0
    for f in files:
        dst_file = dst / f.name
        if dst_file.exists():
            print(f"[filter] skip (exists): {dst_file.name}", flush=True)
            continue
        table = pq.read_table(f)
        mask = pc.greater_equal(table["duplicated"], threshold)
        filtered = table.filter(mask)
        total_in += table.num_rows
        total_out += filtered.num_rows
        if filtered.num_rows > 0:
            pq.write_table(filtered, dst_file, compression="snappy")
            print(f"[filter] {f.name}: {table.num_rows} -> {filtered.num_rows}", flush=True)
        else:
            print(f"[filter] {f.name}: {table.num_rows} -> 0 (skipped)", flush=True)

    if total_in:
        print(f"[filter] Total: {total_in} -> {total_out} ({100*total_out/total_in:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
