"""Shared utilities for dataset sampling scripts."""

import glob
import hashlib
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def normalize_source(source: str, source_mapping: Dict[str, str]) -> str:
    if source in source_mapping:
        return source_mapping[source]
    source_lower = source.lower()
    for key, val in source_mapping.items():
        if key.lower() == source_lower:
            return val
    return source


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_consensus_hash_set(consensus_dir: str) -> Set[str]:
    parquet_files = sorted(glob.glob(os.path.join(consensus_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {consensus_dir}")
    print(f"Loading consensus hashes from {len(parquet_files)} files...")
    hashes: Set[str] = set()
    for pf_path in tqdm(parquet_files, desc="Loading consensus"):
        texts = pq.read_table(pf_path, columns=["text"]).column("text").to_pylist()
        for text in texts:
            hashes.add(text_hash(text))
    print(f"Loaded {len(hashes):,} unique text hashes")
    return hashes


def estimate_tokens_per_source(
    data_dir: str,
    tokenizer,
    source_mapping: Dict[str, str],
    excluded_sources: Optional[Set[str]] = None,
    consensus_hashes: Optional[Set[str]] = None,
    consensus_source: Optional[str] = None,
    sample_size: int = 1000,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Estimate avg tokens per doc for each source.

    If consensus_hashes and consensus_source are provided, documents from
    consensus_source are only counted if their text hash appears in the set.
    """
    excluded = excluded_sources or set()
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    print(f"Found {len(parquet_files)} parquet files")

    doc_counts: Dict[str, int] = defaultdict(int)
    need_text_for_count = consensus_hashes is not None and consensus_source is not None
    read_cols = ["source", "text"] if need_text_for_count else ["source"]

    print("Counting documents per source...")
    for pf_path in tqdm(parquet_files, desc="Counting"):
        table = pq.read_table(pf_path, columns=read_cols)
        sources = table.column("source").to_pylist()
        texts = table.column("text").to_pylist() if need_text_for_count else [None] * len(sources)
        for src, text in zip(sources, texts):
            canonical = normalize_source(src, source_mapping)
            if canonical in excluded:
                continue
            if canonical == consensus_source and consensus_hashes is not None:
                if text_hash(text) not in consensus_hashes:
                    continue
            doc_counts[canonical] += 1

    for src, count in sorted(doc_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count:,}")

    # Sample documents for token estimation
    print(f"\nSampling {sample_size} docs per source for token estimation...")
    docs_by_source: Dict[str, List[str]] = defaultdict(list)
    sample_needs = {src: sample_size for src in doc_counts}

    for pf_path in tqdm(parquet_files, desc="Sampling"):
        if all(n <= 0 for n in sample_needs.values()):
            break
        table = pq.read_table(pf_path, columns=["text", "source"])
        df = table.to_pandas()
        for src, group in df.groupby("source"):
            canonical = normalize_source(src, source_mapping)
            if canonical in excluded or sample_needs.get(canonical, 0) <= 0:
                continue
            eligible = group
            if canonical == consensus_source and consensus_hashes is not None:
                eligible = group[group["text"].apply(lambda t: text_hash(t) in consensus_hashes)]
            if len(eligible) > 0:
                n = min(sample_needs[canonical], len(eligible))
                docs_by_source[canonical].extend(eligible.sample(n=n, random_state=42)["text"].tolist())
                sample_needs[canonical] -= n

    avg_tokens: Dict[str, float] = {}
    print("\nTokenizing samples...")
    for src, docs in tqdm(docs_by_source.items(), desc="Tokenizing"):
        if not docs:
            avg_tokens[src] = 0
            continue
        total = sum(len(tokenizer.encode(d, add_special_tokens=False)) for d in docs[:sample_size])
        avg_tokens[src] = total / len(docs[:sample_size])
        print(f"  {src}: avg {avg_tokens[src]:.1f} tokens/doc")

    return avg_tokens, dict(doc_counts)


def calculate_sampling_plan(
    doc_counts: Dict[str, int],
    avg_tokens: Dict[str, float],
    target_tokens: int,
    target_ratios: Optional[Dict[str, float]] = None,
) -> Dict[str, int]:
    """
    Calculate docs to sample per source.
    If target_ratios is None, samples proportionally based on doc counts.
    """
    available = set(doc_counts.keys())

    if target_ratios is None:
        # Proportional sampling based on document counts
        total_docs = sum(doc_counts.values())
        ratios = {src: 100 * count / total_docs for src, count in doc_counts.items()}
    else:
        ratios = {k: v for k, v in target_ratios.items() if k in available}
        missing = set(target_ratios.keys()) - available
        if missing:
            print(f"Warning: sources not available: {missing}")

    total_ratio = sum(ratios.values())
    normalized = {k: v / total_ratio * 100 for k, v in ratios.items()}

    print("\nTarget ratios:")
    for src, ratio in sorted(normalized.items(), key=lambda x: -x[1]):
        print(f"  {src}: {ratio:.2f}%")

    target_per_source = {src: target_tokens * (r / 100) for src, r in normalized.items()}
    plan: Dict[str, int] = {}
    total_est = 0

    print("\nSampling plan:")
    for src in available:
        if avg_tokens.get(src, 0) == 0:
            plan[src] = 0
            continue
        needed = int(target_per_source.get(src, 0) / avg_tokens[src])
        actual = min(needed, doc_counts[src])
        plan[src] = actual
        est = actual * avg_tokens[src]
        total_est += est
        cap = f" (capped from {needed:,})" if actual < needed else ""
        print(f"  {src}: {actual:,} docs (~{est/1e9:.2f}B tokens){cap}")

    print(f"\nTotal estimated tokens: {total_est/1e9:.2f}B")
    return plan


def sample_and_save(
    data_dir: str,
    output_dir: str,
    sampling_plan: Dict[str, int],
    source_mapping: Dict[str, str],
    tokenizer,
    columns: List[str] = None,
    excluded_sources: Optional[Set[str]] = None,
    consensus_hashes: Optional[Set[str]] = None,
    consensus_source: Optional[str] = None,
    seed: int = 42,
    rows_per_file: int = 500_000,
) -> None:
    """
    Sample documents and save as parquet.

    columns: which columns to keep in output (default: ["text", "id", "source"])
    """
    if columns is None:
        columns = ["text", "id", "source"]
    excluded = excluded_sources or set()

    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))
    need_text_check = consensus_hashes is not None and consensus_source is not None

    # Collect eligible indices per source
    print("Collecting document indices...")
    source_indices: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    index_cols = ["source"]
    if need_text_check:
        index_cols.append("text")

    for pf_path in tqdm(parquet_files, desc="Indexing"):
        table = pq.read_table(pf_path, columns=index_cols)
        sources = table.column("source").to_pylist()
        texts = table.column("text").to_pylist() if need_text_check else [None] * len(sources)
        for idx, (src, text) in enumerate(zip(sources, texts)):
            canonical = normalize_source(src, source_mapping)
            if canonical in excluded:
                continue
            if canonical == consensus_source and consensus_hashes is not None:
                if text_hash(text) not in consensus_hashes:
                    continue
            source_indices[canonical].append((pf_path, idx))

    # Sample indices
    print("Sampling indices...")
    file_indices: Dict[str, set] = defaultdict(set)
    for src, indices in source_indices.items():
        n = sampling_plan.get(src, 0)
        if n > 0 and indices:
            sampled = random.sample(indices, min(n, len(indices)))
            for pf_path, idx in sampled:
                file_indices[pf_path].add(idx)
            print(f"  {src}: {len(sampled):,} docs")

    # Read sampled documents
    print("Reading sampled documents...")
    all_rows = []
    for pf_path in tqdm(parquet_files, desc="Reading"):
        if pf_path not in file_indices:
            continue
        table = pq.read_table(pf_path)
        for idx in sorted(file_indices[pf_path]):
            row = {col: table.column(col)[idx].as_py() for col in columns}
            all_rows.append(row)

    print(f"Total sampled: {len(all_rows):,} docs")
    random.shuffle(all_rows)

    # Estimate tokens
    sample_n = min(5000, len(all_rows))
    sample_tokens = sum(
        len(tokenizer.encode(r["text"], add_special_tokens=False))
        for r in tqdm(random.sample(all_rows, sample_n), desc="Counting tokens")
    )
    avg_tok = sample_tokens / sample_n
    print(f"Estimated total: {avg_tok * len(all_rows) / 1e9:.2f}B tokens")

    # Save
    output_data_dir = os.path.join(output_dir, "data", "sampled")
    os.makedirs(output_data_dir, exist_ok=True)
    n_files = (len(all_rows) + rows_per_file - 1) // rows_per_file

    for i in tqdm(range(n_files), desc="Writing"):
        chunk = all_rows[i * rows_per_file : (i + 1) * rows_per_file]
        table = pa.table({col: [r[col] for r in chunk] for col in columns})
        pq.write_table(table, os.path.join(output_data_dir, f"train-{i:05d}-of-{n_files:05d}.parquet"))

    print(f"Done: {output_data_dir}")


# ---------------------------------------------------------------------------
# Consensus dataset utilities (source column is a *list* of strings)
# ---------------------------------------------------------------------------

def should_keep_consensus_doc(sources: List[str], excluded_source: str) -> bool:
    """Keep doc unless it has excluded_source + only 1 other source."""
    if excluded_source not in sources:
        return True
    return len(sources) >= 3


def count_eligible_consensus_docs(
    data_dir: str, tokenizer, excluded_source: str, sample_size: int = 5000,
) -> Tuple[int, float]:
    """Count eligible docs and estimate avg tokens for consensus datasets."""
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    print(f"Found {len(parquet_files)} parquet files")

    eligible_docs = 0
    sampled_texts: List[str] = []

    for pf_path in tqdm(parquet_files, desc="Counting"):
        table = pq.read_table(pf_path)
        sources_col = table.column("source").to_pylist()
        texts_col = table.column("text").to_pylist()
        for i, sources in enumerate(sources_col):
            if should_keep_consensus_doc(sources, excluded_source):
                eligible_docs += 1
                if len(sampled_texts) < sample_size:
                    sampled_texts.append(texts_col[i])
                elif random.random() < sample_size / eligible_docs:
                    sampled_texts[random.randint(0, sample_size - 1)] = texts_col[i]

    print(f"Eligible: {eligible_docs:,}")
    total_tokens = sum(
        len(tokenizer.encode(t, add_special_tokens=False))
        for t in tqdm(sampled_texts, desc="Tokenizing")
    )
    avg_tokens = total_tokens / len(sampled_texts) if sampled_texts else 0
    print(f"Avg tokens/doc: {avg_tokens:.1f}, estimated total: {avg_tokens * eligible_docs / 1e9:.2f}B")
    return eligible_docs, avg_tokens


def calculate_sampling_rate(eligible_docs: int, avg_tokens: float, target_tokens: int) -> float:
    """Calculate fraction of docs to sample to reach target tokens."""
    estimated_total = eligible_docs * avg_tokens
    if estimated_total <= target_tokens:
        print(f"Warning: eligible ({estimated_total/1e9:.2f}B) < target ({target_tokens/1e9:.2f}B). Using all.")
        return 1.0
    rate = target_tokens / estimated_total
    print(f"Sampling rate: {rate:.4f} ({rate*100:.2f}%), ~{int(eligible_docs * rate):,} docs")
    return rate


def consensus_filter_sample_save(
    data_dir: str, output_dir: str, sampling_rate: float,
    excluded_source: str, tokenizer, seed: int = 42,
) -> None:
    """Filter consensus dataset by excluded source, sample by rate, save."""
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))

    eligible_indices: List[Tuple[str, int]] = []
    for pf_path in tqdm(parquet_files, desc="Indexing"):
        sources = pq.read_table(pf_path, columns=["source"]).column("source").to_pylist()
        for idx, src_list in enumerate(sources):
            if should_keep_consensus_doc(src_list, excluded_source):
                eligible_indices.append((pf_path, idx))

    n_to_sample = int(len(eligible_indices) * sampling_rate)
    sampled = random.sample(eligible_indices, min(n_to_sample, len(eligible_indices)))

    file_indices: Dict[str, set] = defaultdict(set)
    for pf_path, idx in sampled:
        file_indices[pf_path].add(idx)

    all_rows = []
    for pf_path in tqdm(parquet_files, desc="Reading"):
        if pf_path not in file_indices:
            continue
        table = pq.read_table(pf_path)
        has_dup = "duplicated" in table.column_names
        for idx in sorted(file_indices[pf_path]):
            row = {"text": table.column("text")[idx].as_py(), "source": table.column("source")[idx].as_py()}
            if has_dup:
                row["duplicated"] = table.column("duplicated")[idx].as_py()
            all_rows.append(row)

    random.shuffle(all_rows)
    print(f"Sampled {len(all_rows):,} docs")

    sample_n = min(5000, len(all_rows))
    sample_tok = sum(len(tokenizer.encode(r["text"], add_special_tokens=False))
                     for r in tqdm(random.sample(all_rows, sample_n), desc="Counting tokens"))
    print(f"Estimated total: {sample_tok / sample_n * len(all_rows) / 1e9:.2f}B tokens")

    save_rows_to_parquet(all_rows, output_dir)


def save_rows_to_parquet(
    rows: List[Dict], output_dir: str,
    subdir: str = "data", rows_per_file: int = 500_000,
) -> None:
    """Save list of row dicts to chunked parquet files."""
    if not rows:
        return
    output_data_dir = os.path.join(output_dir, subdir)
    os.makedirs(output_data_dir, exist_ok=True)
    cols = list(rows[0].keys())
    n_files = (len(rows) + rows_per_file - 1) // rows_per_file
    for i in tqdm(range(n_files), desc="Writing"):
        chunk = rows[i * rows_per_file : (i + 1) * rows_per_file]
        table = pa.table({col: [r[col] for r in chunk] for col in cols})
        pq.write_table(table, os.path.join(output_data_dir, f"train-{i:05d}-of-{n_files:05d}.parquet"))
    print(f"Done: {output_data_dir}")
