#!/usr/bin/env python3
"""Sample ~31B tokens from TurMix MinHash deduped with proportional source ratios."""

import argparse
from transformers import AutoTokenizer
from sampling_utils import estimate_tokens_per_source, calculate_sampling_plan, sample_and_save

SOURCE_MAPPING = {
    "c4": "C4", "C4": "C4", "mc4": "C4", "mC4": "C4",
    "culturax": "CulturaX", "CulturaX": "CulturaX", "uonlp/CulturaX": "CulturaX",
    "fineweb-2": "FineWeb-2", "FineWeb-2": "FineWeb-2", "fineweb2": "FineWeb-2", "HuggingFaceFW/fineweb-2": "FineWeb-2",
    "hplt-2": "HPLT-2", "HPLT-2": "HPLT-2", "hplt2": "HPLT-2", "HPLT/HPLT2.0_cleaned": "HPLT-2",
    "vngrs": "VNGRS", "VNGRS": "VNGRS", "vngrs_web_corpus": "VNGRS", "VNGRS Web Corpus": "VNGRS",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="~/data/pretraining_data/turmix-minhash-deduped")
    parser.add_argument("--output-dir", default="~/data/turmix-minhash-deduped-31bt/")
    parser.add_argument("--target-tokens", type=float, default=33.0, help="Billions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    target = int(args.target_tokens * 1e9)

    avg_tokens, doc_counts = estimate_tokens_per_source(
        args.input_dir, tokenizer, SOURCE_MAPPING, sample_size=args.sample_size)
    plan = calculate_sampling_plan(doc_counts, avg_tokens, target)
    sample_and_save(args.input_dir, args.output_dir, plan, SOURCE_MAPPING, tokenizer, seed=args.seed)
