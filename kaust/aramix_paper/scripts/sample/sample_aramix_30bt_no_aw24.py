#!/usr/bin/env python3
"""Sample ~30B tokens from Aramix MinHash deduped, excluding ArabicWeb24."""

import argparse
from transformers import AutoTokenizer
from sampling_utils import estimate_tokens_per_source, calculate_sampling_plan, sample_and_save

SOURCE_MAPPING = {
    "uonlp/CulturaX": "CulturaX", "CulturaX": "CulturaX",
    "lightonai/ArabicWeb24": "ArabicWeb24", "ArabicWeb24": "ArabicWeb24",
    "allenai/c4": "C4", "C4": "C4",
    "ClusterlabAi/101_billion_arabic_words_dataset": "101B Arabic Words", "101B Arabic Words": "101B Arabic Words",
    "HPLT/HPLT2.0_cleaned": "HPLT 2.0", "hplt/HPLT": "HPLT 2.0", "HPLT": "HPLT 2.0", "HPLT 2.0": "HPLT 2.0",
    "HuggingFaceFW/fineweb-2": "FineWeb-2", "fineweb-2": "FineWeb-2", "FineWeb-2": "FineWeb-2",
    "HuggingFaceFW/finepdfs": "FinePDFs", "FinePDFs": "FinePDFs", "finepdfs": "FinePDFs",
}

EXCLUDED_SOURCES = {"ArabicWeb24"}

TARGET_RATIOS = {
    "CulturaX": 29.55, "HPLT 2.0": 24.31, "FineWeb-2": 19.33,
    "C4": 15.84, "101B Arabic Words": 6.61, "FinePDFs": 4.36,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="~/data/pretraining_data/aramix-minhash-deduped")
    parser.add_argument("--output-dir", default="~/data/aramix_paper/aramix-minhash-no_aw24-30bt/")
    parser.add_argument("--target-tokens", type=float, default=31.5, help="Billions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    target = int(args.target_tokens * 1e9)

    avg_tokens, doc_counts = estimate_tokens_per_source(
        args.input_dir, tokenizer, SOURCE_MAPPING, excluded_sources=EXCLUDED_SOURCES, sample_size=args.sample_size)
    plan = calculate_sampling_plan(doc_counts, avg_tokens, target, TARGET_RATIOS)
    sample_and_save(args.input_dir, args.output_dir, plan, SOURCE_MAPPING, tokenizer,
                    excluded_sources=EXCLUDED_SOURCES, seed=args.seed)
