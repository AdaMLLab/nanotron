#!/usr/bin/env python3
"""Sample ~30B tokens from a consensus dataset, filtering docs that have excluded_source + only 1 other source.

# Aramix (drop ArabicWeb24-only pairs):
#   python sample_consensus_30bt_no_source.py --input-dir ~/data/aramix_paper/aramix-consensus \
#     --output-dir ~/data/aramix_paper/aramix-consensus-no_aw24-30bt --excluded-source arabicweb24
#
# HinMix (drop CulturaX-only pairs):
#   python sample_consensus_30bt_no_source.py --input-dir ~/data/aramix_paper/hinmix-consensus \
#     --output-dir ~/data/aramix_paper/hinmix-consensus-no_culturax-30bt --excluded-source culturax
"""

import argparse
from transformers import AutoTokenizer
from sampling_utils import (
    count_eligible_consensus_docs, calculate_sampling_rate, consensus_filter_sample_save,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--excluded-source", required=True, help="Source to filter (e.g. arabicweb24, culturax)")
    parser.add_argument("--target-tokens", type=float, default=31.5, help="Billions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    target = int(args.target_tokens * 1e9)

    eligible_docs, avg_tokens = count_eligible_consensus_docs(
        args.input_dir, tokenizer, args.excluded_source, args.sample_size)
    rate = calculate_sampling_rate(eligible_docs, avg_tokens, target)
    consensus_filter_sample_save(
        args.input_dir, args.output_dir, rate, args.excluded_source, tokenizer, args.seed)
