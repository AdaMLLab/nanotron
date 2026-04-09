# kaust

## aratrain

LLaMA 1.46B trained in two phases: 10B tokens on AraMix-HQ, then 3 continuation runs of 20B tokens each (30B total per branch).

```
kaust/aratrain/
├── .gitignore
├── configs/
│   ├── config_llama_1.46B_aramix_hq_10bt.yaml       # Base: 4,770 steps (10B tokens)
│   ├── config_llama_1.46B_finewiki_en_ar.yaml        # Continuation: 14,306 steps (30B total)
│   ├── config_llama_1.46B_ultradata_math.yaml        # Continuation: 50/50 mix, 14,306 steps
│   └── config_llama_1.46B_openresearcher.yaml        # Continuation: 14,306 steps
└── scripts/
    ├── tokenize_datasets.sh      # Tokenizes 4 HF datasets with --column ar_text
    ├── run_base.sh               # Runs base 10B training
    └── run_continuations.sh      # Runs all 3 continuations sequentially
```

- All configs use `lr_decay_steps: 13806` (spans full 30B), `warmup: 500`, `decay_starting_step: 500`
- Base trains 4,770 steps with `save_final_state: true`
- Continuations resume from the base checkpoint via `resume_checkpoint_path`
- Ultradata math uses `dataset_folder` list + `dataset_weights: [0.5, 0.5]`
- Tokenization calls `tools/preprocess_data.py` directly with `--column ar_text`
