# FineWeb2 Arabic Evaluation

Reproduces Arabic 30B-token ablation from [FineWeb2 paper](https://arxiv.org/abs/2506.20920).

## Versions
```
lighteval==0.13.0
vllm==0.10.1
torch==2.7.1
```

## Bug Fixes

**lighteval vLLM stop parameter** (`/path/to/lighteval/models/vllm/vllm_model.py`):
```python
# Line 423: change
sampling_params.stop = stop_tokens
# to
sampling_params.stop = list(stop_tokens) if stop_tokens else []

# Line 609: change
sampling_params.stop = [] if self.use_chat_template else doc.stop_sequences
# to
sampling_params.stop = [] if self.use_chat_template else list(doc.stop_sequences or [])
```

**lighteval vLLM version check** (`/path/to/lighteval/utils/imports.py`):
```python
# Line 66: add before "return installed in package.specifier"
if package.name == "vllm":
    return True
```

## Usage

```bash
# Convert and evaluate all checkpoints (8 GPUs parallel)
./scripts/parallel_eval.sh /path/to/checkpoints model_name google/gemma-2b 8
```

## Scripts
- `convert_checkpoint.sh` - Convert nanotron checkpoint to HF
- `parallel_eval.sh` - Parallel evaluation across GPUs with vLLM
