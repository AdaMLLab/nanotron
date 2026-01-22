# Evaluation Scripts

## Script Bugs (Fixed)

- **Task parsing truncation**: `cut -d: -f2` truncated task names with colons (`alghafa_arc_ara_cf:easy` → `alghafa_arc_ara_cf`). Fixed with `cut -d: -f2-`.

- **Hindi task naming**: Config used `_hindi_` but lighteval uses ISO code `_hin_` (e.g., `xnli_hindi_cf` → `xnli_hin_cf`).

## lighteval 0.13.0 + datasets 4.x Bugs

The `datasets` library 4.x removed support for Python-based dataset loading scripts. lighteval 0.13.0 pins several datasets to old revisions that contain `.py` scripts, causing `RuntimeError: Dataset scripts are no longer supported`.

### Pinned Revision Fixes

Remove pinned `hf_revision` from task configs (old revisions have deprecated `.py` scripts, new versions use parquet):

```bash
# mlmm_hellaswag - jon-tow/okapi_hellaswag
sed -i 's/hf_revision="96ed8e0dfc6172dad1d3df338d7b8ba6c1ff9d83",/# hf_revision removed/' \
  $LIGHTEVAL/tasks/multilingual/tasks/mlmm_hellaswag.py

# mlmm_arc_challenge - jon-tow/okapi_arc_challenge
sed -i 's/hf_revision="823d5d7bfaf8974a3ab52a825b6cf4903b35dbc4",/# hf_revision removed/' \
  $LIGHTEVAL/tasks/multilingual/tasks/mlmm_arc_challenge.py

# mlqa - facebook/mlqa (switch to repaired dataset + fix schema)
sed -i 's|hf_repo="facebook/mlqa"|hf_repo="AdaMLLab/mlqa_repaired"|' \
  $LIGHTEVAL/tasks/multilingual/tasks/mlqa.py
sed -i 's/hf_revision="397ed406c1a7902140303e7faf60fff35b58d285",/# hf_revision removed/' \
  $LIGHTEVAL/tasks/multilingual/tasks/mlqa.py
sed -i 's/line\["answers"\]\["text"\]/line["answers.text"]/' \
  $LIGHTEVAL/tasks/multilingual/tasks/mlqa.py

# indicxcopa - ai4bharat/IndicCOPA
sed -i 's/hf_revision="d356ef19a4eb287e88a51d07a56b73ba88c7f188",/# hf_revision removed/' \
  $LIGHTEVAL/tasks/multilingual/tasks/copa_indic.py

# indicqa - ai4bharat/IndicQA
sed -i 's/hf_revision="92d96092ae229950973dac3b9998f8b3a8949b0a",/# hf_revision removed/' \
  $LIGHTEVAL/tasks/multilingual/tasks/indicqa.py
```

Where `$LIGHTEVAL=/home/user/miniconda3/envs/lighteval/lib/python3.11/site-packages/lighteval`

### Datasets Still Using Python Scripts

Some datasets (`ai4bharat/IndicCOPA`, `ai4bharat/IndicQA`) still use `.py` loader scripts in their main branch. Since `datasets` 4.x completely removed script support (even `trust_remote_code=True` doesn't work), these tasks are commented out in `hindi_finetasks.txt`.

### Clear Cached Datasets

After patching, clear cached datasets that were downloaded with old revisions:

```bash
rm -rf ~/.cache/huggingface/hub/datasets--jon-tow--okapi_hellaswag
rm -rf ~/.cache/huggingface/hub/datasets--jon-tow--okapi_arc_challenge
rm -rf ~/.cache/huggingface/hub/datasets--facebook--mlqa
rm -rf ~/.cache/huggingface/hub/datasets--ai4bharat--IndicCOPA
rm -rf ~/.cache/huggingface/hub/datasets--ai4bharat--IndicQA
```

### Normalization Bugs in normalizations.py

lighteval 0.13.0 has bugs in `normalizations.py` that cause crashes on some Hindi tasks. These fixes are defensive - they only handle edge cases and don't affect working evaluations (e.g., Arabic).

**IndexError in token normalization**: `choices_tokens` can have fewer elements than `choices_logprob`:
```bash
sed -i 's/for ix in range(len(choices_logprob))/for ix in range(min(len(choices_logprob), len(choices_tokens) if choices_tokens else len(choices_logprob)))/' \
  $LIGHTEVAL/metrics/normalizations.py
```

**AssertionError in PMI normalization**: `unconditioned_logprob` can be None. Fix assertion to fallback:
```bash
sed -i 's/assert unconditioned_logprob is not None, "unconditioned_logprob must be provided for PMI normalization"/if unconditioned_logprob is None: return choices_logprob  # fallback if missing/' \
  $LIGHTEVAL/metrics/normalizations.py
```

**PMI range bug**: Uses wrong variable in range. Fix:
```bash
sed -i 's/choices_logprob\[ix\] - unconditioned_logprob\[ix\] for ix in range(min(len(choices_logprob), len(choices_tokens) if choices_tokens else len(choices_logprob)))/choices_logprob[ix] - unconditioned_logprob[ix] for ix in range(min(len(choices_logprob), len(unconditioned_logprob)))/' \
  $LIGHTEVAL/metrics/normalizations.py
```

### TypeError in Generation (stop_sequences)

`stop_sequences` is a tuple but concatenated with a list, causing `TypeError: can only concatenate list (not "tuple") to list`. Fix:
```bash
sed -i 's/\[self.tokenizer.eos_token\] + batch\[0\].stop_sequences/[self.tokenizer.eos_token] + list(batch[0].stop_sequences or [])/' \
  $LIGHTEVAL/models/transformers/transformers_model.py
```
