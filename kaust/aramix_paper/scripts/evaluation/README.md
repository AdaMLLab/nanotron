# Evaluation Scripts - Known Bugs

## Fixed bugs

- **Task parsing truncation**: `run_finetasks.sh` used `cut -d: -f2` which truncated task names containing colons (e.g., `alghafa_arc_ara_cf:easy` became `alghafa_arc_ara_cf`). Fixed by using `cut -d: -f2-`.

- **Hindi task naming**: `hindi_finetasks.txt` used `_hindi_` in task names but lighteval uses ISO code `_hin_` (e.g., `xnli_hindi_cf` → `xnli_hin_cf`).

## lighteval evaluation fixes

- **mlmm_hellaswag revision**: lighteval 0.13.0 pins `jon-tow/okapi_hellaswag` to an old revision containing a deprecated `.py` loader script, causing `RuntimeError: Dataset scripts are no longer supported`. Fix:
  ```bash
  sed -i 's/hf_revision="96ed8e0dfc6172dad1d3df338d7b8ba6c1ff9d83",/# hf_revision removed/' \
    /home/user/miniconda3/envs/lighteval/lib/python3.11/site-packages/lighteval/tasks/multilingual/tasks/mlmm_hellaswag.py
  ```

- **mlqa dataset**: Use `AdaMLLab/mlqa_repaired` instead of `facebook/mlqa` and remove pinned revision. Fix:
  ```bash
  sed -i 's|hf_repo="facebook/mlqa"|hf_repo="AdaMLLab/mlqa_repaired"|' \
    /home/user/miniconda3/envs/lighteval/lib/python3.11/site-packages/lighteval/tasks/multilingual/tasks/mlqa.py
  sed -i 's/hf_revision="397ed406c1a7902140303e7faf60fff35b58d285",/# hf_revision removed/' \
    /home/user/miniconda3/envs/lighteval/lib/python3.11/site-packages/lighteval/tasks/multilingual/tasks/mlqa.py
  ```
