uv pip install -e .
uv pip install datasets transformers datatrove[io] numba wandb
# Fused kernels
uv pip install ninja triton "flash-attn>=2.5.0" --no-build-isolationX

