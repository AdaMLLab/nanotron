#!/bin/bash
#SBATCH --job-name=deps-install
#SBATCH --partition=pi-orabonf
#SBATCH --qos=pi-orabonf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/u/alrashsm/nanotron_checkpoints/logs/flash_attn_install_%j.out
#SBATCH --error=/mnt/data/u/alrashsm/nanotron_checkpoints/logs/flash_attn_install_%j.err

set -e

source /home/alrashsm/miniconda3/etc/profile.d/conda.sh
conda activate nanotron

echo "=== Environment Info (before) ==="
echo "Node: $(hostname)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

echo ""
echo "=== Uninstalling old flash-attn ==="
uv pip uninstall flash-attn -p /home/alrashsm/miniconda3/envs/nanotron 2>/dev/null || true
pip uninstall flash-attn -y 2>/dev/null || true
rm -f /home/alrashsm/miniconda3/envs/nanotron/lib/python3.11/site-packages/flash_attn_2_cuda*.so

echo ""
echo "=== Installing flash-attn (uv, no-build-isolation, no-cache) ==="
uv pip install flash-attn --no-build-isolation --no-cache -p /home/alrashsm/miniconda3/envs/nanotron

echo ""
echo "=== Installing grouped_gemm ==="
uv pip install "git+https://github.com/fanshiqing/grouped_gemm@main" --no-build-isolation --no-cache -p /home/alrashsm/miniconda3/envs/nanotron

echo ""
echo "=== Verifying ==="
python -c "import flash_attn; print(f'flash-attn {flash_attn.__version__} installed successfully')"
python -c "from flash_attn.flash_attn_interface import flash_attn_varlen_func; print('flash-attn CUDA extension loads OK')"
python -c "import grouped_gemm.ops as ops; print('grouped_gemm loads OK')"

echo ""
echo "=== Done ==="
