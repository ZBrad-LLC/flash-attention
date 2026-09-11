#!/bin/bash
# tuned/build.sh <variant> — build/install flash_attn from source for a
# single GPU variant (gb10/rtx40/rtx50) only, single-arch, into a
# per-variant editable venv.
#
# This builds the standalone Dao-AILab flash_attn package (root setup.py,
# FLASH_ATTN_CUDA_ARCHS) -- NOT the vllm fork's vllm_flash_attn. Only the
# main FA2 package is built; hopper/ (FA3) is a separate package requiring
# sm_90 (Hopper) and is out of scope for gb10/rtx40/rtx50, all of which are
# Ampere/Ada/Blackwell.
#
# Unlike the vllm fork, this repo's setup.py install_requires just says
# "torch" (unpinned) -- no exact-version conflict with a protected
# GB10-tuned torch pin, so no --no-deps workaround is needed here.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

command -v python3 &>/dev/null || { echo "ERROR: python3 not found on PATH." >&2; exit 1; }

# "-base" distinguishes this from zbrad/flash-attention-vllm's own
# .venv-<variant>-vllm -- same variant name, sibling repo, previously both
# bare ".venv-<variant>" and easy to target by mistake (hit this session:
# a venv literally copied cross-repo, and a manual pip install landing in
# the wrong one).
VENV_DIR="${REPO_ROOT}/.venv-${GPU_TUNED_VARIANT}-base"
if [[ "${GPU_TUNED_CLEAN_VENV:-}" == "1" && -d "${VENV_DIR}" ]]; then
    echo "GPU_TUNED_CLEAN_VENV=1: removing ${VENV_DIR} for a clean rebuild"
    rm -rf "${VENV_DIR}"
fi
if [[ -d "${VENV_DIR}" ]]; then
    # Reusing an existing venv by default -- verify it wasn't
    # copied/contaminated from another repo before building on top of it.
    # Set GPU_TUNED_CLEAN_VENV=1 to force a fresh venv instead.
    gpu_tuned_verify_venv "${VENV_DIR}" "${REPO_ROOT}"
else
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# Deliberately pinned rather than left to setup.py's psutil-based
# auto-heuristic, so a recorded build time is reproducible independent of
# whatever else is using host RAM/GPU memory at build time. 20 logical
# cores / 121GB unified memory on this GB10 box; MAX_JOBS=5 * NVCC_THREADS=4
# == 20, matching nproc, same values the auto-heuristic would have picked
# under low memory pressure.
export MAX_JOBS="${MAX_JOBS:-5}"
export NVCC_THREADS="${NVCC_THREADS:-4}"

echo "=========================================="
echo "Building flash_attn for ${GPU_TUNED_HW_LABEL} only"
echo "=========================================="
echo "FLASH_ATTN_CUDA_ARCHS: ${FLASH_ATTN_CUDA_ARCHS}"
echo "MAX_JOBS:              ${MAX_JOBS}"
echo "NVCC_THREADS:          ${NVCC_THREADS}"
echo "CUDA_HOME:              ${CUDA_HOME:-<unset>}"
echo "Python:                 $(python3 --version)"
echo "Git commit:             $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo ""

if ! python3 -c "import torch" &>/dev/null; then
    if [[ "${GPU_TUNED_VARIANT}" == "gb10" ]]; then
        echo "ERROR: no torch installed in ${VENV_DIR}, and gb10 has no official" >&2
        echo "       upstream aarch64/sm_121 torch wheel. Install zbrad/pytorch's" >&2
        echo "       own tuned-builds gb10 wheel into this venv first:" >&2
        echo "       ${VENV_DIR}/bin/pip install <path-or-URL-to-zbrad-pytorch-gb10-wheel>" >&2
        exit 1
    else
        echo "No torch found -- installing latest PyPI CUDA torch..."
        pip install --upgrade pip
        pip install torch
    fi
fi
python3 -c "import torch; print(f'Using torch {torch.__version__} (CUDA {torch.version.cuda})')"

echo "Installing other build-time requirements (psutil, packaging, setuptools, wheel, ninja)..."
pip install psutil packaging "setuptools>=49.4.0" wheel ninja einops

echo "Building flash_attn (this will take a long time)..."
pip install --no-build-isolation -v -e . 2>&1

echo ""
echo "Smoke test: import flash_attn, confirm the real (non-shim) extension loads..."
python3 - <<'PYEOF'
import torch
import flash_attn
from flash_attn import flash_attn_func

print(f"flash_attn imported OK from {flash_attn.__file__} (version {flash_attn.__version__})")
import flash_attn_2_cuda  # noqa: F401 -- the actual compiled extension; ImportError here means the build didn't really produce it
print("flash_attn_2_cuda extension present and importable.")
assert torch.cuda.is_available(), "CUDA device not available -- cannot exercise a real kernel call here"
PYEOF

echo ""
echo "=========================================="
echo "Build complete (${GPU_TUNED_HW_LABEL})."
echo "=========================================="
echo "Venv: ${VENV_DIR}"
echo "Next: bash tuned/wheel.sh ${GPU_TUNED_VARIANT}"
