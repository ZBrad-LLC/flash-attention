#!/bin/bash
# tuned/build.sh <variant> — build the flash_attn wheel for one GPU variant (gb10/rtx40/rtx50) and one CUDA
# toolkit, single-arch. It prepares the variant's build venv (.venv-<variant>-base), checks that a torch is
# installed in it, then makes the unstamped wheel with tuned/wheel.sh. The compile is the long step.
#
#   CUDA_TAG=cu134 bash tuned/build.sh gb10
#
# Stage order (docs/ARCHITECTURE.md B3): build.sh -> full_test.sh -> package.sh -> smoke_test.sh ->
# (review and commit the release notes) -> release.sh -> publish.sh.
#
# This builds the standalone Dao-AILab flash_attn package (root setup.py, FLASH_ATTN_CUDA_ARCHS) -- NOT the
# vllm fork's vllm_flash_attn. Only the main FA2 package is built; hopper/ (FA3) is a separate package
# requiring sm_90 (Hopper) and is out of scope for gb10/rtx40/rtx50, all of which are Ampere/Ada/Blackwell.
#
# Unlike the vllm fork, this repo's setup.py install_requires just says "torch" (unpinned) -- no exact-version
# conflict with a protected GB10-tuned torch pin, so no --no-deps workaround is needed here.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/build.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

command -v python3 &>/dev/null || { echo "ERROR: python3 not found on PATH." >&2; exit 1; }

# "-base" distinguishes this from zbrad/flash-attention-vllm's own .venv-<variant>-vllm -- same variant name,
# sibling repo, previously both bare ".venv-<variant>" and easy to target by mistake.
VENV_DIR="${GPU_TUNED_VENV_DIR}"
if [[ "${GPU_TUNED_CLEAN_VENV:-}" == "1" && -d "${VENV_DIR}" ]]; then
    echo "GPU_TUNED_CLEAN_VENV=1: removing ${VENV_DIR} for a clean rebuild"
    rm -rf "${VENV_DIR}"
fi
if [[ -d "${VENV_DIR}" ]]; then
    # Reusing an existing venv by default -- verify it wasn't copied/contaminated from another repo before
    # building on top of it. Set GPU_TUNED_CLEAN_VENV=1 to force a fresh venv instead.
    gpu_tuned_verify_venv "${VENV_DIR}" "${REPO_ROOT}"
else
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# Deliberately pinned rather than left to setup.py's psutil-based auto-heuristic, so a recorded build time is
# reproducible independent of whatever else is using host RAM/GPU memory at build time. 20 logical cores /
# 121GB unified memory on this GB10 box; MAX_JOBS=5 * NVCC_THREADS=4 == 20, matching nproc.
export MAX_JOBS="${MAX_JOBS:-5}"
export NVCC_THREADS="${NVCC_THREADS:-4}"

echo "=========================================="
echo "Building flash_attn for ${GPU_TUNED_HW_LABEL} only"
echo "=========================================="
echo "Toolkit:               ${CUDA_TAG} (${CUDA_HOME})"
echo "FLASH_ATTN_CUDA_ARCHS: ${FLASH_ATTN_CUDA_ARCHS}"
echo "MAX_JOBS:              ${MAX_JOBS} (NVCC_THREADS=${NVCC_THREADS})"
echo "Venv:                  ${VENV_DIR}"
echo "Python:                $(python3 --version)"
echo "Git commit:            $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
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

echo "Installing other build-time requirements (psutil, packaging, setuptools, wheel, ninja, build)..."
pip install psutil packaging "setuptools>=49.4.0" wheel ninja einops build

echo ""
bash "${REPO_ROOT}/tuned/wheel.sh" "${GPU_TUNED_VARIANT}"

echo ""
echo "=========================================="
echo "Build complete (${GPU_TUNED_HW_LABEL})."
echo "=========================================="
echo "Label:        $(cat "${GPU_TUNED_BUILD_ROOT}/release-label")"
echo "Build commit: $(cat "${GPU_TUNED_BUILD_ROOT}/build-commit")"
echo "Next: CUDA_TAG=${CUDA_TAG} bash tuned/full_test.sh ${GPU_TUNED_VARIANT}"
