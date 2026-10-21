#!/bin/bash
# tuned/env.sh <variant> — device, toolkit and build-environment setup for a tuned
# single-arch flash_attn build (gb10/rtx40/rtx50). Source this file with the
# variant as $1; do not execute it directly.
#
#   CUDA_TAG=cu134 bash tuned/build.sh gb10     (or CUDA_VER=13.4)
#
# This is Dao-AILab/flash-attention (the standalone flash_attn package,
# root setup.py) -- NOT zbrad/flash-attention-vllm's vllm_flash_attn.
# Different arch-flag scheme: this repo's setup.py reads
# FLASH_ATTN_CUDA_ARCHS as a semicolon-separated list of plain integer
# codes ("80;90;100;110;120;121"), not the vllm fork's "12.1a"-style
# family-suffix string. The code per variant is GPU_TUNED_FA_ARCH in
# tuned/repo.env; ../SM121_SUPPORT.md covers how "121" became buildable.
#
# Device facts come from tuned/common-data/devices/<variant>.conf and the CUDA
# toolkit from tuned/common-data/toolkits/<id>.env (both vendored from
# zbrad/tuned-common by sync-common.sh), never from /usr/local/cuda or from what
# happens to be installed. There is no default: CUDA_TAG or CUDA_VER must name
# one. The environment is derived on every run and never saved; what a build
# used is recorded in its build info (package.sh).
#
# Exported: GPU_TUNED_VARIANT/PLATFORM/CUDA_ARCH/HW_LABEL (shared), GPU_TUNED_FA_ARCH
# (repo.env), CUDA_HOME/CUDA_TAG/CUDA_VER (the toolkit), CUDA_VERSION_COMPACT,
# FLASH_ATTN_CUDA_ARCHS, GPU_TUNED_BUILD_ROOT (tuned/build/<cuda-tag>), and
# GPU_TUNED_VENV_DIR, GPU_TUNED_WHEEL_DIR / GPU_TUNED_DIST_DIR / GPU_TUNED_RELEASES_DIR under it.

GPU_TUNED_ARG_VARIANT="${1:-}"
if [[ -z "${GPU_TUNED_ARG_VARIANT}" ]]; then
    echo "ERROR: env.sh requires a variant argument (gb10/rtx40/rtx50)" >&2
    return 1 2>/dev/null || exit 1
fi

GPU_TUNED_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_TUNED_DATA_DIR="${GPU_TUNED_SELF_DIR}/common-data"
if [[ ! -f "${GPU_TUNED_DATA_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" ]]; then
    echo "ERROR: no device '${GPU_TUNED_ARG_VARIANT}' in ${GPU_TUNED_DATA_DIR}/devices (have: $(cd "${GPU_TUNED_DATA_DIR}/devices" && ls | sed 's/\.conf$//' | tr '\n' ' '))" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck source=common-data/devices/gb10.conf
source "${GPU_TUNED_DATA_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" || return 1 2>/dev/null || exit 1
export GPU_TUNED_VARIANT GPU_TUNED_PLATFORM GPU_TUNED_CUDA_ARCH GPU_TUNED_HW_LABEL

# shellcheck source=common.sh
# Vendored from https://github.com/zbrad/tuned-common (pinned commit --
# see common.sh's own header/sync instructions to update).
source "${GPU_TUNED_SELF_DIR}/common.sh" || return 1 2>/dev/null || exit 1

gpu_tuned_assert_platform "${GPU_TUNED_PLATFORM}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1

# --- CUDA toolkit: CUDA_TAG (cu134) or CUDA_VER (13.4) name one from common-data/toolkits. ---
gpu_tuned_select_toolkit "${GPU_TUNED_DATA_DIR}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1

# --- Repo-specific settings (loaded after the shared device and toolkit values; may not change them). ---
gpu_tuned_load_repo_env "${GPU_TUNED_SELF_DIR}/repo.env" || return 1 2>/dev/null || exit 1
# A sourced file's own `return 1` does not stop the loader, so check that repo.env really set its key.
if [[ -z "${GPU_TUNED_FA_ARCH:-}" ]]; then
    echo "ERROR: tuned/repo.env did not set GPU_TUNED_FA_ARCH for ${GPU_TUNED_VARIANT}." >&2
    return 1 2>/dev/null || exit 1
fi
export GPU_TUNED_FA_ARCH GPU_TUNED_UPSTREAM_REF

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${GPU_TUNED_TOOLKIT_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_PATH="${CUDA_HOME}" CUDACXX="${CUDA_HOME}/bin/nvcc"
# CUDA_VERSION_COMPACT (e.g. "134" for CUDA 13.4) is the toolkit id's digits.
CUDA_VERSION_COMPACT="${CUDA_TAG#cu}"
export CUDA_VERSION_COMPACT

# --- The one build tree for this toolkit (docs/ARCHITECTURE.md C1/C2): tuned/build/<cuda-tag>/. ---
GPU_TUNED_REPO_ROOT="$(cd "${GPU_TUNED_SELF_DIR}/.." && pwd)"
# A tree last used by another variant is refused, never mixed (gpu_tuned_claim_toolkit_tree).
gpu_tuned_claim_toolkit_tree "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1
GPU_TUNED_BUILD_ROOT="${GPU_TUNED_REPO_ROOT}/tuned/build/${CUDA_TAG}"
# The venv stays at the repo root (not in the build tree) so existing venvs keep their torch install.
GPU_TUNED_VENV_DIR="${GPU_TUNED_REPO_ROOT}/.venv-${GPU_TUNED_VARIANT}-base"
GPU_TUNED_WHEEL_DIR="${GPU_TUNED_BUILD_ROOT}/wheel"
GPU_TUNED_DIST_DIR="$(gpu_tuned_out_dir dist "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}")"
GPU_TUNED_RELEASES_DIR="$(gpu_tuned_out_dir releases "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}")"
export GPU_TUNED_BUILD_ROOT GPU_TUNED_VENV_DIR GPU_TUNED_WHEEL_DIR GPU_TUNED_DIST_DIR GPU_TUNED_RELEASES_DIR

export FLASH_ATTN_CUDA_ARCHS="${GPU_TUNED_FA_ARCH}"
# Force a real compile -- setup.py's CachedWheelsCommand otherwise tries a
# prebuilt-wheel download first (wrong arch for any of these variants).
export FLASH_ATTENTION_FORCE_BUILD=TRUE
# No PTX, no JIT fallback -- a tuned build only ever runs on the exact GPU
# it was built for; embedding forward-compat PTX would let it silently
# JIT onto a different SM instead of failing loudly. See setup.py's
# add_cuda_gencodes() comment on this flag (zbrad/flash-attention-only
# addition, not upstream).
export FLASH_ATTN_NO_PTX=TRUE

echo "[tuned/env] GPU_TUNED_VARIANT=${GPU_TUNED_VARIANT} FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS} CUDA_HOME=${CUDA_HOME} tree=${GPU_TUNED_BUILD_ROOT}"
