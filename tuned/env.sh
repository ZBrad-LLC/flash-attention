#!/bin/bash
# tuned/env.sh <variant> — device config + build-env setup for a tuned
# single-arch flash_attn build (gb10/rtx40/rtx50). Source this file with
# the variant as $1; do not execute it directly.
#
# This is Dao-AILab/flash-attention (the standalone flash_attn package,
# root setup.py) -- NOT zbrad/flash-attention-vllm's vllm_flash_attn.
# Different arch-flag scheme: this repo's setup.py reads
# FLASH_ATTN_CUDA_ARCHS as a semicolon-separated list of plain integer
# codes ("80;90;100;110;120;121"), not the vllm fork's "12.1a"-style
# family-suffix string forwarded as CUDA_ARCHS/FA2_TUNED_ARCH. See
# tuned/devices/*.conf for the per-variant code and
# ../SM121_SUPPORT.md for how "121" became buildable at all.
#
# Exported: GPU_TUNED_VARIANT/PLATFORM/FA_ARCH/HW_LABEL (from
# tuned/devices/<variant>.conf), CUDA_HOME (autodetected highest installed
# toolkit), CUDA_VERSION_COMPACT, FLASH_ATTN_CUDA_ARCHS.

GPU_TUNED_ARG_VARIANT="$1"
if [[ -z "${GPU_TUNED_ARG_VARIANT}" ]]; then
    echo "ERROR: env.sh requires a variant argument (gb10/rtx40/rtx50)" >&2
    return 1 2>/dev/null || exit 1
fi

GPU_TUNED_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=devices/rtx50.conf
source "${GPU_TUNED_SELF_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" || return 1 2>/dev/null || exit 1
export GPU_TUNED_VARIANT GPU_TUNED_PLATFORM GPU_TUNED_FA_ARCH GPU_TUNED_HW_LABEL

# shellcheck source=common.sh
# Vendored from https://github.com/zbrad/tuned-common (pinned commit --
# see common.sh's own header/sync instructions to update).
source "${GPU_TUNED_SELF_DIR}/common.sh" || return 1 2>/dev/null || exit 1

gpu_tuned_assert_platform "${GPU_TUNED_PLATFORM}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1

# --- Resolve CUDA_HOME to the highest installed toolkit when not explicitly set ---
if [ -z "${CUDA_HOME:-}" ]; then
    _fa_highest="$(gpu_tuned_installed_cuda_toolkits | tail -1)"
    if [ -n "$_fa_highest" ]; then
        export CUDA_HOME="/usr/local/cuda-${_fa_highest}"
    else
        echo "[tuned/env] WARNING: no /usr/local/cuda-<ver> toolkit found; leaving CUDA_HOME unset." >&2
        echo "[tuned/env]          Set CUDA_HOME explicitly to an installed toolkit." >&2
    fi
    unset _fa_highest
fi
[ -n "${CUDA_HOME:-}" ] && export PATH="$CUDA_HOME/bin:$PATH"

if [ -n "${CUDA_HOME:-}" ]; then
    CUDA_VERSION_COMPACT="$(basename "$CUDA_HOME" | sed -E 's/^cuda-([0-9]+)\.([0-9]+).*/\1\2/')"
    export CUDA_VERSION_COMPACT
fi

# gb10 (sm_121) needs CUDA >= 13.0 for the compute_121f gencode added by
# the cherry-picked PR -- fail loudly here rather than deep into an nvcc
# invocation with a silently-skipped gencode.
if [[ "${GPU_TUNED_VARIANT}" == "gb10" && -n "${CUDA_VERSION_COMPACT:-}" && "${CUDA_VERSION_COMPACT}" -lt 130 ]]; then
    echo "ERROR: gb10 (sm_121) needs CUDA >= 13.0, found ${CUDA_HOME}" >&2
    return 1 2>/dev/null || exit 1
fi

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

echo "[tuned/env] GPU_TUNED_VARIANT=${GPU_TUNED_VARIANT} FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS} CUDA_HOME=${CUDA_HOME:-<unset>}"

# embed_build_info <so_path> <variant> <package> <version> [hw_label] —
# thin wrapper over gpu_tuned_embed_build_info (common.sh), pinning the
# section name to .flash_attn_build_info.
embed_build_info() {
    local so_path="$1" variant="$2" package="$3" version="$4" hw_label="$5"
    gpu_tuned_embed_build_info "${so_path}" "${variant}" "${package}" "${version}" \
        "${hw_label}" "https://github.com/ZBrad-LLC/flash-attention" "flash_attn_build_info"
}
