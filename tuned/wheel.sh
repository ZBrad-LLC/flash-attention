#!/bin/bash
# tuned/wheel.sh <variant> — package a built flash_attn tree into a wheel
# and publish it as a real GitHub release. Requires tuned/build.sh
# <variant> to have already succeeded (this reuses that venv, does not
# rebuild).
set -euo pipefail

GPU_TUNED_ARG_VARIANT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

VENV_DIR="${REPO_ROOT}/.venv-${GPU_TUNED_VARIANT}-base"
[[ -d "${VENV_DIR}" ]] || {
    echo "ERROR: ${VENV_DIR} not found. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2
    exit 1
}
gpu_tuned_verify_venv "${VENV_DIR}" "${REPO_ROOT}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

[[ -n "${CUDA_VERSION_COMPACT:-}" ]] || {
    echo "ERROR: CUDA_VERSION_COMPACT not set (CUDA_HOME must resolve to a" \
         "/usr/local/cuda-X.Y directory) -- cannot derive the version string." >&2
    exit 1
}

# setup.py's get_package_version() reads __version__ from flash_attn/__init__.py
# and appends "+FLASH_ATTN_LOCAL_VERSION" when set -- set it explicitly so
# multiple tuned variants at the same CUDA version don't collide on an
# identical wheel filename (same convention as every other repo's
# wheel.sh this session).
#
# tuning.N = commits on tuned-builds since it diverged from main (i.e.
# commits ahead of upstream/Dao-AILab) -- same convention adopted
# fleet-wide from zbrad/pytorch's tuned/wheel.sh: the static __version__
# above only moves when upstream bumps it, so on its own it can't say
# "how much of our own tuned-builds work landed since an earlier wheel
# was built."
TUNED_COMMIT_COUNT="$(gpu_tuned_tuning_count)"
FLASH_ATTN_LOCAL_VERSION="$(gpu_tuned_local_version "${GPU_TUNED_VARIANT}" "${CUDA_VERSION_COMPACT}" "${TUNED_COMMIT_COUNT}")"
export FLASH_ATTN_LOCAL_VERSION

echo "=========================================="
echo "Packaging flash_attn wheel (${GPU_TUNED_HW_LABEL})"
echo "=========================================="
echo "FLASH_ATTN_LOCAL_VERSION: ${FLASH_ATTN_LOCAL_VERSION}"
echo ""

pip install --upgrade build wheel
rm -rf "${REPO_ROOT}/dist"
python3 -m build --wheel --no-isolation "${REPO_ROOT}"

WHEEL="$(ls "${REPO_ROOT}"/dist/flash_attn-*.whl 2>/dev/null | head -1)"
[[ -z "${WHEEL}" ]] && { echo "ERROR: no wheel found in dist/" >&2; exit 1; }
echo "Built wheel: $(basename "${WHEEL}") ($(du -sh "${WHEEL}" | awk '{print $1}'))"

WHEEL_VERSION="$(gpu_tuned_wheel_version "${WHEEL}" flash_attn)" || exit 1

# flash_attn_2_cuda.*.so is where the actual device code lands -- verify +
# stamp it here, on the wheel's OWN contents, not a pre-build copy: a
# separate packaging invocation (python -m build) may rebuild/relink
# rather than reuse an already-stamped file byte-for-byte -- confirmed
# for pytorch's equivalent build this session (a test marker stamped
# before the build was completely absent afterward; see
# gpu_tuned_verify_build_info's header comment). Applying the same
# proven-safe pattern here rather than assuming setuptools' incremental
# behavior is different in a way that happens to help. Unpack -> stamp ->
# repack regenerates RECORD correctly, unlike a raw zip edit.
echo "Stamping build-info into the wheel's own flash_attn_2_cuda*.so"
UNPACK_DIR="$(mktemp -d)"
python3 -m wheel unpack "${WHEEL}" --dest "${UNPACK_DIR}"
WHEEL_SO="$(find "${UNPACK_DIR}" -name 'flash_attn_2_cuda*.so' | head -1)"
[[ -z "${WHEEL_SO}" ]] && { echo "ERROR: flash_attn_2_cuda*.so not found inside ${WHEEL}." >&2; exit 1; }
gpu_tuned_verify_arch "${WHEEL_SO}" "${GPU_TUNED_FA_ARCH}"
embed_build_info "${WHEEL_SO}" "${GPU_TUNED_VARIANT}" "flash_attn" "${WHEEL_VERSION}" "${GPU_TUNED_HW_LABEL}"
gpu_tuned_verify_build_info "${WHEEL_SO}" "flash_attn" "${WHEEL_VERSION}" "flash_attn_build_info"
rm -f "${WHEEL}"
UNPACKED_CONTENT_DIR="$(find "${UNPACK_DIR}" -maxdepth 1 -mindepth 1 -type d)"
python3 -m wheel pack "${UNPACKED_CONTENT_DIR}" --dest-dir "${REPO_ROOT}/dist"
rm -rf "${UNPACK_DIR}"
WHEEL="$(ls "${REPO_ROOT}"/dist/flash_attn-*.whl 2>/dev/null | head -1)"
echo "Re-packed with build-info stamp: $(basename "${WHEEL}")"
# WHEEL_VERSION already includes the full "+FLASH_ATTN_LOCAL_VERSION" local
# segment (variant, cuda tag, and now tuning.N) -- use it directly rather
# than stripping and re-appending only part of it, which would silently
# drop tuning.N from the tag while it stayed visible in the title below.
# Friendly title only -- RELEASE_TAG stays the exact WHEEL_VERSION.
GIT_SHA="$(git rev-parse --short HEAD)"
WHEEL_BASE_VERSION="${WHEEL_VERSION%%+*}"

RELEASE_TAG="v${WHEEL_VERSION}"
RELEASE_TITLE="flash_attn ${WHEEL_BASE_VERSION} — ${GPU_TUNED_VARIANT} tuning.${TUNED_COMMIT_COUNT} (cu${CUDA_VERSION_COMPACT}, ${GIT_SHA}) — ${GPU_TUNED_HW_LABEL} wheel"

echo ""
echo "Publishing wheel to GitHub release ${RELEASE_TAG}..."
gpu_tuned_publish_release "zbrad-llc/flash-attention" "${RELEASE_TAG}" "${RELEASE_TITLE}" \
    "flash_attn ${WHEEL_VERSION} wheel for ${GPU_TUNED_HW_LABEL}, single-arch (FLASH_ATTN_CUDA_ARCHS=${GPU_TUNED_FA_ARCH})." \
    "${WHEEL}#$(basename "${WHEEL}")"

echo ""
echo "Release: https://github.com/ZBrad-LLC/flash-attention/releases/tag/${RELEASE_TAG}"
echo "Done."
