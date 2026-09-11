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

VENV_DIR="${REPO_ROOT}/.venv-${GPU_TUNED_VARIANT}"
[[ -d "${VENV_DIR}" ]] || {
    echo "ERROR: ${VENV_DIR} not found. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2
    exit 1
}
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
# tuning-vN = commits on tuned-builds since it diverged from main (i.e.
# commits ahead of upstream/Dao-AILab) -- same convention adopted
# fleet-wide from zbrad/pytorch's tuned/wheel.sh: the static __version__
# above only moves when upstream bumps it, so on its own it can't say
# "how much of our own tuned-builds work landed since an earlier wheel
# was built."
TUNED_COMMIT_COUNT="$(git rev-list --count main..HEAD)"
export FLASH_ATTN_LOCAL_VERSION="${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.tuning-v${TUNED_COMMIT_COUNT}"

echo "=========================================="
echo "Packaging flash_attn wheel (${GPU_TUNED_HW_LABEL})"
echo "=========================================="
echo "FLASH_ATTN_LOCAL_VERSION: ${FLASH_ATTN_LOCAL_VERSION}"
echo ""

# flash_attn_2_cuda.*.so is where the actual device code lands -- verify +
# stamp it before packaging, same discipline as the other tuned-builds
# repos' wheel.sh.
FA2_SO="$(find "${REPO_ROOT}/build" -maxdepth 4 -name 'flash_attn_2_cuda*.so' 2>/dev/null | head -1)"
if [[ -z "${FA2_SO}" ]]; then
    # editable installs (build.sh's -e .) may instead place it directly
    # under the package dir depending on setuptools version.
    FA2_SO="$(find "${REPO_ROOT}" -maxdepth 2 -name 'flash_attn_2_cuda*.so' 2>/dev/null | head -1)"
fi
if [[ -n "${FA2_SO}" ]]; then
    gpu_tuned_verify_arch "${FA2_SO}" "${GPU_TUNED_FA_ARCH}"
    embed_build_info "${FA2_SO}" "${GPU_TUNED_VARIANT}" "flash_attn" "${FLASH_ATTN_LOCAL_VERSION}" "${GPU_TUNED_HW_LABEL}"
else
    echo "ERROR: flash_attn_2_cuda*.so not found anywhere under ${REPO_ROOT} -- run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2
    exit 1
fi

pip install --upgrade build
rm -rf "${REPO_ROOT}/dist"
python3 -m build --wheel --no-isolation "${REPO_ROOT}"

WHEEL="$(ls "${REPO_ROOT}"/dist/flash_attn-*.whl 2>/dev/null | head -1)"
[[ -z "${WHEEL}" ]] && { echo "ERROR: no wheel found in dist/" >&2; exit 1; }
echo "Built wheel: $(basename "${WHEEL}") ($(du -sh "${WHEEL}" | awk '{print $1}'))"

WHEEL_VERSION="$(gpu_tuned_wheel_version "${WHEEL}" flash_attn)" || exit 1
WHEEL_BASE_VERSION="${WHEEL_VERSION%%+*}"

RELEASE_TAG="v${WHEEL_BASE_VERSION}-${GPU_TUNED_VARIANT}-cu${CUDA_VERSION_COMPACT}"
RELEASE_TITLE="flash_attn ${WHEEL_VERSION} — ${GPU_TUNED_HW_LABEL} wheel"

echo ""
echo "Publishing wheel to GitHub release ${RELEASE_TAG}..."
gpu_tuned_publish_release "zbrad-llc/flash-attention" "${RELEASE_TAG}" "${RELEASE_TITLE}" \
    "flash_attn ${WHEEL_VERSION} wheel for ${GPU_TUNED_HW_LABEL}, single-arch (FLASH_ATTN_CUDA_ARCHS=${GPU_TUNED_FA_ARCH})." \
    "${WHEEL}#$(basename "${WHEEL}")"

echo ""
echo "Release: https://github.com/ZBrad-LLC/flash-attention/releases/tag/${RELEASE_TAG}"
echo "Done."
