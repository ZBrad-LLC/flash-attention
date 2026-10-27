#!/bin/bash
# tuned/package.sh <variant> — package the wheel tuned/build.sh made and tuned/full_test.sh tested: stamp a
# STAGED copy of it with the JSON build info (docs/ARCHITECTURE.md H7), repack, write the sha256 sidecar,
# package-info.txt and, if no notes are tracked yet, a DRAFT release-notes file. It never builds, tests,
# tags or publishes (B3). Stage order:
#   build.sh -> full_test.sh -> package.sh -> smoke_test.sh -> (review and commit the notes) -> release.sh -> publish.sh
#
#   CUDA_TAG=cu134 bash tuned/package.sh gb10
#
# Input:  tuned/build/<cuda-tag>/wheel/flash_attn-*.whl (unstamped; never modified here, H3), release-label and
#         build-commit from wheel.sh, and the passing test log from full_test.sh.
# Output: tuned/build/<cuda-tag>/dist/{flash_attn-*.whl, flash_attn-*.whl.sha256, package-info.txt} and
#         tuned/releases/RELEASE_NOTES_<version>.md (DRAFT) when no notes exist. package-info.txt is written last
#         (H2): its presence means every check above it passed.
#
# Stamped (H7): flash_attn_2_cuda*.so, the one extension this repo builds. The stamp records the build inputs
# that shaped the binary (the allowlist ENV_RECORD below: arch code, build flags, the vendored tuned-common
# commit), never the whole environment. The environment itself is derived on each run and not saved.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/package.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

[[ -d "${GPU_TUNED_VENV_DIR}" ]] || { echo "ERROR: ${GPU_TUNED_VENV_DIR} not found. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
gpu_tuned_verify_venv "${GPU_TUNED_VENV_DIR}" "${REPO_ROOT}"
# The wheel was built in this venv: its setuptools must meet the common minimum (H10, tuned-common).
gpu_tuned_check_venv_setuptools "${GPU_TUNED_VENV_DIR}" || exit 1
# shellcheck source=/dev/null
source "${GPU_TUNED_VENV_DIR}/bin/activate"

BUILT="$(ls "${GPU_TUNED_WHEEL_DIR}"/flash_attn-*.whl 2>/dev/null | head -1)"
[[ -n "${BUILT}" ]] || { echo "ERROR: no wheel in ${GPU_TUNED_WHEEL_DIR}. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
LABEL="$(cat "${GPU_TUNED_BUILD_ROOT}/release-label" 2>/dev/null || true)"
BUILD_COMMIT="$(cat "${GPU_TUNED_BUILD_ROOT}/build-commit" 2>/dev/null || true)"
[[ -n "${LABEL}" && -n "${BUILD_COMMIT}" ]] || { echo "ERROR: release-label or build-commit missing in ${GPU_TUNED_BUILD_ROOT}: the last build did not finish." >&2; exit 1; }
[[ "${BUILD_COMMIT}" != *-dirty ]] || { echo "ERROR: the build was made from a dirty tree (${BUILD_COMMIT}); commit and rebuild." >&2; exit 1; }

# The label is the one wheel.sh allocated and baked in; it is never recomputed here (E4).
BUILT_NAME="$(basename "${BUILT}")"
VERSION="$(gpu_tuned_wheel_version "${BUILT}" flash_attn)"
[[ "${VERSION}" == *"+${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL}" ]] || {
    echo "ERROR: ${BUILT_NAME} is version ${VERSION}, not carrying +${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL} from release-label." >&2
    exit 1
}
gpu_tuned_check_wheel_label "${BUILT}"
BASE_VERSION="${VERSION%%+*}"
RELEASE_TAG="v${VERSION}"
gpu_tuned_check_release_tag "${RELEASE_TAG}"

# H6: refuse a build from other source than HEAD; HEAD may differ from the build only in tuned/.
git cat-file -e "${BUILD_COMMIT}^{commit}" 2>/dev/null || { echo "ERROR: build commit ${BUILD_COMMIT} is not in this checkout." >&2; exit 1; }
STALE_SRC="$(git diff --name-only "${BUILD_COMMIT}" HEAD -- . ':!tuned' | head -5)"
if [[ -n "${STALE_SRC}" ]]; then
    echo "ERROR: the wheel is from ${BUILD_COMMIT}, but source outside tuned/ changed since:" >&2
    echo "${STALE_SRC}" | sed 's/^/    /' >&2
    echo "  Rebuild and re-test before packaging." >&2
    exit 1
fi
TEST_LOG="${GPU_TUNED_RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
[[ -f "${TEST_LOG}" ]] || { echo "ERROR: ${TEST_LOG} not found; run CUDA_TAG=${CUDA_TAG} bash tuned/full_test.sh ${GPU_TUNED_VARIANT} before packaging." >&2; exit 1; }
grep -q "^=== Full test suite summary: gates + numeric check passed ===\$" "${TEST_LOG}" \
    || { echo "ERROR: ${TEST_LOG} does not show a full, passing run." >&2; exit 1; }
[[ "${BUILT}" -nt "${TEST_LOG}" ]] && { echo "ERROR: ${BUILT_NAME} is newer than the test log; re-run tuned/full_test.sh ${GPU_TUNED_VARIANT}." >&2; exit 1; }
TESTED_COMMIT="$(sed -n 's/^commit: *//p' "${TEST_LOG}" | head -1)"
TESTED_SHA="$(sed -n 's/^wheel:.*sha256 //p' "${TEST_LOG}" | head -1)"
[[ "${TESTED_SHA}" == "$(sha256sum "${BUILT}" | awk '{print $1}')" ]] || { echo "ERROR: the test log tested a different wheel (sha256 ${TESTED_SHA:-none}) than ${BUILT_NAME}." >&2; exit 1; }

echo "=========================================="
echo "Packaging flash_attn wheel (${GPU_TUNED_HW_LABEL})"
echo "=========================================="
echo "Version:     ${VERSION}"
echo "Release tag: ${RELEASE_TAG}"
echo "Built at:    ${BUILD_COMMIT}; tested at ${TESTED_COMMIT}; packaging at $(git rev-parse HEAD)"
echo ""

mkdir -p "${GPU_TUNED_DIST_DIR}"
# package-info.txt is removed first and written last: its presence means every check below passed.
rm -f "${GPU_TUNED_DIST_DIR}/package-info.txt" "${GPU_TUNED_DIST_DIR}"/flash_attn-*.whl "${GPU_TUNED_DIST_DIR}"/flash_attn-*.whl.sha256

command -v readelf >/dev/null && command -v objcopy >/dev/null && command -v cuobjdump >/dev/null \
    || { echo "ERROR: readelf, objcopy and cuobjdump are needed for the packaging checks (H4)." >&2; exit 1; }

echo "Stamping build info into a staged copy of the wheel"
UNPACK_DIR="$(mktemp -d)"
trap 'rm -rf "${UNPACK_DIR}"' EXIT
python3 -m wheel unpack "${BUILT}" --dest "${UNPACK_DIR}"
UNPACKED="$(find "${UNPACK_DIR}" -maxdepth 1 -mindepth 1 -type d)"
[[ -d "${UNPACKED}" ]] || { echo "ERROR: ${BUILT_NAME} did not unpack to one directory." >&2; exit 1; }

# No stock torch shared-library name in the wheel: flash_attn ships none, so a stock name would be a leak from elsewhere.
STOCK_RE='^lib(c10|c10_cuda|torch|torch_cpu|torch_cuda|torch_cuda_linalg|torch_python|torch_global_deps|caffe2_nvrtc|shm)\.so(\.[0-9]+)*$'
leaks="$(find "${UNPACKED}" -type f -name '*.so*' -exec basename {} \; | grep -E "${STOCK_RE}" || true)"
[[ -z "${leaks}" ]] || { echo "ERROR: the wheel ships shared objects under stock torch names: ${leaks}" >&2; exit 1; }

# The Python packages `pip install <wheel>` pulls in: none may be a known-vulnerable version that has a fix (H9, tuned-common;
# needs the network, and a check that cannot run is a failure). Advisories can be accepted in tuned/audit_allow.txt.
gpu_tuned_audit_deps "${BUILT}" "${GPU_TUNED_BUILD_ROOT}/dependency-audit.log" || { echo "ERROR: dependency audit failed (see above and ${GPU_TUNED_BUILD_ROOT}/dependency-audit.log)." >&2; exit 1; }

mapfile -t STAMP_LIBS < <(find "${UNPACKED}" -maxdepth 1 -type f -name 'flash_attn_2_cuda*.so')
[[ ${#STAMP_LIBS[@]} -eq 1 ]] || { echo "ERROR: expected one flash_attn_2_cuda*.so at the top of ${BUILT_NAME}, found ${#STAMP_LIBS[@]}." >&2; exit 1; }
for so in "${STAMP_LIBS[@]}"; do
    gpu_tuned_so_has_device_code "${so}" || { echo "ERROR: ${so##*/} has no device code: the stamp would not describe a GPU build." >&2; exit 1; }
    gpu_tuned_verify_arch "${so}" "${GPU_TUNED_FA_ARCH}" || exit 1
    # Pre-compiled only: PTX would be JIT-compiled by the driver at first use. A cuobjdump that cannot run or read
    # the file is a failure, not "no PTX" (H4).
    PTX_OUT="$(cuobjdump --list-ptx "${so}" 2>&1)" \
        || { echo "ERROR: cuobjdump --list-ptx failed on ${so##*/}; the PTX check cannot run: ${PTX_OUT}" >&2; exit 1; }
    PTX_FOUND="$(printf '%s\n' "${PTX_OUT}" | grep -oE 'sm_[0-9]+[a-z]?' | sort -u | tr '\n' ' ' || true)"
    if [[ -n "${PTX_FOUND// /}" ]]; then
        echo "ERROR: ${so##*/} embeds PTX (${PTX_FOUND}): the driver would JIT-compile it at runtime." >&2
        exit 1
    fi
done

# The env record: an allowlist of what shaped the binary, as name=value stamp fields. Never the whole environment
# (paths, tokens and host names would reach a public release). fa.arch is this repo's arch code, which for rtx40
# is sm_80 although the device is sm_89 (tuned/repo.env), so the stamp's own `arch` key alone would mislead.
ENV_RECORD=(
    "fa.arch=${GPU_TUNED_FA_ARCH}"
    "build.no_ptx=${FLASH_ATTN_NO_PTX}"
    "build.force_build=${FLASH_ATTENTION_FORCE_BUILD}"
    "build.tuned_common=$(cut -c1-12 tuned/common.sh.sha)"
)
for so in "${STAMP_LIBS[@]}"; do
    # fork.commit is the commit the wheel was built from, not HEAD at packaging (package-info.txt has both).
    gpu_tuned_stamp_lib_json "${so}" flash_attn "${VERSION}" "" \
        "upstream.declared_version=${BASE_VERSION}" "fork.commit=${BUILD_COMMIT}" "fork.dirty:bool=false" \
        "${ENV_RECORD[@]}" \
        || { echo "ERROR: could not stamp ${so##*/}." >&2; exit 1; }
done
echo "OK: ${#STAMP_LIBS[@]} extension stamped and read back (release, variant, cuda.tag)"

python3 -m wheel pack "${UNPACKED}" --dest-dir "${GPU_TUNED_DIST_DIR}"
WHEEL="${GPU_TUNED_DIST_DIR}/${BUILT_NAME}"
[[ -f "${WHEEL}" ]] || { echo "ERROR: repacked wheel is not ${BUILT_NAME}." >&2; exit 1; }
gpu_tuned_check_wheel_label "${WHEEL}"
gpu_tuned_write_sha256_sidecar "${WHEEL}"
WHEEL_SHA="$(awk '{print $1}' "${WHEEL}.sha256")"
echo "Wheel:  ${BUILT_NAME} ($(du -sh "${WHEEL}" | awk '{print $1}'))"
echo "SHA256: ${WHEEL_SHA}"

# H5: a DRAFT notes file if none is tracked yet; release.sh refuses it until a person has reviewed it and removed
# the banner. The numbers come from the attached test log (G4).
NOTES="tuned/releases/RELEASE_NOTES_${RELEASE_TAG#v}.md"
if [[ ! -f "${NOTES}" ]]; then
    mkdir -p tuned/releases
    {
        echo "<!-- DRAFT — not yet published: review every line, then delete this banner and commit. -->"
        echo "# flash_attn ${BASE_VERSION} — ${GPU_TUNED_VARIANT} ${CUDA_TAG} (${RELEASE_TAG#v})"
        echo ""
        echo "Single-arch flash_attn wheel for ${GPU_TUNED_HW_LABEL} (FLASH_ATTN_CUDA_ARCHS=${GPU_TUNED_FA_ARCH}, no PTX), CUDA ${CUDA_VER}."
        echo ""
        echo "- Release label: ${LABEL} (UTC date of the tuned commit)"
        echo "- Built from commit ${BUILD_COMMIT}; tested at ${TESTED_COMMIT}"
        echo "- Upstream base: ${GPU_TUNED_UPSTREAM_REF} merge-base $(git merge-base HEAD "${GPU_TUNED_UPSTREAM_REF}" 2>/dev/null || echo unknown)"
        echo "- Wheel sha256: ${WHEEL_SHA}"
        echo ""
        echo "## Tests"
        echo ""
        echo "Scope (from the attached test log): $(sed -n 's/^scope: *//p' "${TEST_LOG}" | head -1)"
        echo "Skips by design: $(sed -n 's/^skips by design: *//p' "${TEST_LOG}" | head -1)"
        echo "Result: $(grep '^=== Full test suite summary' "${TEST_LOG}" | tail -1)"
        echo ""
        echo "## Install"
        echo ""
        echo '```'
        echo "pip install \"flash_attn @ https://github.com/ZBrad-LLC/flash-attention/releases/download/$(echo "${RELEASE_TAG}" | sed 's/+/%2B/g')/${BUILT_NAME}\""
        echo '```'
    } > "${NOTES}"
    echo "Wrote DRAFT release notes: ${NOTES}"
fi

# Written last.
cat > "${GPU_TUNED_DIST_DIR}/package-info.txt" <<INFO
release_tag=${RELEASE_TAG}
wheel=${BUILT_NAME}
version=${VERSION}
upstream_version=${BASE_VERSION}
label=${LABEL}
commit=$(git rev-parse HEAD)
built_commit=${BUILD_COMMIT}
tested_commit=${TESTED_COMMIT}
cuda_tag=${CUDA_TAG}
variant=${GPU_TUNED_VARIANT}
sha256=${WHEEL_SHA}
INFO

echo ""
echo "Packaged (not published): ${WHEEL} (${RELEASE_TAG})"
echo "Next: CUDA_TAG=${CUDA_TAG} bash tuned/smoke_test.sh ${GPU_TUNED_VARIANT}, review and commit ${NOTES}, then CUDA_TAG=${CUDA_TAG} bash tuned/release.sh ${GPU_TUNED_VARIANT}"
