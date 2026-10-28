#!/bin/bash
# tuned/full_test.sh <variant> — test the wheel tuned/build.sh made, for ONE variant and ONE CUDA toolkit, and
# write a results log in that toolkit's tree (tuned/build/<cuda-tag>/releases/TEST_RESULTS_<variant>.log).
# It replaces tuned/verify.sh, whose numeric check now lives in tuned/numeric_check.py.
#
#   CUDA_TAG=cu134 bash tuned/full_test.sh gb10
#
# The unstamped wheel (tuned/build/<cuda-tag>/wheel/flash_attn-*.whl) is installed --no-deps into the build venv
# and every check runs from an empty directory, so flash_attn and its extension are the wheel's, not the source
# tree's. The gate is:
#   1. wheel label and build commit (release-label, build-commit), H6 source check;
#   2. flash_attn and flash_attn_2_cuda load from the installed wheel, the toolkit libraries resolve (G2), the
#      extension is single-arch (verify_arch) and carries no PTX (FLASH_ATTN_NO_PTX);
#   3. tuned/numeric_check.py: flash_attn_func against SDPA (plain, causal, GQA) on the device.
# Skips by design, named in the log header: upstream's pytest suite (tests/) is not run, and nothing multi-GPU
# (a tuned build is single-GPU). This narrower gate is the same shape as vllm's recorded G3 decision and needs
# its own attestation in compliance.conf.
#
# The log ends with "=== Full test suite summary: gates + numeric check passed ===" only when every step
# passed; release.sh greps for that exact line. The other outcome is "=== Full test suite summary: FAILED
# (see above) ===". tuned/package.sh and tuned/release.sh refuse a log without the pass line.
set -uo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/full_test.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/flash_attn-*.whl 2>/dev/null | head -1)"
[[ -n "${WHEEL}" ]] || { echo "ERROR: no wheel in ${GPU_TUNED_WHEEL_DIR}. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
BUILD_COMMIT="$(cat "${GPU_TUNED_BUILD_ROOT}/build-commit" 2>/dev/null || true)"
LABEL="$(cat "${GPU_TUNED_BUILD_ROOT}/release-label" 2>/dev/null || true)"
[[ -n "${BUILD_COMMIT}" && -n "${LABEL}" ]] || { echo "ERROR: ${GPU_TUNED_BUILD_ROOT}/build-commit or release-label missing: the last build did not finish. Re-run tuned/build.sh ${GPU_TUNED_VARIANT}." >&2; exit 1; }
[[ "$(basename "${WHEEL}")" == flash_attn-*"+${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL}-"* ]] \
    || { echo "ERROR: $(basename "${WHEEL}") does not carry label ${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL} from release-label." >&2; exit 1; }
[[ "${BUILD_COMMIT}" != *-dirty ]] || { echo "ERROR: the build was made from a dirty tree (${BUILD_COMMIT}); commit and rebuild." >&2; exit 1; }
git cat-file -e "${BUILD_COMMIT}^{commit}" 2>/dev/null || { echo "ERROR: build commit ${BUILD_COMMIT} is not in this checkout." >&2; exit 1; }
# H6: the wheel must be from the source at HEAD; only tuned/ may differ.
STALE_SRC="$(git diff --name-only "${BUILD_COMMIT}" HEAD -- . ':!tuned' | head -5)"
if [[ -n "${STALE_SRC}" ]]; then
    echo "ERROR: the wheel is from ${BUILD_COMMIT}, but source outside tuned/ changed since:" >&2
    echo "${STALE_SRC}" | sed 's/^/    /' >&2
    echo "  Rebuild (tuned/build.sh ${GPU_TUNED_VARIANT}) before testing." >&2
    exit 1
fi

PY="${GPU_TUNED_VENV_DIR}/bin/python"
[[ -x "${PY}" ]] || { echo "ERROR: ${PY} not found. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
mkdir -p "${GPU_TUNED_RELEASES_DIR}"
RESULTS_FILE="${GPU_TUNED_RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
rm -f "${RESULTS_FILE}"

# Tuned builds are single-GPU: the tests see exactly one device (GPU_TUNED_TEST_GPUS, default 0).
export CUDA_VISIBLE_DEVICES="${GPU_TUNED_TEST_GPUS:-0}"
TESTDIR="$(mktemp -d)"
trap 'rm -rf "${TESTDIR}"' EXIT

echo "installing $(basename "${WHEEL}") into the build venv (--no-deps)"
"${PY}" -m pip install --quiet --no-deps --force-reinstall "${WHEEL}" \
    || { echo "ERROR: pip could not install the wheel." >&2; exit 1; }

SITE_DIR="$(cd "${TESTDIR}" && "${PY}" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
{
    echo "flash_attn tuned test run -- variant=${GPU_TUNED_VARIANT} cuda=${CUDA_VER} (${CUDA_TAG})"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit:  $(git rev-parse HEAD)"
    echo "build:   ${BUILD_COMMIT} (label ${LABEL})"
    echo "wheel:   $(basename "${WHEEL}") sha256 $(sha256sum "${WHEEL}" | awk '{print $1}')"
    echo "torch:   $(cd "${TESTDIR}" && "${PY}" -I -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no device")' 2>&1 | tail -1)"
    echo "scope:   gates; flash_attn_2_cuda load, arch and PTX checks; tuned/numeric_check.py (flash_attn_func vs SDPA: plain, causal, GQA)"
    echo "devices: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (single-GPU build: multi-GPU tests are not part of its scope)"
    echo "skips by design: upstream's pytest suite (tests/) is not run; no multi-GPU tests"
    echo "gates: wheel installed --no-deps in the build venv; extension loads from it; toolkit libs resolve under ${CUDA_HOME}; single-arch ${GPU_TUNED_FA_ARCH}, no PTX; build commit ${BUILD_COMMIT} == HEAD outside tuned/"
    echo ""
} > "${RESULTS_FILE}"

# Gates. A function, so its status comes back through the tee pipeline's PIPESTATUS.
gate_checks() {
    local rc=0 so loaded_from ptx
    (cd "${TESTDIR}" && "${PY}" -I -c '
import flash_attn, flash_attn_2_cuda
print("flash_attn: " + flash_attn.__file__)
print("flash_attn_2_cuda: " + flash_attn_2_cuda.__file__)
') || rc=1
    so="$(cd "${TESTDIR}" && "${PY}" -I -c 'import torch, flash_attn_2_cuda as m; print(m.__file__)' 2>/dev/null | tail -1)"
    if [[ "${so}" == "${SITE_DIR}/"* ]]; then echo "  ok:   flash_attn_2_cuda is the wheel's: ${so}"
    else echo "  FAIL: flash_attn_2_cuda loads from '${so}', not from ${SITE_DIR}"; return 1; fi
    if gpu_tuned_verify_toolkit_libs "${so}"; then echo "  ok:   toolkit libraries resolve under ${CUDA_HOME}"
    else echo "  FAIL: toolkit libraries do not resolve under ${CUDA_HOME}"; rc=1; fi
    gpu_tuned_verify_arch "${so}" "${GPU_TUNED_FA_ARCH}" || rc=1
    # Pre-compiled only: PTX would be JIT-compiled by the driver at first use (FLASH_ATTN_NO_PTX). A cuobjdump that
    # cannot run is a failure, not "no PTX" (H4).
    if ptx="$(cuobjdump --list-ptx "${so}" 2>&1)"; then
        ptx="$(printf '%s\n' "${ptx}" | grep -oE 'sm_[0-9]+[a-z]?' | sort -u | tr '\n' ' ' || true)"
        if [[ -n "${ptx// /}" ]]; then echo "  FAIL: ${so##*/} embeds PTX (${ptx})"; rc=1; else echo "  ok:   no PTX in ${so##*/}"; fi
    else
        echo "  FAIL: cuobjdump --list-ptx failed; the PTX check cannot run: ${ptx}"; rc=1
    fi
    return "${rc}"
}
# logrun <command...>: run with output to the terminal and the log; returns the command's own status.
logrun() { "$@" 2>&1 | tee -a "${RESULTS_FILE}"; return "${PIPESTATUS[0]}"; }

STATUS=0
echo "=== gates ===" | tee -a "${RESULTS_FILE}"
logrun gate_checks || STATUS=1
echo "" | tee -a "${RESULTS_FILE}"
echo "=== tuned/numeric_check.py ===" | tee -a "${RESULTS_FILE}"
(cd "${TESTDIR}" && "${PY}" -I "${REPO_ROOT}/tuned/numeric_check.py") 2>&1 | tee -a "${RESULTS_FILE}"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { STATUS=1; echo "numeric_check.py FAILED" | tee -a "${RESULTS_FILE}"; }

echo "" >> "${RESULTS_FILE}"
echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RESULTS_FILE}"
if [[ "${STATUS}" -eq 0 ]]; then
    echo "=== Full test suite summary: gates + numeric check passed ===" | tee -a "${RESULTS_FILE}"
else
    echo "=== Full test suite summary: FAILED (see above) ===" | tee -a "${RESULTS_FILE}"
fi
echo "Results written to ${RESULTS_FILE}"
exit "${STATUS}"
