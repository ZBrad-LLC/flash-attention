#!/bin/bash
# tuned/release.sh <variant> — run the release gates for the wheel tuned/package.sh made. Stage order:
#   build.sh -> full_test.sh -> package.sh -> smoke_test.sh -> (review and commit tuned/releases/RELEASE_NOTES_<version>.md) -> this -> publish.sh
# It never builds, tests, packages or publishes (docs/ARCHITECTURE.md B3, I5). If every gate passes it writes the
# ready-marker (gpu_tuned_release_ready, RELEASE_READY_<variant>); tuned/publish.sh <variant> publishes exactly what
# the gates saw. The gates are the common ones in tuned-common (gpu_tuned_release_gates), all reported together.
#
#   CUDA_TAG=cu134 bash tuned/release.sh gb10
#
# The release is named by the wheel's date label (v<version>, the version ending +<variant>.cu<cuda>.<YYYYMMDD>[.<seq>]),
# read from package-info.txt and never recomputed here: the notes commit moves HEAD and its date.
set -uo pipefail

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/release.sh <variant>}"

# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1
cd "${REPODIR}" || exit 1

INFO="${GPU_TUNED_DIST_DIR}/package-info.txt"
TEST_LOG="${GPU_TUNED_RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
SMOKE_LOG="${GPU_TUNED_RELEASES_DIR}/SMOKE_RESULTS_${GPU_TUNED_VARIANT}.log"
MARKER="${GPU_TUNED_RELEASES_DIR}/RELEASE_READY_${GPU_TUNED_VARIANT}"
gpu_tuned_release_marker_clear "${MARKER}"

if [[ ! -f "${INFO}" ]]; then
    echo "ERROR: ${INFO} not found -- run 'CUDA_TAG=${CUDA_TAG} bash tuned/package.sh ${GPU_TUNED_ARG_VARIANT}' at the commit being released first." >&2
    exit 1
fi
info() { sed -n "s/^$1=//p" "${INFO}"; }
RELEASE_TAG="$(info release_tag)"; WHEEL_NAME="$(info wheel)"; WHEEL="${GPU_TUNED_DIST_DIR}/${WHEEL_NAME}"
PACKAGED_COMMIT="$(info commit)"; BASE_VERSION="$(info upstream_version)"
# The compared build artifact is the unstamped wheel in wheel/, not the packaged copy in dist/: packaging writes
# dist/ on every run, so its mtime would move past the test log.
BUILT_WHEEL="${GPU_TUNED_WHEEL_DIR}/${WHEEL_NAME}"
RELEASE_NOTES_FILE="tuned/releases/RELEASE_NOTES_${RELEASE_TAG#v}.md"
RELEASE_TITLE="flash_attn ${BASE_VERSION} — ${GPU_TUNED_VARIANT} ${CUDA_TAG} (${RELEASE_TAG#v}) — ${GPU_TUNED_HW_LABEL} wheel"

echo "Release ${RELEASE_TAG} (flash_attn ${BASE_VERSION}, packaged at ${PACKAGED_COMMIT}):"

gpu_tuned_release_gates tag="${RELEASE_TAG}" variant="${GPU_TUNED_VARIANT}" cuda="${CUDA_TAG}" info="${INFO}" \
    built="${BUILT_WHEEL}" test_log="${TEST_LOG}" \
    test_pass='^=== Full test suite summary: gates \+ numeric check passed ===$' \
    tarball="${WHEEL}" smoke_log="${SMOKE_LOG}" notes="${RELEASE_NOTES_FILE}" packaged="${PACKAGED_COMMIT}" \
    || { echo "" >&2; echo "ERROR: a release gate failed; nothing was published (and no ready-marker was written)." >&2; exit 1; }

echo ""
gpu_tuned_release_ready "${MARKER}" "zbrad-llc/flash-attention" "${RELEASE_TAG}" "${RELEASE_TITLE}" \
    "${RELEASE_NOTES_FILE}" \
    "${WHEEL}#${WHEEL_NAME}" \
    "${WHEEL}.sha256#${WHEEL_NAME}.sha256" \
    "${TEST_LOG}#Test results (${GPU_TUNED_VARIANT})" \
    "${SMOKE_LOG}#Package smoke test results (${GPU_TUNED_VARIANT})" || exit 1

echo "Done."
