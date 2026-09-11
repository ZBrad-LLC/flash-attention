#!/bin/bash
# tuned/verify.sh <variant> — local regression check for a tuned flash_attn
# build: confirms the build is still single-arch, then runs the same
# numeric-correctness comparison (flash_attn_func vs
# torch.nn.functional.scaled_dot_product_attention) that was originally
# done once by hand. Not a CI replacement for upstream's own workflows --
# those are gated to hopper/FA4 (`flash_attn/cute/*.py`) or `main`/
# `ci-fix`, neither of which this fork's `tuned-builds` work ever touched,
# so disabling Actions on this fork (2026-09-08) didn't actually drop any
# coverage that applied to us. This exists because we never had a local
# check of our own before that point.
#
# Requires tuned/build.sh <variant> to have already succeeded (reuses that
# venv, does not build).
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

echo "=========================================="
echo "Verifying flash_attn build (${GPU_TUNED_HW_LABEL})"
echo "=========================================="

# Same .so lookup as wheel.sh -- build/ dir first, editable-install
# fallback second.
FA2_SO="$(find "${REPO_ROOT}/build" -maxdepth 4 -name 'flash_attn_2_cuda*.so' 2>/dev/null | head -1)"
if [[ -z "${FA2_SO}" ]]; then
    FA2_SO="$(find "${REPO_ROOT}" -maxdepth 2 -name 'flash_attn_2_cuda*.so' 2>/dev/null | head -1)"
fi
if [[ -z "${FA2_SO}" ]]; then
    echo "ERROR: flash_attn_2_cuda*.so not found anywhere under ${REPO_ROOT} -- run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2
    exit 1
fi
gpu_tuned_verify_arch "${FA2_SO}" "${GPU_TUNED_FA_ARCH}"

echo ""
echo "Running numeric-correctness check (flash_attn_func vs SDPA)..."
python3 - <<'PYEOF'
import sys

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

torch.manual_seed(0)

# Same shape this was originally spot-checked with by hand: batch=2,
# seqlen=256, nheads=8, headdim=64 (nheads_kv=2 for the GQA case).
BATCH, SEQLEN, NHEADS, HEADDIM = 2, 256, 8, 64
# Normal bf16 kernel-order noise is ~1e-3 in this shape (0.00049-0.00098
# observed originally); a real regression should blow well past this.
TOLERANCE = 5e-3

device = "cuda"
dtype = torch.bfloat16


def sdpa_ref(q, k, v, causal, nheads_kv):
    # flash_attn_func takes (B, S, H, D); SDPA wants (B, H, S, D), and
    # needs the KV heads repeated up to NHEADS for a GQA comparison.
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    if nheads_kv != NHEADS:
        rep = NHEADS // nheads_kv
        kt = kt.repeat_interleave(rep, dim=1)
        vt = vt.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
    return out.transpose(1, 2)


def run_case(name, causal, nheads_kv):
    q = torch.randn(BATCH, SEQLEN, NHEADS, HEADDIM, device=device, dtype=dtype)
    k = torch.randn(BATCH, SEQLEN, nheads_kv, HEADDIM, device=device, dtype=dtype)
    v = torch.randn(BATCH, SEQLEN, nheads_kv, HEADDIM, device=device, dtype=dtype)

    fa_out = flash_attn_func(q, k, v, causal=causal)
    ref_out = sdpa_ref(q, k, v, causal, nheads_kv)

    max_diff = (fa_out - ref_out).abs().max().item()
    status = "OK" if max_diff < TOLERANCE else "FAIL"
    print(f"  {name:10s} max_abs_diff={max_diff:.5f}  [{status}]")
    return max_diff < TOLERANCE


results = [
    run_case("plain", causal=False, nheads_kv=NHEADS),
    run_case("causal", causal=True, nheads_kv=NHEADS),
    run_case("gqa", causal=False, nheads_kv=2),
]

if not all(results):
    print(f"\nFAIL: one or more cases exceeded tolerance ({TOLERANCE}).", file=sys.stderr)
    sys.exit(1)

print(f"\nAll cases within tolerance ({TOLERANCE}).")
PYEOF

echo ""
echo "=========================================="
echo "Verify complete (${GPU_TUNED_HW_LABEL}): build is single-arch and numerically correct."
echo "=========================================="
