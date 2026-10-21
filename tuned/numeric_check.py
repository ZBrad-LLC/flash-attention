"""Numeric-correctness check of the installed flash_attn.

Compares flash_attn_func with torch.nn.functional.scaled_dot_product_attention for
the plain, causal and GQA cases. Run by tuned/full_test.sh (the unstamped wheel) and
tuned/smoke_test.sh (the packaged wheel), from an empty directory with ``python -I``,
so the flash_attn it imports is the installed one. Exits 1 if any case is out of
tolerance. This was tuned/verify.sh's inline check, moved here unchanged in shape.
"""

import sys

from flash_attn import flash_attn_func
from torch import Tensor, bfloat16, manual_seed, randn
from torch.nn.functional import scaled_dot_product_attention

# Batch=2, seqlen=256, nheads=8, headdim=64 (nheads_kv=2 for the GQA case): the shape
# this was originally spot-checked with by hand.
BATCH = 2
SEQLEN = 256
NHEADS = 8
HEADDIM = 64
# Normal bf16 kernel-order noise is ~1e-3 in this shape (0.00049-0.00098 observed
# originally); a real regression should blow well past this.
TOLERANCE = 5e-3
DEVICE = "cuda"


class NumericCheck:
    """Run flash_attn_func against SDPA and report the largest absolute difference."""

    @staticmethod
    def _sdpa_ref(
        q: Tensor, k: Tensor, v: Tensor, causal: bool, nheads_kv: int
    ) -> Tensor:
        """Reference attention: SDPA with the KV heads repeated up to NHEADS.

        flash_attn_func takes (B, S, H, D); SDPA wants (B, H, S, D).
        """
        qt = q.transpose(1, 2)
        kt = k.transpose(1, 2)
        vt = v.transpose(1, 2)
        if nheads_kv != NHEADS:
            rep = NHEADS // nheads_kv
            kt = kt.repeat_interleave(rep, dim=1)
            vt = vt.repeat_interleave(rep, dim=1)
        out = scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
        return out.transpose(1, 2)

    @staticmethod
    def run_case(name: str, causal: bool, nheads_kv: int) -> bool:
        """Compare one case.

        Args:
            name: label printed with the result.
            causal: whether to apply a causal mask.
            nheads_kv: number of KV heads (NHEADS for plain, fewer for GQA).

        Returns:
            True if the largest absolute difference is within TOLERANCE.
        """
        q = randn(BATCH, SEQLEN, NHEADS, HEADDIM, device=DEVICE, dtype=bfloat16)
        k = randn(BATCH, SEQLEN, nheads_kv, HEADDIM, device=DEVICE, dtype=bfloat16)
        v = randn(BATCH, SEQLEN, nheads_kv, HEADDIM, device=DEVICE, dtype=bfloat16)
        fa_out = flash_attn_func(q, k, v, causal=causal)
        ref_out = NumericCheck._sdpa_ref(q, k, v, causal, nheads_kv)
        max_diff = (fa_out - ref_out).abs().max().item()
        ok = max_diff < TOLERANCE
        print(f"  {name:10s} max_abs_diff={max_diff:.5f}  [{'OK' if ok else 'FAIL'}]")
        return ok

    @classmethod
    def main(cls) -> int:
        """Run all three cases; return the process exit status."""
        manual_seed(0)
        results = [
            cls.run_case("plain", causal=False, nheads_kv=NHEADS),
            cls.run_case("causal", causal=True, nheads_kv=NHEADS),
            cls.run_case("gqa", causal=False, nheads_kv=2),
        ]
        if not all(results):
            print(
                f"\nFAIL: one or more cases exceeded tolerance ({TOLERANCE}).",
                file=sys.stderr,
            )
            return 1
        print(f"\nAll cases within tolerance ({TOLERANCE}).")
        return 0


if __name__ == "__main__":
    sys.exit(NumericCheck.main())
