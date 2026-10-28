# flash_attn 2.8.4 — gb10 cu134 (2.8.4+gb10.cu134.20261028.0)

Single-arch flash_attn wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FLASH_ATTN_CUDA_ARCHS=121a, no PTX), CUDA 13.4.

- Release label: 20261028.0 (UTC date of the tuned commit)
- Built from commit 667710583fe31fb4abc52f1e5b85aecd910b6209; tested at 667710583fe31fb4abc52f1e5b85aecd910b6209
- Upstream base: upstream/main merge-base 1fc0221183e799363001394a12dd9e78e15f5dda
- Wheel sha256: 7cbc7b269b4ae9954428024f0095ecffff5bec30f2097738024e7432702c76f5

## Changes since the previous release

- Built and tested against torch 2.15.0+gb10.cu134.20261022.0 (no CPU BLAS, LAPACK or MPI), installed into the build venv by hand: this repo does not pin a torch (the wheel's `install_requires` says only `torch`), so the build is only reproducible if that same wheel is installed first.
- The build venv has setuptools 84.2 or newer, and package.sh checks that and audits the wheel's runtime dependencies.
- No kernel or source changes from this fork's tuned code.

## Tests

Scope (from the attached test log): gates; flash_attn_2_cuda load, arch and PTX checks; tuned/numeric_check.py (flash_attn_func vs SDPA: plain, causal, GQA)
Skips by design: upstream's pytest suite (tests/) is not run; no multi-GPU tests
Result: === Full test suite summary: gates + numeric check passed ===

## Install

```
pip install "flash_attn @ https://github.com/ZBrad-LLC/flash-attention/releases/download/v2.8.4%2Bgb10.cu134.20261028.0/flash_attn-2.8.4+gb10.cu134.20261028.0-cp314-cp314-linux_aarch64.whl"
```
