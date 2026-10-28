# tuned/ — single-arch flash_attn wheels

Builds the standalone `flash_attn` wheel (Dao-AILab's FA2 package, root `setup.py`) for one GPU variant and one CUDA
toolkit. This is not the vLLM fork's `vllm_flash_attn` (see zbrad/flash-attention-vllm), and `hopper/` (FA3, sm_90) is
out of scope. The layout and rules are `docs/ARCHITECTURE.md` in
[zbrad/tuned-common](https://github.com/zbrad/tuned-common). Only gb10 is built from this repo today; rtx40 and rtx50 have
settings in `repo.env`.

```
CUDA_TAG=cu134 bash tuned/build.sh gb10        # build venv + unstamped wheel   -> tuned/build/cu134/wheel/
CUDA_TAG=cu134 bash tuned/full_test.sh gb10    # tests + numeric check          -> .../releases/TEST_RESULTS_gb10.log
CUDA_TAG=cu134 bash tuned/package.sh gb10      # stamp, sidecars, DRAFT notes   -> .../dist/
CUDA_TAG=cu134 bash tuned/smoke_test.sh gb10   # the packaged bits run          -> .../releases/SMOKE_RESULTS_gb10.log
# review and commit tuned/releases/RELEASE_NOTES_<version>.md
CUDA_TAG=cu134 bash tuned/release.sh gb10      # gates only; writes the ready-marker, never publishes
CUDA_TAG=cu134 bash tuned/publish.sh gb10      # publishes exactly what release.sh saw
```

| File | Purpose |
|---|---|
| `env.sh` | device, toolkit and tree setup; sourced by every script |
| `repo.env` | flash-attention-only settings: upstream ref, the arch spelling `setup.py` takes (`GPU_TUNED_FA_ARCH`, `121a` for gb10) |
| `common.sh`, `sync-common.sh`, `common-data/` | vendored from tuned-common; refresh with `bash tuned/sync-common.sh sync` |
| `build.sh`, `wheel.sh`, `full_test.sh`, `package.sh`, `smoke_test.sh`, `release.sh`, `publish.sh` | the stages above (`build.sh` prepares the venv, then `wheel.sh` compiles) |
| `numeric_check.py` | numeric comparison used by the test stage |
| `compliance.conf` | attestations for the checks `tools/tuned_compliance.sh` cannot make mechanically |

## Build venv and gates

- The build venv is `.venv-<variant>-base` at the repo root (`GPU_TUNED_VENV_DIR`), made by `gpu_tuned_setup_build_venv`.
  `GPU_TUNED_CLEAN_VENV=1` removes and recreates it. It installs the current setuptools (at least 84.2) last (C8).
- flash_attn compiles against an existing torch and its `setup.py` leaves `torch` unpinned, so a tuned torch wheel
  (zbrad/pytorch) must already be installed in the venv; `build.sh` stops if none is.
- `package.sh` fails if the venv's setuptools is below the minimum (H10) and runs the dependency audit on the wheel (H9),
  logged to `tuned/build/<cuda-tag>/dependency-audit.log`.
- Everything lives under `tuned/build/<cuda-tag>/`, which is gitignored.
