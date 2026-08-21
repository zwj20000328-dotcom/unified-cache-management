# AGENTS.md

## Project Overview

UCM (Unified Cache Management) — Python + C++ library that persists and reuses LLM KV Cache to accelerate inference. Integrates with vLLM, SGLang, and MindIE. Python ≥ 3.10.

## Build

```bash
# Required: set PLATFORM before any build or install
export PLATFORM=cuda          # cuda | ascend | ascend-a3 | musa | maca

# Build wheel
python -m build --no-isolation --wheel

# Editable install (injects ucm_patch.pth into site-packages for import hooks)
pip install -e .
```

Key env vars for `setup.py`:
- `PLATFORM` — **required**, target platform (or build defaults to `simu` with sparse disabled)
- `ENABLE_SPARSE` — compile sparse attention modules (`TRUE`/`false`)
- `UCM_ENABLE_MINDIE` — enable MindIE integration (requires `UCM_CXX11_ABI=0|1`)
- `BUILD_UCM_ASU` — enable ASU transport module

C++ is built via CMake (C++17). Root `CMakeLists.txt` drives everything under `ucm/`.

## Lint & Format

```bash
pip install -r requirements-lint.txt
pre-commit install                    # optional: enable git pre-commit hooks
bash format.sh                        # run all checks (black, isort, codespell, actionlint)
bash format.sh ci                     # CI mode: includes manual-stage hooks
```

- Formatter: **black**
- Import sorting: **isort** (profile=black)
- Spell check: **codespell** (skips `ucm/csrc/**`, `.github/**`, `gsa_on_device/csrc/**`)
- Type checking: `mypy` (installed via `requirements-lint.txt`)

## Testing

Tests live in `test/` with their own `pytest.ini`, `conftest.py`, `config.yaml`, and `requirements.txt`.

```bash
cd test
pip install -r requirements.txt
pytest                                # run all tests in suites/
pytest --stage=0                      # unit tests only (0=Unit,1=Smoke,2=Regression,3=Release)
pytest --stage=1+                     # smoke and above
pytest --feature=xxx                  # filter by feature tag
pytest --platform=gpu                 # filter by platform
pytest suites/E2E/test_foo.py::TestClass::test_bar  # single test
```

- Test discovery: `test/suites/` only (`testpaths = suites`)
- GPU allocation: conftest auto-acquires GPU via `pynvml` + file locks; tests declare needs via `@pytest.mark.gpu_mem(MB)` and `@pytest.mark.gpu_count(N)`
- Test config: `test/config.yaml` (model paths, DB backends, report settings)
- Results: saved to `test/results/` as JSONL/CSV by default

## Architecture

```
ucm/
├── store/          # Storage backends: nfsstore, dram, mooncakestore, posix, ds3fs, pcstore, asu, fake, empty, delegator, compress, pipeline, cache
├── sparse/         # Sparse attention algorithms: gsa, esa, kvstar, rerope, blend, gsa_on_device
├── transport/      # KV transport layer: kv/ (ASU), p2p/ (HiXL protocol)
├── integration/    # Framework adapters: vllm/, sglang/, mindie/
├── shared/         # Shared C++ libs: pool (buffer mgmt), trans (platform transfer), metrics, infra, vendor
├── connector/      # Connector module (C++)
└── pd/             # Prefill-Decode disaggregation
```

- **vLLM integration entry**: `ucm/integration/vllm/ucm_connector.py`
- **Import hook mechanism**: `ucm_patch.pth` is injected into site-packages on editable install; it auto-installs monkey-patches for vLLM and MindIE at import time
- **Store factory**: `ucm/store/factory.py` and `ucm/store/factory_v1.py` instantiate backends
- **Sparse factory**: `ucm/sparse/factory.py` instantiates sparse algorithms
- Platform-specific C++ code is selected at build time via `RUNTIME_ENVIRONMENT` CMake variable (set from `PLATFORM`)

## Build Scripts (CI-oriented)

```bash
scripts/build_cuda.sh       # CUDA wheel build (sets PLATFORM=cuda, ENABLE_SPARSE=TRUE)
scripts/build_ascend.sh     # Ascend NPU build (sets PLATFORM=ascend, ENABLE_SPARSE=TRUE, BUILD_UCM_ASU=ON)
scripts/build_sglang.sh     # SGLang/CUDA build (ENABLE_SPARSE=false)
scripts/build_mindie.sh     # MindIE/Ascend build (UCM_ENABLE_MINDIE=1, UCM_CXX11_ABI=1)
```

These expect `$WORKSPACE` to point to a parent directory containing `unified-cache-management/`.

## Conventions

- Python formatting: black + isort (profile=black). Always run `pre-commit run --all-files` before committing.
- C++ standard: C++17, compiled with `-Wall -Werror -fPIC`.
- Version source of truth: `version.ini` (`VLLM_UC_VERSION=...`).
- Package name on PyPI: `uc-manager`.
