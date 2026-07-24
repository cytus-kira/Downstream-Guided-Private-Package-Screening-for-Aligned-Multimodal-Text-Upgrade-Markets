# Validation record

Validation is intentionally split by execution level. The following checks were
performed on 2026-07-20 from the assembled release directory.

| Check | Result | Scope |
|---|---|---|
| `python verify_release.py` | pass | required files, forbidden artifacts/paths, syntax compilation of 19 Python files |
| main 5k `--dry-run` | pass | expands the seed-42 experiment and summarizer commands without reading data |
| `python verify_release.py --with-go` | pass | `go test ./...`; `vtd/internal/vtd` tests pass |
| PowerShell parser check | pass | all packaged `.ps1` entry points parse without errors |
| evidence consistency check | pass | 567 RQ1 rows, 252 package-size rows, 9 CKKS methods, 24/24 attacks rejected, 2,016 downstream rows |
| CKKS KRR CMake build | pass | Ubuntu 20.04 WSL, GCC 9.4, CMake 3.27.3, Microsoft SEAL 4.1.2 |
| SEAL CPU CMake build | pass | all four targets: `seal_cpu_bench_1`, `sgf_ckks_ledger`, `packing_ablation`, `packing_ablation_main_fast` |
| CKKS numeric smoke | pass | 16 rows, dimension 4, Random/Cosine/package-KRR, `--validate`, exit code 0 |

The CUDA-linked `seal_gpu` variant was retained as source but was not built in
this packaging run. Its presence does not mean the Microsoft SEAL operations
are GPU-native.

A formal model/market rerun was not performed during packaging because the
multi-GB feature caches and generated markets are intentionally outside this
archive. The included formal evidence remains a saved checkout snapshot, not a
new run produced by the release audit.
