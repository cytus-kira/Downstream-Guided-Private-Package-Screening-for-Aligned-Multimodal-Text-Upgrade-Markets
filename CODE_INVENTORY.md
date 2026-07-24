# Code inventory

## Locked execution path

1. `hm_tdsc_nonpackage_ablations.py` provides feature loading, downstream-pair modeling, market construction helpers, selection baselines, and evaluation primitives.
2. `paper/run_quick_downstream_collapse_experiment.py` builds the candidate market, offline supervision, exact/quartic KRR students, buyer-synchronized memberships, score-only selection, and optional downstream replay.
3. `paper/main_text_experiments/run_main_text_experiments.py` fixes datasets, hostile profiles, methods, budgets, seeds, and run manifests; sibling scripts handle ablation, ratio replay, package-size sweep, fidelity, and summary generation.
4. `encrypted_benchmarks/krr_ckks_seal/ckks_seal_bench.cpp` implements the SEAL arithmetic micro-benchmarks used by the CKKS tables.
5. `run_two_party_release.py` orchestrates the separate-process buyer/market flow implemented in `vtd/`; `vtd/build_release_report.py` checks the attack matrix and aggregates the saved results.

## Included source families

| Family | Current purpose | Entry point |
|---|---|---|
| RQ1 main selection | 5k/20k targeted hostile markets | `run_main_score_5k.ps1`, `run_main_score_20k.ps1` |
| RQ2 ablation | teacher/reference, task operator, KRR signal removals | `run_main_ablation_5k.ps1` |
| Downstream replay | purchase-ratio training and AUROC/F1/accuracy | `run_ratio_downstream_5k.ps1` |
| RQ3 packaging | buyer-synchronized package sizes 2/4/6/8 | `run_package_size_sweep.py` |
| RQ4 approximation/cost | quartic replay and SEAL full-flow timings | `build_quartic_fidelity_replay.py`, `run_ckks_student_fullflow_comm.ps1` |
| RQ5 release integrity | two-party CKKS, Groth16 relation, attack tests | `run_two_party_release.py`, `go test ./...` |

## Intentionally excluded

- Top-level `hm_*final.py`, `*_fixed.py`, `*_audited.py`, and similar files from the working directory are historical exploration branches and are not imported by the locked path.
- `feature_cache/`, generated market `.npz` files, model/checkpoint files, and multi-GB raw run directories are derived data, not source.
- `vtd/runs/**/private`, message transcripts, proving/setup artifacts, compiled binaries, CMake build trees, logs, PID files, IDE metadata, LaTeX auxiliaries, and older ZIP files are excluded.
- Paper prose and figures are outside this code-release scope; only experiment-facing documentation and evidence snapshots are retained.

The release does not assign a new license. Distribution rights remain those of the project owners and third-party dependencies keep their own licenses.
