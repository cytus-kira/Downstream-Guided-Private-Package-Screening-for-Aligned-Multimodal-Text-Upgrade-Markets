# Main Text Experiments

This folder contains the clean entry points for the paper main-text score-only
targeted-market experiments.

## What This Runs

- Datasets: `hateful_memes`, `hatespeech`, `mscoco`
- Markets: `noise`, `coreset_far_wrong`, `typiclust_dense`, `kmeans_center`,
  `uncertainty_badge`, `cosine`, `all_average`
- Good ratio: `10%`
- Purchase budget: `50`
- Evaluation: score-only selected-good count
- Target-aware market construction:
  - `good_source = downstream_any`
  - `good_target_avoid_weight = 0.05`
- Main online methods:
  - raw online student: `ours_kernel_ridge_student`
  - packaged online student: `ours_sample_package_krr`

The packaged method performs buyer-synchronized pre-score sample packaging.
The buyer centers its own image/weak-text representation, computes the first
principal direction, stably orders rows by that projection, and defines fixed
package memberships before any student score is evaluated. The seller reuses
those exact row-index sets for the corresponding candidate rows; it does not
run a second PCA or independently rematch rows. Each seller package is
represented by mean candidate-side `phi`, and a separate package-level KRR
student is trained with the same buyer-defined grouping rule on calibration
data.

For privacy-oriented execution, the buyer releases only the authorized package
membership metadata required by the matching contract. The seller locally
forms summaries for those memberships, encrypts only the summaries, and submits
them for CKKS scoring. CKKS then places many package summaries in SIMD slots and
evaluates the compact linear scoring backend across packages in parallel.

The target-aware term prevents the target baseline's top-score region from
being accidentally labeled as good, while keeping the good label anchored to
the downstream operator.

Important constraint: online scoring methods do not call the downstream model
on market candidates.  The downstream model is only used offline to define the
market labels and to produce calibration supervision for the student.  Direct
operator variants such as `ours_downstream_direct` are teacher/reference
diagnostics, not deployable online methods.

The default student supervision is `--student-supervision top_quantile`: the
offline teacher marks the top `good_ratio` calibration rows as positives, and
KRR learns a score used later on unseen market candidates.

## Run Commands

Short main-text run, matching the current checked setting:

```powershell
Set-Location paper
.\main_text_experiments\run_main_score_5k.ps1 -Seeds "42" -Device "cuda" -PackageSize 2
```

Larger 20k market run:

```powershell
Set-Location paper
.\main_text_experiments\run_main_score_20k.ps1 -Seeds "42" -Device "cuda" -PackageSize 2
```

Multi-seed run:

```powershell
.\main_text_experiments\run_main_score_5k.ps1 -Seeds "42,43,44" -Device "cuda" -PackageSize 2
```

Buyer-synchronized package-size sensitivity (`2,4,6,8`) with a common
whole-package budget of `48`. The sensitivity run uses `4,992` market rows,
which is the nearest 5k-scale size divisible by all four package sizes:

```powershell
python .\main_text_experiments\run_package_size_sweep.py `
  --package-sizes "2,4,6,8" `
  --seeds "42,43,44" `
  --preset main_score_5k `
  --device cuda `
  --market-size 4992 `
  --purchase-total 48
```

This writes the sweep runs under
`runs\buyer_sync_package_size_sweep_4992`, summary CSVs under
`paper_ready\buyer_sync_package_size_sweep_4992`, and the paper figure to
`latex\fig\fig_package_size_sensitivity_buyer_sync.pdf`.

Dry-run command inspection:

```powershell
python .\main_text_experiments\run_main_text_experiments.py `
  --preset main_score_5k `
  --seeds "42" `
  --dry-run
```

## Outputs

For `run_main_score_5k.ps1`, outputs are written under:

```text
paper\main_text_experiments\runs\main_score_5k
```

Each seed has a separate directory:

```text
runs\main_score_5k\seed_42
```

The merged summary is written to:

```text
runs\main_score_5k\summary
```

Important summary files:

- `targeted_summary_mean.csv`
- `targeted_summary_per_seed.csv`
- `good_count_pivot_mean.csv`
- `method_good_count_aggregate.csv`
- `market_diagnostic_mean.csv`
- `ckks_summary_mean.csv`
- `ablation_summary_overall.csv`
- `downstream_metrics_mean_std.csv`
- `table_targeted_main.tex`
- `table_method_aggregate.tex`
- `table_ckks.tex`
- `table_ablation.tex`
- `summary.md`

## Ablation

The ablation run keeps the same markets and compares:

- teacher/reference operator: `ours_downstream_direct`
- teacher/reference components: `ours_influence_only`,
  `ours_loss_reduction_only`
- closed-form online task operator: `ours_task_operator`
- online kernel-ridge student: `ours_kernel_ridge_student`
- online student component ablations: `ours_krr_influence_only`,
  `ours_krr_loss_reduction_only`
- package versions of the same online/reference variants

```powershell
.\main_text_experiments\run_main_ablation_5k.ps1 -Seeds "42" -Device "cuda"
```

Outputs are under:

```text
runs\main_ablation_5k\summary
```

Use `ablation_summary_overall.csv`, `ablation_summary_by_market.csv`, and
`table_ablation.tex`.

## Ratio Purchase Downstream Training

This run disables score-only mode.  For each purchase ratio, selected rows are
added to the buyer training set and the downstream pair model is trained.

Default ratios for 5k markets are `0.0025,0.005,0.01,0.02`, corresponding to
approximately `12,25,50,100` purchased rows.

```powershell
.\main_text_experiments\run_ratio_downstream_5k.ps1 `
  -Seeds "42" `
  -Ratios "0.0025,0.005,0.01,0.02" `
  -Device "cuda"
```

Outputs are under:

```text
runs\ratio_downstream_5k\summary
```

Use `downstream_metrics_mean_std.csv` for AUROC, macro-F1, accuracy, selected
good count, and training time grouped by dataset, market, method, and purchase
ratio.

## CKKS

CKKS estimates are read from the existing SEAL summary files configured in
`run_quick_downstream_collapse_experiment.py`.  The main score runs attach CKKS
columns for:

- `ours_kernel_ridge_student`
- `ours_sample_package_krr`
- `ours_task_operator`
- `ours_sample_package_task_operator`
- package ablation variants

To refresh the student raw/package full-flow benchmark, including input
encryption time, encrypted compute time, decrypt/decode time, and communication
bytes:

```powershell
.\main_text_experiments\run_ckks_student_fullflow_comm.ps1 `
  -Rows "5000" `
  -Dims "64" `
  -Repeats 3 `
  -Warmups 1 `
  -PackageSize 2
```

The output is:

```text
main_text_experiments\runs\ckks_poly4_all_methods\ckks_seal_summary.csv
```

The SEAL run is a single-key arithmetic micro-benchmark. It measures the
combined systems effect of reducing `N` raw rows to `ceil(N/2)` package means
and placing many package objects in CKKS SIMD slots. It evaluates the actual
quartic KRR circuit, including the encrypted package-norm term.
`ThresholdParties` is release-policy metadata; distributed threshold-share
generation is not implemented by this benchmark.

The summarizer exports:

- `ckks_summary_per_seed.csv`
- `ckks_summary_mean.csv`
- `table_ckks.tex`

## Verifiable Threshold Release Prototype

The release-integrity experiment is a Level-B witness-assisted transcript
verifier for the `R_CKKS-VTD` ring relations. It is not a production
zero-knowledge proof implementation. It verifies ciphertext/session binding,
registered key-share consistency, bounded smudging noise, reconstruction
tolerance, and rejection of wrong-ciphertext, wrong-key, malformed-share, and
excessive-noise attacks.

```powershell
python `
  .\main_text_experiments\run_verifiable_threshold_release_prototype.py `
  --released-scores "1,32,128" `
  --threshold 3 `
  --participants 5 `
  --ring-degree 1024
```

Outputs are written to:

```text
main_text_experiments\runs\verifiable_threshold_release_prototype
```

Use `vtd_prototype_summary.csv`, `vtd_attack_results.csv`,
`vtd_test_report.json`, and `table_vtd_prototype.tex`.

## Degree-4 CKKS KRR and Baseline Timing

The deployed online KRR now evaluates the actual RBF landmark score rather than
the former linear timing proxy. Row representations are L2 normalized before
packaging. Arithmetic package means and package landmarks therefore have norm at
most one without post-package renormalization. The registered bandwidth
satisfies `sigma^2 >= 0.5`, so

```text
u = -||x - landmark||^2 / (2 sigma^2) in [-4, 0].
```

The CKKS path evaluates a degree-4 Chebyshev approximation of `exp(u)`. The
power schedule computes `u^2`, then `u^3` and `u^4` in parallel, giving two
ciphertext-ciphertext nonlinear levels after the linear log-kernel argument.
Landmarks, KRR weights, bandwidth, and polynomial coefficients are plaintext
buyer parameters. The five-node Chebyshev interpolation remainder gives the
analytic uniform bound `max |exp(u)-p4(u)| <= 1/60` on `[-4,0]`; the measured
grid error is about `3.664e-3`.

Run the complete 5,000-row comparison:

```powershell
.\main_text_experiments\run_ckks_student_fullflow_comm.ps1 `
  -Rows 5000 -Dims 64 -Repeats 3 -Warmups 1 `
  -PackageSize 2 -ThresholdParties 3 `
  -KrrLandmarks 1000 -CoresetReferences 800 `
  -KmeansCenters 20 -TypiclustNeighbors 8
```

This benchmark includes Random, Cosine, Uncertainty, CoreSet, BADGE,
KMeans-center, TypiClust, raw KRR, and package KRR. Nonlinear scalar functions
use degree-4 polynomial circuits; squared-distance rankings remain squared
because this is exactly order-equivalent and avoids an unnecessary square root.
CoreSet output is a per-round distance release, so the report builder scales it
to the ten acquisition rounds used by the 50-row, 5-row-per-round protocol.

Build the approximation, fidelity, and paper-ready timing tables:

```powershell
python `
  .\main_text_experiments\build_ckks_poly4_report.py `
  --krr-summary `
  .\main_text_experiments\runs\ckks_poly4_krr_final\ckks_seal_summary.csv `
  --selection-results-root `
  .\main_text_experiments\runs\poly4_krr_main5k
```

This command also derives `ckks_amortized_batch_throughput.csv` from the
existing full-flow measurements and writes the derived throughput as panel (b)
of `table_ckks_poly4_all_methods.tex`. It does not rerun the SEAL benchmark.
CoreSet's object-rate fields are reported as row-rounds because the implemented
protocol scans all 5,000 candidates in each of ten acquisition rounds.

Replay exact-exponential and quartic KRR on the saved 63 main cases, including
the selected-good fidelity check used by Table 6 and its corresponding figure:

```powershell
python `
  .\main_text_experiments\build_quartic_fidelity_replay.py `
  --run-root .\main_text_experiments\runs\poly4_krr_main5k `
  --output-dir .\main_text_experiments\runs\poly4_krr_main5k\fidelity_replay `
  --figure-dir .\latex\fig `
  --purchase-total 50 `
  --main-cases `
  --with-selected-good-delta
```

This is a deterministic plaintext replay from the saved market NPZ files. It
checks the same labels, budget, stable tie-breaking, and size-2 package
memberships, and verifies every quartic selected-good count against the deployed
main-experiment CSV before writing the table and figure.

## Paper-Ready Table Package

After the experiment summaries exist, rebuild the ordered paper tables with:

```powershell
python .\main_text_experiments\build_paper_experiment_package.py
```

The package is organized in the intended paper order:

- `01_*`: comparison with advanced methods and targeted-market collapse
- `02_*`: component ablations
- `03_*`: internal raw/package, CKKS full-flow, communication, and downstream
  purchase-ratio comparisons

By default, the output goes to:

```text
main_text_experiments\paper_ready\latest_online_krr_ckks_parts_20260605_105428
```

To rebuild the paper figures from the same summaries:

```powershell
python .\main_text_experiments\build_paper_experiment_figures.py
```

This writes the main comparison line chart, the per-scenario mean-with-standard-
deviation bar chart, the pure standard-deviation diagnostic bar chart, and a
supplementary standard-deviation heatmap to:

```text
latex\fig
```

## Re-summarize Only

If results already exist:

```powershell
python .\main_text_experiments\summarize_main_text_results.py `
  --run-root .\main_text_experiments\runs\main_score_5k
```
