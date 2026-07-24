# CKKS SEAL Benchmark

This directory contains the Microsoft SEAL implementation used to benchmark the encrypted middle computation of the paper scheme.

The default comparison set is:

- `baseline_random_noop`: random purchase baseline with no encrypted middle-layer scoring.
- `baseline_cosine_ctpt`: one CKKS ciphertext-plaintext cosine score.
- `baseline_coreset_distance_ctpt`: CoreSet/coverage-style single-center encrypted distance proxy with ciphertext-plaintext products. Center selection and min-distance/top-k comparisons are outside the encrypted evaluator timing.
- `baseline_linear_student_ctpt`: a linear student-style encrypted score.
- `ours_dcc_row_simd_ctpt`: current Downstream-Calibrated Coverage scorer at row level. Each CKKS slot stores one candidate row; encrypted coverage and task-summary coordinates are combined with plaintext buyer-calibrated coefficients.
- `ours_dcc_pkg_simd_ctpt`: the same DCC scorer after seller-side package summarization. Each CKKS slot stores one package summary, so `logical_rows` raw rows are scored as `ceil(logical_rows / package_size)` encrypted package objects.
- `ours_student_pkg_ctpt`: latest deployed downstream-guided package student. Seller package summaries are encrypted, student coefficients are plaintext constants, and the scorer is a residual linear function over package-summary features. `logical_rows` is still the raw market size; the benchmark scores `ceil(logical_rows / package_size)` encrypted package summaries.
- `ours_student_row_ctpt`: raw-row reference for the same linear student idea, included to show the cost of avoiding package compression.
- `ours_structural_ctpt`: the paper structural triad score with ciphertext-plaintext products.
- `ours_packaged_structural_ctpt`: `ours_structural_ctpt` plus encrypted package aggregation.
- `ours_structural_ctct`: the same structural triad with ciphertext-ciphertext products.
- `ours_packaged_structural_ctct`: `ours_structural_ctct` plus encrypted package aggregation.
- `baseline_poly2_fusion_ctct`: a deeper degree-2 encrypted fusion baseline.

By default, the benchmark reports the encrypted evaluator region used in the middle score computation and also records the one-time input preparation/encryption time. Add `--measure-decrypt-all` through the PowerShell wrapper to decrypt and decode every released score ciphertext and report a full-flow micro-benchmark split into input preparation/encryption, encrypted evaluation, and authorized output decrypt/decode. The `--threshold-parties` option records the intended release policy size as metadata; this SEAL benchmark still uses the single-key CKKS API and does not implement distributed threshold-share generation.

For the latest DCC variants, packaging is assumed to have already happened locally at the seller when `ours_dcc_pkg_simd_ctpt` is used. The online encrypted path evaluates only a low-depth CT-PT linear score over encrypted SIMD slots. Numeric DCC coefficients are plaintext constants learned in the buyer/task calibration stage; the benchmark cost is independent of their particular values.

## Run from PowerShell

```powershell
Set-Location encrypted_benchmarks\krr_ckks_seal
.\run_ckks_seal_bench_wsl.ps1 `
  -SealDir C:\path\to\SEAL\build\cmake `
  -OutputDir .\results `
  -Rows "100,500,1000,5000,10000" `
  -Dims "16,32,64" `
  -Schemes "baseline_random_noop,baseline_cosine_ctpt,baseline_coreset_distance_ctpt,ours_dcc_row_simd_ctpt,ours_dcc_pkg_simd_ctpt" `
  -Repeats 5 `
  -Warmups 1 `
  -PackageSize 4 `
  -StudentSummaryDim 10 `
  -MeasureDecryptAll `
  -ThresholdParties 3
```

For a quick smoke test:

```powershell
.\run_ckks_seal_bench_wsl.ps1 -Rows "16" -Dims "4" -Repeats 1 -Schemes "baseline_random_noop,baseline_cosine_ctpt,baseline_coreset_distance_ctpt,ours_dcc_row_simd_ctpt,ours_dcc_pkg_simd_ctpt" -PackageSize 4 -StudentSummaryDim 10 -Validate
```

Outputs:

- `ckks_seal_results.csv`: per-repeat raw timings, including `input_prepare_encrypt_ms`, `encrypted_compute_ms`, `decrypt_decode_ms`, and `total_full_flow_ms` when decrypt/decode measurement is enabled.
- `ckks_seal_summary.csv`: mean/std summary grouped by scheme, row count, feature dimension, and threshold-release metadata. It also reports `input_ciphertexts`, `input_ciphertext_bytes`, `output_ciphertext_bytes`, and `total_communication_bytes`.
- `ckks_seal_summary.md`: compact markdown table for checking the trend.
