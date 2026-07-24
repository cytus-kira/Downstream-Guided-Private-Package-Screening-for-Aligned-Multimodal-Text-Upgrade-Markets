# Experiment summary and evidence ledger

## Evidence boundary

The packaged CSV/JSON files are copied from completed runs in the active checkout and are grouped under `evidence/`. Main selection and ablation use seeds 42/43/44, three datasets (`hateful_memes`, `hatespeech`, `mscoco`), seven hostile profiles, a 5,000-row market, 10% useful rows, and a 50-row purchase budget unless stated otherwise.

| Claim area | Protocol and metric | Artifact | Status |
|---|---|---|---|
| RQ1 selection | 63 cases; selected useful rows out of 50 | `evidence/rq1_main/` | formal multi-seed score-only evidence |
| RQ2 components | same 63 cases and budget; full vs signal removals | `evidence/rq2_ablation/` | formal ablation evidence |
| RQ3 package size | sizes 2/4/6/8; 4,992 rows; whole-package budget 48 | `evidence/rq3_package_size/` | formal sensitivity evidence |
| RQ4 approximation/cost | identical-candidate quartic replay; SEAL full-flow micro-benchmark | `evidence/rq4_ckks/` | fidelity plus hardware-specific systems evidence |
| RQ5 integrity | production-parameter two-party runs at 1/32/128 released scores | `evidence/rq5_vtd/` | research-prototype evidence |
| Downstream replay | ratios 0.25%/0.5%/1%/2%; AUROC/F1/accuracy | `evidence/downstream/` | mixed/negative diagnostic evidence |

## RQ1: hostile-market selection

Across 63 cases, package KRR selected `42.30 ± 7.97` useful rows out of 50 and raw KRR selected `39.63 ± 11.04`. The strongest non-proposed aggregate baseline was KMeans at `24.59 ± 20.61`; Random obtained `5.11 ± 1.74`. Large baseline variances and per-profile zeroes remain visible in the per-case CSV, so the aggregate should not be read as uniform dominance on every market.

Online KRR methods do not call the downstream model on market candidates. Downstream outputs are used offline for market labels and calibration supervision. BADGE and uncertainty are explicitly marked as model-based baselines in the CSV.

## RQ2: components and supervision

The deployable package KRR is `42.30 ± 7.97`. Removing its influence component gives `37.60 ± 10.97`; removing its loss component gives `31.33 ± 14.77`. Raw KRR is `39.63 ± 11.04`, with corresponding removals at `33.92 ± 15.65` and `26.79 ± 16.44`.

Teacher-direct (`49.46 ± 2.31`) and teacher-package-direct (`49.46 ± 1.16`) are non-deployable references, not online competitors. The online closed-form task operators are weaker than learned KRR in this protocol.

## RQ3: package granularity

With a fixed 48-row whole-package budget on 4,992-row markets, selected-useful means fall as package size grows: size 2 = `40.05`, size 4 = `30.14`, size 6 = `23.08`, size 8 = `20.90`. This supports a granularity/cost trade-off; it does not support the claim that larger packages preserve acquisition quality.

## RQ4: quartic approximation and CKKS cost

The degree-4 Chebyshev approximation of `exp(u)` on `[-4,0]` has measured grid maximum absolute error `3.664e-3`. On 63 saved main cases, raw KRR has mean top-50 overlap `0.879` (minimum `0.66`) and package KRR `0.910` (minimum `0.76`) against exact exponential scoring.

On the recorded 5,000-row SEAL benchmark, full-flow time is about `175.68 s` for raw KRR and `107.01 s` for size-2 package KRR; recorded communication is `48.73 MiB` and `24.73 MiB`, respectively. These are single-machine, single-key arithmetic micro-benchmark measurements. `ThresholdParties` in this benchmark is metadata and does not turn the SEAL benchmark into distributed threshold decryption.

## RQ5: two-party verifiable release

The production-parameter runs use Lattigo multiparty CKKS (`logN=13`, `logQ=[45,32,32,32,32]`, scale `2^32`) and gnark Groth16 over BN254. Batch sizes 1/32/128 all accepted the honest release. All 24 injected invalid cases across eight attack types were rejected. The post-purchase audit accepted 161 honest packages and detected all 483 modified-row, feature, or membership cases.

Proof generation is about `15.3–15.8 s`, verification about `50.8–52.2 ms`, proof size 196 B, and the saved local two-process end-to-end path about `244–245 s`. That end-to-end number includes repeated loading of very large setup files and is not a WAN latency result.

The proof establishes the exact implemented coefficient-wise release and registered-share relations for the unsmudged case. It does not prove candidate authenticity before opening, buyer ciphertext formation, public transcript privacy, downstream utility, or settlement fairness; it is not a production security audit.

## Downstream replay: mixed result

The ratio run confirms that KRR acquires many more labeled-useful rows than Random (for example, at budget 50: roughly `39.43` versus `5.11`). However, averaged downstream AUROC does not improve: at budget 50, Random is about `0.653`, raw KRR `0.632`, and package KRR `0.630`. Therefore the current downstream replay is retained as a negative/mixed result and must not be used to claim downstream performance gains.
