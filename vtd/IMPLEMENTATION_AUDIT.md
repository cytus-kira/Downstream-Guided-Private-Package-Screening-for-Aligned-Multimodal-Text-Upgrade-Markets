# Two-Party Verifiable Threshold Release Audit

## Implemented path

- Lattigo v6.2.0 multiparty CKKS with native two-party collective public-key and relinearization-key generation.
- Buyer and market commands execute as separate OS processes and persist only their own secret share.
- Real cached package summaries are encrypted and evaluated through the degree-4, 1,000-landmark RQ4 arithmetic shape.
- The market computes the exact output-level partial-decryption contribution.
- gnark v0.15.0 Groth16 over BN254 proves the exact coefficient-wise NTT/RNS release relation and registered market CKG-share relation.
- The buyer rebuilds the public witness from the exact ciphertext, contribution, signed policy, registration, and public key material; it never reads the market witness.

## Production-parameter evidence

Parameters: `logN=13`, `logQ=[45,32,32,32,32]`, scale `2^32`, output level `0`, and 1,000 quartic scorer landmark terms.

| Scores | Seed | Prove (s) | Verify (ms) | Proof (B) | Response (KiB) | Local two-process E2E (s) | Max error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 15.32 | 52.2 | 196 | 129.5 | 243.7 | 2.49e-5 |
| 32 | 43 | 15.76 | 52.2 | 196 | 129.5 | 245.1 | 9.57e-5 |
| 128 | 44 | 15.53 | 50.8 | 196 | 129.5 | 244.9 | 1.42e-4 |

The circuit has 3,182,950 constraints and 32,771 public variables. One-time Groth16 setup took 177.8 s and produced a 438 MB R1CS and 621 MB proving key. The reported E2E path includes repeated loading of these files by short-lived processes and is not WAN latency.

All three honest releases were accepted. All 24 injected wrong-target, wrong-share, malformed-contribution, large-contribution-tampering, wrong-level, proof-tampering, replay, and unauthorized-output cases were rejected. The post-purchase audit accepted 161 honest packages and detected all 483 injected row, feature, or membership changes.

## Reproduction

The first production run creates reusable Groth16 setup artifacts:

```powershell
python run_two_party_release.py --scores 1 --seed 42 --preset production --landmarks 1000 --repeats 1 --run-root vtd\runs\exact_relation_production_b1_s42
```

Compatible runs can reuse that setup:

```powershell
python run_two_party_release.py --scores 32 --seed 43 --preset production --landmarks 1000 --repeats 1 --reuse-proof-from vtd\runs\exact_relation_production_b1_s42 --run-root vtd\runs\exact_relation_production_b32_s43
python run_two_party_release.py --scores 128 --seed 44 --preset production --landmarks 1000 --repeats 1 --reuse-proof-from vtd\runs\exact_relation_production_b1_s42 --run-root vtd\runs\exact_relation_production_b128_s44
```

Regenerate aggregate CSVs, LaTeX rows, and the figure with:

```powershell
python vtd\build_release_report.py --runs vtd\runs\exact_relation_production_b1_s42 vtd\runs\exact_relation_production_b32_s43 vtd\runs\exact_relation_production_b128_s44
```

## Claim boundary

This is a general-purpose Groth16 proof backend, not a lattice-native proof. The implemented circuit proves the exact unsmudged subrelation (`eta=0`) and does not implement the general paper relation's coefficient range bounds. It does not prove how the buyer produced the ciphertext, public-contribution transcript privacy, package authenticity before opening, downstream utility, or settlement fairness.
