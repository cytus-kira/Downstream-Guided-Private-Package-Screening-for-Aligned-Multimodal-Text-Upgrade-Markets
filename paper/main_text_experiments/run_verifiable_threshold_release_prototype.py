#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Level-B prototype for verifiable threshold CKKS score release.

This is a witness-assisted transcript verifier, not a production NIZK system.
It exercises the ring relations intended for R_CKKS-VTD, binds the transcript
to the authorized ciphertext and release metadata, and measures release-layer
overhead plus attack rejection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


Q = 2_147_483_647


@dataclass(frozen=True)
class Registration:
    participant_id: int
    pk_share: np.ndarray
    commitment: str


@dataclass(frozen=True)
class Witness:
    secret_share: np.ndarray
    opening_randomness: bytes
    key_error: np.ndarray
    smudging_noise: np.ndarray


@dataclass(frozen=True)
class Ciphertext:
    c0: np.ndarray
    c1: np.ndarray
    level: int
    scale: int


@dataclass(frozen=True)
class Transcript:
    participant_id: int
    output_id: int
    delta: np.ndarray
    statement_digest: str
    fiat_shamir_challenge: str
    serialized_size_bytes: int


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def center_mod(x: np.ndarray, modulus: int = Q) -> np.ndarray:
    y = np.mod(np.asarray(x, dtype=np.int64), modulus)
    half = modulus // 2
    return np.where(y > half, y - modulus, y).astype(np.int64)


def ring_add(*polys: np.ndarray, modulus: int = Q) -> np.ndarray:
    total = np.zeros_like(np.asarray(polys[0], dtype=np.int64))
    for poly in polys:
        total = np.mod(total + np.asarray(poly, dtype=np.int64), modulus)
    return total.astype(np.int64)


def ring_mul(a: np.ndarray, b: np.ndarray, modulus: int = Q) -> np.ndarray:
    """Negacyclic multiplication in Z_q[X]/(X^N+1)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    n = len(a)
    conv = np.convolve(a, b).astype(np.int64)
    out = conv[:n].copy()
    tail = conv[n:]
    out[: len(tail)] -= tail
    return np.mod(out, modulus).astype(np.int64)


def encode_array(x: np.ndarray) -> bytes:
    return np.asarray(x, dtype="<i8").tobytes(order="C")


def hash_bytes(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)
    return h.hexdigest()


def canonical_json_bytes(value: Dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def commitment(secret_share: np.ndarray, randomness: bytes) -> str:
    return hash_bytes(b"CKKS-VTD-COMMIT", encode_array(secret_share), randomness)


def ciphertext_hash(ct: Ciphertext) -> str:
    meta = canonical_json_bytes({"level": ct.level, "scale": ct.scale})
    return hash_bytes(b"CKKS-VTD-CT", meta, encode_array(ct.c0), encode_array(ct.c1))


def statement_fields(
    sid: str,
    output_id: int,
    ct: Ciphertext,
    output_policy_hash: str,
    registration: Registration,
    lagrange: int,
    delta: np.ndarray,
) -> Dict[str, object]:
    return {
        "sid": sid,
        "output_id": int(output_id),
        "ciphertext_hash": ciphertext_hash(ct),
        "level": int(ct.level),
        "scale": int(ct.scale),
        "output_policy_hash": output_policy_hash,
        "participant_id": int(registration.participant_id),
        "lagrange": int(lagrange),
        "pk_share_hash": hash_bytes(encode_array(registration.pk_share)),
        "secret_share_commitment": registration.commitment,
        "delta_hash": hash_bytes(encode_array(delta)),
    }


def make_transcript(
    sid: str,
    output_id: int,
    ct: Ciphertext,
    output_policy_hash: str,
    registration: Registration,
    lagrange: int,
    delta: np.ndarray,
) -> Transcript:
    fields = statement_fields(
        sid,
        output_id,
        ct,
        output_policy_hash,
        registration,
        lagrange,
        delta,
    )
    statement_blob = canonical_json_bytes(fields)
    statement_digest = hash_bytes(b"CKKS-VTD-STATEMENT", statement_blob)
    challenge = hash_bytes(
        b"CKKS-VTD-FIAT-SHAMIR",
        statement_blob,
        bytes.fromhex(statement_digest),
    )
    public_payload_size = (
        len(statement_blob)
        + len(encode_array(delta))
        + len(bytes.fromhex(statement_digest))
        + len(bytes.fromhex(challenge))
    )
    return Transcript(
        participant_id=registration.participant_id,
        output_id=int(output_id),
        delta=np.asarray(delta, dtype=np.int64),
        statement_digest=statement_digest,
        fiat_shamir_challenge=challenge,
        serialized_size_bytes=int(public_payload_size),
    )


def verify_transcript_with_test_witness(
    transcript: Transcript,
    witness: Witness,
    sid: str,
    output_id: int,
    ct: Ciphertext,
    output_policy_hash: str,
    registration: Registration,
    common_a: np.ndarray,
    lagrange: int,
    bounds: Dict[str, int],
) -> bool:
    """Verify R_CKKS-VTD using a test witness, not a production NIZK."""
    if int(output_id) != int(transcript.output_id):
        return False
    fields = statement_fields(
        sid,
        output_id,
        ct,
        output_policy_hash,
        registration,
        lagrange,
        transcript.delta,
    )
    statement_blob = canonical_json_bytes(fields)
    expected_statement = hash_bytes(b"CKKS-VTD-STATEMENT", statement_blob)
    expected_challenge = hash_bytes(
        b"CKKS-VTD-FIAT-SHAMIR",
        statement_blob,
        bytes.fromhex(expected_statement),
    )
    if transcript.statement_digest != expected_statement:
        return False
    if transcript.fiat_shamir_challenge != expected_challenge:
        return False
    if commitment(witness.secret_share, witness.opening_randomness) != registration.commitment:
        return False
    if int(np.max(np.abs(witness.secret_share))) > int(bounds["secret"]):
        return False
    if int(np.max(np.abs(witness.key_error))) > int(bounds["key_error"]):
        return False
    if int(np.max(np.abs(witness.smudging_noise))) > int(bounds["smudge"]):
        return False

    expected_pk = ring_add(
        -ring_mul(common_a, witness.secret_share),
        witness.key_error,
    )
    if not np.array_equal(expected_pk, registration.pk_share):
        return False
    expected_delta = ring_add(
        int(lagrange) * ring_mul(ct.c1, witness.secret_share),
        witness.smudging_noise,
    )
    return bool(np.array_equal(expected_delta, transcript.delta))


def sample_small(
    rng: np.random.Generator,
    n: int,
    bound: int,
) -> np.ndarray:
    return rng.integers(-bound, bound + 1, size=n, dtype=np.int64)


def register_participants(
    rng: np.random.Generator,
    participant_count: int,
    n: int,
    common_a: np.ndarray,
    bounds: Dict[str, int],
) -> Tuple[List[Registration], List[Witness]]:
    registrations: List[Registration] = []
    witnesses: List[Witness] = []
    for participant_id in range(participant_count):
        secret = sample_small(rng, n, bounds["secret"])
        key_error = sample_small(rng, n, bounds["key_error"])
        randomness = rng.bytes(32)
        pk_share = ring_add(-ring_mul(common_a, secret), key_error)
        registrations.append(
            Registration(
                participant_id=participant_id,
                pk_share=pk_share,
                commitment=commitment(secret, randomness),
            )
        )
        witnesses.append(
            Witness(
                secret_share=secret,
                opening_randomness=randomness,
                key_error=key_error,
                smudging_noise=np.zeros(n, dtype=np.int64),
            )
        )
    return registrations, witnesses


def encrypt_score_for_joint_key(
    rng: np.random.Generator,
    score: float,
    joint_secret: np.ndarray,
    n: int,
    scale: int,
    level: int,
    ckks_error_bound: int,
) -> Ciphertext:
    c1 = rng.integers(0, Q, size=n, dtype=np.int64)
    message = np.zeros(n, dtype=np.int64)
    message[0] = int(round(float(score) * scale))
    ckks_error = sample_small(rng, n, ckks_error_bound)
    c0 = ring_add(message, ckks_error, -ring_mul(c1, joint_secret))
    return Ciphertext(c0=c0, c1=c1, level=int(level), scale=int(scale))


def partial_decrypt(
    ct: Ciphertext,
    witness: Witness,
    lagrange: int,
) -> np.ndarray:
    return ring_add(
        int(lagrange) * ring_mul(ct.c1, witness.secret_share),
        witness.smudging_noise,
    )


def reconstruct(ct: Ciphertext, transcripts: Iterable[Transcript]) -> np.ndarray:
    parts = [ct.c0] + [t.delta for t in transcripts]
    return ring_add(*parts)


def decode_constant(poly: np.ndarray, scale: int) -> float:
    return float(center_mod(poly)[0]) / float(scale)


def replace_witness_noise(witness: Witness, noise: np.ndarray) -> Witness:
    return Witness(
        secret_share=witness.secret_share,
        opening_randomness=witness.opening_randomness,
        key_error=witness.key_error,
        smudging_noise=np.asarray(noise, dtype=np.int64),
    )


def time_ms(fn):
    start = time.perf_counter_ns()
    result = fn()
    return result, float((time.perf_counter_ns() - start) / 1e6)


def run_configuration(
    score_count: int,
    cli: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    n = int(cli.ring_degree)
    threshold = int(cli.threshold)
    participant_count = int(cli.participants)
    bounds = {
        "secret": int(cli.secret_bound),
        "key_error": int(cli.key_error_bound),
        "smudge": int(cli.smudge_bound),
    }
    common_a = rng.integers(0, Q, size=n, dtype=np.int64)
    registrations, base_witnesses = register_participants(
        rng,
        participant_count,
        n,
        common_a,
        bounds,
    )
    selected = list(range(threshold))
    # Keep the joint key in its centered small-coefficient representation.
    # Modular reduction happens inside ring operations; using q-1 for -1 here
    # would unnecessarily inflate convolution products and risk int64 overflow.
    joint_secret = np.sum(
        np.stack([base_witnesses[i].secret_share for i in selected], axis=0),
        axis=0,
        dtype=np.int64,
    ).astype(np.int64)
    output_policy = {
        "sid": cli.sid,
        "authorized_outputs": list(range(score_count)),
        "threshold": threshold,
        "participants": participant_count,
    }
    output_policy_hash = hash_bytes(canonical_json_bytes(output_policy))

    partial_times: List[float] = []
    transcript_times: List[float] = []
    hash_times: List[float] = []
    verify_times: List[float] = []
    reconstruction_times: List[float] = []
    release_errors: List[float] = []
    transcript_sizes: List[int] = []
    attack_rows: List[Dict[str, object]] = []

    for output_id in range(score_count):
        score = float(rng.normal(0.0, 0.25))
        ct = encrypt_score_for_joint_key(
            rng,
            score,
            joint_secret,
            n,
            int(cli.scale),
            int(cli.level),
            int(cli.ckks_error_bound),
        )
        baseline_poly = ring_add(ct.c0, ring_mul(ct.c1, joint_secret))
        baseline_score = decode_constant(baseline_poly, ct.scale)
        accepted: List[Transcript] = []

        for participant_id in selected:
            noise = sample_small(rng, n, bounds["smudge"])
            witness = replace_witness_noise(base_witnesses[participant_id], noise)
            delta, partial_ms = time_ms(lambda: partial_decrypt(ct, witness, 1))
            transcript, transcript_ms = time_ms(
                lambda: make_transcript(
                    cli.sid,
                    output_id,
                    ct,
                    output_policy_hash,
                    registrations[participant_id],
                    1,
                    delta,
                )
            )
            _, hash_ms = time_ms(lambda: ciphertext_hash(ct))
            valid, verify_ms = time_ms(
                lambda: verify_transcript_with_test_witness(
                    transcript,
                    witness,
                    cli.sid,
                    output_id,
                    ct,
                    output_policy_hash,
                    registrations[participant_id],
                    common_a,
                    1,
                    bounds,
                )
            )
            if not valid:
                raise AssertionError("Honest transcript failed verification.")
            accepted.append(transcript)
            partial_times.append(partial_ms)
            transcript_times.append(transcript_ms)
            hash_times.append(hash_ms)
            verify_times.append(verify_ms)
            transcript_sizes.append(transcript.serialized_size_bytes)

        reconstructed, reconstruction_ms = time_ms(lambda: reconstruct(ct, accepted))
        reconstruction_times.append(reconstruction_ms)
        released_score = decode_constant(reconstructed, ct.scale)
        release_error = abs(released_score - baseline_score)
        declared_bound = threshold * bounds["smudge"] / float(cli.scale)
        if release_error > declared_bound + 1e-15:
            raise AssertionError(
                f"Release error {release_error} exceeds bound {declared_bound}"
            )
        release_errors.append(release_error)

        attacked_participant = selected[0]
        registration = registrations[attacked_participant]
        honest_base = base_witnesses[attacked_participant]
        honest_noise = sample_small(rng, n, bounds["smudge"])
        honest_witness = replace_witness_noise(honest_base, honest_noise)

        wrong_ct = encrypt_score_for_joint_key(
            rng,
            float(rng.normal(0.0, 0.25)),
            joint_secret,
            n,
            int(cli.scale),
            int(cli.level),
            int(cli.ckks_error_bound),
        )
        wrong_ct_delta = partial_decrypt(wrong_ct, honest_witness, 1)
        wrong_ct_transcript = make_transcript(
            cli.sid,
            output_id,
            wrong_ct,
            output_policy_hash,
            registration,
            1,
            wrong_ct_delta,
        )

        wrong_secret = sample_small(rng, n, bounds["secret"])
        wrong_key_witness = Witness(
            secret_share=wrong_secret,
            opening_randomness=rng.bytes(32),
            key_error=sample_small(rng, n, bounds["key_error"]),
            smudging_noise=honest_noise,
        )
        wrong_key_delta = partial_decrypt(ct, wrong_key_witness, 1)
        wrong_key_transcript = make_transcript(
            cli.sid,
            output_id,
            ct,
            output_policy_hash,
            registration,
            1,
            wrong_key_delta,
        )

        malformed_delta = rng.integers(0, Q, size=n, dtype=np.int64)
        malformed_transcript = make_transcript(
            cli.sid,
            output_id,
            ct,
            output_policy_hash,
            registration,
            1,
            malformed_delta,
        )

        excessive_noise = np.full(n, bounds["smudge"] + 1, dtype=np.int64)
        excessive_witness = replace_witness_noise(honest_base, excessive_noise)
        excessive_delta = partial_decrypt(ct, excessive_witness, 1)
        excessive_transcript = make_transcript(
            cli.sid,
            output_id,
            ct,
            output_policy_hash,
            registration,
            1,
            excessive_delta,
        )

        attacks = [
            ("wrong_ciphertext", wrong_ct_transcript, honest_witness),
            ("wrong_key", wrong_key_transcript, wrong_key_witness),
            ("malformed_share", malformed_transcript, honest_witness),
            ("excessive_noise", excessive_transcript, excessive_witness),
        ]
        for attack_name, attack_transcript, attack_witness in attacks:
            accepted_attack = verify_transcript_with_test_witness(
                attack_transcript,
                attack_witness,
                cli.sid,
                output_id,
                ct,
                output_policy_hash,
                registration,
                common_a,
                1,
                bounds,
            )
            attack_rows.append(
                {
                    "released_scores": int(score_count),
                    "output_id": int(output_id),
                    "attack": attack_name,
                    "detected": int(not accepted_attack),
                }
            )

    detected = int(sum(row["detected"] for row in attack_rows))
    total_attacks = int(len(attack_rows))
    row = {
        "implementation_level": "Level B",
        "released_scores": int(score_count),
        "threshold_setting": f"{threshold}-of-{participant_count}",
        "accepted_shares": int(threshold * score_count),
        "ring_degree": n,
        "modulus_bits": int(Q.bit_length()),
        "transcript_size_bytes_per_share": float(statistics.mean(transcript_sizes)),
        "partial_decryption_ms_per_share": float(statistics.mean(partial_times)),
        "transcript_generation_ms_per_share": float(statistics.mean(transcript_times)),
        "ciphertext_hash_ms": float(statistics.mean(hash_times)),
        "verification_ms_per_share": float(statistics.mean(verify_times)),
        "reconstruction_ms_per_score": float(statistics.mean(reconstruction_times)),
        "maximum_release_error": float(max(release_errors)),
        "declared_error_bound": float(
            threshold * bounds["smudge"] / float(cli.scale)
        ),
        "correct_release_pass_rate": 1.0,
        "attack_detection_rate": float(detected / max(total_attacks, 1)),
        "attacks_tested": total_attacks,
    }
    return row, attack_rows


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{Level-B prototype overhead of verifiable threshold score release. Times are measured by a witness-assisted transcript verifier over the stated ring; this is not a production zero-knowledge proof benchmark.}",
        r"\label{tab:vtd_prototype}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{cccccccccc}",
        r"\toprule",
        r"Scores & Threshold & Accepted & Transcript/share & Partial dec. & Transcript gen. & Verify/share & Reconstruct & Max error & Attack detection \\",
        r" &  & shares & (KB) & (ms) & (ms) & (ms) & (ms/score) &  &  \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{int(row.released_scores)} & {row.threshold_setting} & "
            f"{int(row.accepted_shares)} & "
            f"{row.transcript_size_bytes_per_share / 1024.0:.2f} & "
            f"{row.partial_decryption_ms_per_share:.3f} & "
            f"{row.transcript_generation_ms_per_share:.3f} & "
            f"{row.verification_ms_per_share:.3f} & "
            f"{row.reconstruction_ms_per_score:.3f} & "
            f"${row.maximum_release_error:.2e}$ & "
            f"{100.0 * row.attack_detection_rate:.1f}\\% \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--released-scores", default="1,32,128")
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--participants", type=int, default=5)
    parser.add_argument("--ring-degree", type=int, default=1024)
    parser.add_argument("--scale", type=int, default=2**30)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--secret-bound", type=int, default=1)
    parser.add_argument("--key-error-bound", type=int, default=3)
    parser.add_argument("--smudge-bound", type=int, default=8)
    parser.add_argument("--ckks-error-bound", type=int, default=4)
    parser.add_argument("--sid", default="paper-vtd-prototype-20260611")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / "verifiable_threshold_release_prototype",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.threshold <= 0 or cli.threshold > cli.participants:
        raise ValueError("threshold must be in [1, participants]")
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cli.seed))
    rows: List[Dict[str, object]] = []
    attack_rows: List[Dict[str, object]] = []
    for score_count in parse_csv_ints(cli.released_scores):
        row, attacks = run_configuration(score_count, cli, rng)
        rows.append(row)
        attack_rows.extend(attacks)
        print(
            f"[vtd] scores={score_count} threshold={row['threshold_setting']} "
            f"verify={row['verification_ms_per_share']:.3f}ms "
            f"error={row['maximum_release_error']:.3e} "
            f"attacks={100.0 * row['attack_detection_rate']:.1f}%",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    attacks = pd.DataFrame(attack_rows)
    summary.to_csv(output_dir / "vtd_prototype_summary.csv", index=False)
    attacks.to_csv(output_dir / "vtd_attack_results.csv", index=False)
    write_latex_table(summary, output_dir / "table_vtd_prototype.tex")
    report = {
        "implementation_level": "Level B",
        "production_nizk": False,
        "witness_assisted_verifier": True,
        "all_correct_releases_passed": bool(
            (summary["correct_release_pass_rate"] == 1.0).all()
        ),
        "all_attacks_detected": bool((attacks["detected"] == 1).all()),
        "all_errors_within_declared_bound": bool(
            (
                summary["maximum_release_error"]
                <= summary["declared_error_bound"] + 1e-15
            ).all()
        ),
        "parameters": {
            "threshold": int(cli.threshold),
            "participants": int(cli.participants),
            "ring_degree": int(cli.ring_degree),
            "modulus": Q,
            "scale": int(cli.scale),
            "secret_bound": int(cli.secret_bound),
            "key_error_bound": int(cli.key_error_bound),
            "smudge_bound": int(cli.smudge_bound),
        },
    }
    (output_dir / "vtd_test_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"[done] {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
