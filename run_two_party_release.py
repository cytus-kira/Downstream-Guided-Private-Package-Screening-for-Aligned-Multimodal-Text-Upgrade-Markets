#!/usr/bin/env python
"""Run the real isolated buyer/market 2-of-2 verifiable CKKS release path."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VTD = ROOT / "vtd"
DEFAULT_GO = Path(shutil.which("go") or "go")


def run(cmd: list[str], *, cwd: Path = ROOT, capture: bool = False) -> tuple[float, str]:
    start = time.perf_counter()
    completed = subprocess.run(cmd, cwd=str(cwd), check=True, text=True, encoding="utf-8", capture_output=capture)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, completed.stdout if capture else ""


def build_binary(go: Path) -> Path:
    binary = VTD / "bin" / "vtd.exe"
    binary.parent.mkdir(parents=True, exist_ok=True)
    sources = list((VTD / "cmd").rglob("*.go")) + list((VTD / "internal").rglob("*.go")) + [VTD / "go.mod", VTD / "go.sum"]
    if not binary.exists() or any(p.stat().st_mtime > binary.stat().st_mtime for p in sources if p.exists()):
        env = os.environ.copy()
        env["GOTOOLCHAIN"] = "local"
        subprocess.run([str(go), "build", "-buildvcs=false", "-o", str(binary), "./cmd/vtd"], cwd=str(VTD), env=env, check=True)
    return binary


def command(binary: Path, name: str, run_root: Path, *args: str, capture: bool = False) -> tuple[float, str]:
    return run([str(binary), name, "--root", str(run_root), *args], capture=capture)


def numeric_json(stdout: str) -> dict:
    return json.loads(stdout[stdout.index("{") :])


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=int, choices=[1, 32, 128], default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="hateful_memes")
    parser.add_argument("--profile", default="noise")
    parser.add_argument("--preset", choices=["smoke", "production"], default="production")
    parser.add_argument("--landmarks", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument(
        "--reuse-proof-from",
        type=Path,
        default=None,
        help="Reuse Groth16 circuit/setup artifacts from a compatible completed run.",
    )
    parser.add_argument("--go", type=Path, default=DEFAULT_GO)
    args = parser.parse_args()
    landmarks = args.landmarks if args.landmarks is not None else (8 if args.preset == "smoke" else 1000)
    if not args.go.exists():
        raise SystemExit(f"Go toolchain missing at {args.go}; install Go 1.25.9 or pass --go C:\\path\\to\\go.exe")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = (args.run_root or (VTD / "runs" / f"vtd_{args.preset}_b{args.scores}_s{args.seed}_{stamp}")).resolve()
    if run_root.exists():
        shutil.rmtree(run_root)
    binary = build_binary(args.go)

    process_times: dict[str, object] = {}
    process_times["init_ms"], _ = command(binary, "init", run_root, "--preset", args.preset)
    # Each key-generation call is a distinct OS process with access only to its role path.
    process_times["buyer_round1_ms"], _ = command(binary, "party-round1", run_root, "--role", "buyer")
    process_times["market_round1_ms"], _ = command(binary, "party-round1", run_root, "--role", "market")
    process_times["coordinator_round1_ms"], _ = command(binary, "coordinator-round1", run_root)
    process_times["buyer_round2_ms"], _ = command(binary, "party-round2", run_root, "--role", "buyer")
    process_times["market_round2_ms"], _ = command(binary, "party-round2", run_root, "--role", "market")
    process_times["coordinator_finalize_ms"], _ = command(binary, "coordinator-finalize", run_root)

    fixture = run_root / "auditor" / "package_fixture.json"
    export_cmd = [sys.executable, str(VTD / "export_package_fixture.py"), "--repo", str(ROOT), "--output", str(fixture), "--scores", str(args.scores), "--seed", str(args.seed), "--dataset", args.dataset, "--profile", args.profile, "--landmarks", str(landmarks)]
    process_times["fixture_export_ms"], _ = run(export_cmd, capture=True)
    fixture_data = json.loads(fixture.read_text(encoding="utf-8"))
    process_times["market_encrypt_ms"], _ = command(binary, "market-encrypt", run_root, "--fixture", str(fixture))
    score_ms, score_out = command(binary, "buyer-score", run_root, "--landmarks", str(landmarks), capture=True)
    process_times["buyer_score_process_ms"] = score_ms
    scorer = numeric_json(score_out)
    process_times["buyer_authorize_ms"], _ = command(binary, "buyer-authorize", run_root, "--session", f"seed-{fixture_data['seed']}-batch-{args.scores}", "--output", f"{args.dataset}-{args.profile}-{args.scores}")

    if args.reuse_proof_from is not None:
        source = args.reuse_proof_from.resolve()
        source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        target_config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
        if source_config != target_config:
            raise SystemExit("--reuse-proof-from has incompatible CKKS/circuit parameters")
        shutil.copytree(source / "public" / "proof", run_root / "public" / "proof")
        shutil.copy2(source / "results" / "proof_setup.json", run_root / "results" / "proof_setup.json")
        proof_setup = json.loads((run_root / "results" / "proof_setup.json").read_text(encoding="utf-8"))
        process_times["proof_setup_process_ms"] = 0.0
        process_times["proof_setup_reused_from"] = str(source)
    else:
        proof_setup_ms, proof_setup_out = command(binary, "proof-setup", run_root, capture=True)
        process_times["proof_setup_process_ms"] = proof_setup_ms
        proof_setup = numeric_json(proof_setup_out)

    rows: list[dict] = []
    decoded_last: list[float] = []
    for repeat in range(args.repeats):
        end_start = time.perf_counter()
        market_ms, market_out = command(binary, "market-release", run_root, "--landmarks", str(landmarks), capture=True)
        buyer_ms, buyer_out = command(binary, "buyer-verify", run_root, capture=True)
        e2e_ms = (time.perf_counter() - end_start) * 1000.0
        market_result = json.loads((run_root / "results" / "market_release.json").read_text(encoding="utf-8"))
        buyer_result = numeric_json(buyer_out)
        decoded_last = buyer_result["decoded_scores"]
        expected = fixture_data["plaintext_scores"]
        errors = [abs(a - b) for a, b in zip(decoded_last, expected)]
        rows.append(
            {
                "dataset": args.dataset,
                "seed": fixture_data["seed"],
                "package_id": "packed-batch",
                "batch_size": args.scores,
                "repeat": repeat,
                "ckks_parameter_digest": json.loads((run_root / "config.json").read_text())["log_q"],
                "proof_backend": "gnark-v0.15.0-groth16-bn254",
                "partial_decryption_ms": market_result["partial_decryption_ms"],
                "proof_generation_ms": market_result["proof_generation_ms"],
                "verification_ms": buyer_result["proof_verification_ms"],
                "proof_bytes": buyer_result["proof_bytes"],
                "communication_bytes": buyer_result["market_to_buyer_bytes"],
                "reconstruction_ms": buyer_result["reconstruction_ms"],
                "ckks_decoding_ms": buyer_result["ckks_decoding_ms"],
                "end_to_end_localhost_ms": e2e_ms,
                "market_process_ms": market_ms,
                "buyer_process_ms": buyer_ms,
                "decoded_max_abs_error": max(errors),
                "decoded_mean_abs_error": statistics.mean(errors),
                "accepted": buyer_result["accepted"],
                "attack_type": "honest",
                "constraints": market_result["constraints"],
                "public_variables": market_result["public_variables"],
            }
        )

    summary = {
        "run_root": str(run_root),
        "preset": args.preset,
        "scores": args.scores,
        "requested_seed": args.seed,
        "resolved_seed": fixture_data["seed"],
        "dataset": args.dataset,
        "profile": args.profile,
        "landmarks": landmarks,
        "repeats": args.repeats,
        "source_npz": fixture_data["source_npz"],
        "scorer": scorer,
        "proof_setup": proof_setup,
        "process_times": process_times,
        "proof_generation_ms": summarize([row["proof_generation_ms"] for row in rows]),
        "proof_verification_ms": summarize([row["verification_ms"] for row in rows]),
        "end_to_end_localhost_ms": summarize([row["end_to_end_localhost_ms"] for row in rows]),
        "decoded_max_abs_error": max(row["decoded_max_abs_error"] for row in rows),
        "decoded_mean_abs_error": statistics.mean(row["decoded_mean_abs_error"] for row in rows),
        "proof_bytes": rows[0]["proof_bytes"],
        "communication_bytes": rows[0]["communication_bytes"],
        "honest_acceptance": all(row["accepted"] for row in rows),
    }

    protected = {
        name: (run_root / "messages" / name).read_bytes()
        for name in ["market_contribution.bin", "release.proof", "release_statement.json", "signed_policy.json"]
    }
    attack_rows: list[dict] = []

    def restore() -> None:
        for name, data in protected.items():
            (run_root / "messages" / name).write_bytes(data)

    for attack, reported_name in [
        ("wrong-target", "wrong-target"),
        ("wrong-key", "wrong-key"),
        ("malformed-share", "malformed-share"),
        ("excessive-noise", "large-contribution-tampering"),
    ]:
        restore()
        command(binary, "attack-contribution", run_root, "--attack", attack)
        attempted = 1
        accepted = 0
        try:
            command(binary, "buyer-verify", run_root, capture=True)
            accepted = 1
        except subprocess.CalledProcessError:
            pass
        attack_rows.append({"dataset": args.dataset, "seed": fixture_data["seed"], "batch_size": args.scores, "attack_type": reported_name, "attempted_cases": attempted, "accepted_cases": accepted, "rejected_cases": attempted - accepted, "expected": "reject", "observed": "accept" if accepted else "reject"})

    restore()
    statement_path = run_root / "messages" / "release_statement.json"
    statement = json.loads(statement_path.read_text(encoding="utf-8"))
    statement["ckks_level"] = statement["ckks_level"] + 1
    statement_path.write_text(json.dumps(statement, indent=2), encoding="utf-8")
    try:
        command(binary, "buyer-verify", run_root, capture=True); accepted = 1
    except subprocess.CalledProcessError:
        accepted = 0
    attack_rows.append({"dataset": args.dataset, "seed": fixture_data["seed"], "batch_size": args.scores, "attack_type": "wrong-level", "attempted_cases": 1, "accepted_cases": accepted, "rejected_cases": 1-accepted, "expected": "reject", "observed": "accept" if accepted else "reject"})

    restore()
    proof_path = run_root / "messages" / "release.proof"
    tampered = bytearray(proof_path.read_bytes()); tampered[len(tampered)//2] ^= 1; proof_path.write_bytes(tampered)
    try:
        command(binary, "buyer-verify", run_root, capture=True); accepted = 1
    except subprocess.CalledProcessError:
        accepted = 0
    attack_rows.append({"dataset": args.dataset, "seed": fixture_data["seed"], "batch_size": args.scores, "attack_type": "proof-tampering", "attempted_cases": 1, "accepted_cases": accepted, "rejected_cases": 1-accepted, "expected": "reject", "observed": "accept" if accepted else "reject"})

    restore()
    command(binary, "buyer-authorize", run_root, "--session", f"replay-seed-{fixture_data['seed']}", "--output", f"{args.dataset}-{args.profile}-{args.scores}")
    try:
        command(binary, "buyer-verify", run_root, capture=True); accepted = 1
    except subprocess.CalledProcessError:
        accepted = 0
    attack_rows.append({"dataset": args.dataset, "seed": fixture_data["seed"], "batch_size": args.scores, "attack_type": "replay", "attempted_cases": 1, "accepted_cases": accepted, "rejected_cases": 1-accepted, "expected": "reject", "observed": "accept" if accepted else "reject"})

    restore()
    command(binary, "buyer-authorize", run_root, "--session", f"unauthorized-seed-{fixture_data['seed']}", "--output", f"{args.dataset}-{args.profile}-{args.scores}", "--permit=false")
    try:
        command(binary, "market-release", run_root, "--landmarks", str(landmarks), capture=True); accepted = 1
    except subprocess.CalledProcessError:
        accepted = 0
    attack_rows.append({"dataset": args.dataset, "seed": fixture_data["seed"], "batch_size": args.scores, "attack_type": "unauthorized-output", "attempted_cases": 1, "accepted_cases": accepted, "rejected_cases": 1-accepted, "expected": "refuse", "observed": "accept" if accepted else "refuse"})
    restore()
    summary["attack_rejection"] = {row["attack_type"]: bool(row["rejected_cases"]) for row in attack_rows}
    audit_out = run_root / "results" / "post_purchase_audit.csv"
    audit_cmd = [sys.executable, str(VTD / "post_purchase_audit.py"), "--fixture", str(fixture), "--decoded", str(run_root / "results" / "buyer_release.json"), "--output", str(audit_out), "--tolerance", str(max(1e-5, summary["decoded_max_abs_error"] + 1e-5))]
    _, audit_stdout = run(audit_cmd, capture=True)
    summary["post_purchase_audit"] = numeric_json(audit_stdout)
    write_csv(run_root / "results" / "vtd_2of2_release_runs.csv", rows)
    write_csv(run_root / "results" / "vtd_2of2_attacks.csv", attack_rows)
    (run_root / "results" / "vtd_2of2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
