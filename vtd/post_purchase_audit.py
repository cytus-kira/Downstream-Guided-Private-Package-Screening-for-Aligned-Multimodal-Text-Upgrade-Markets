#!/usr/bin/env python
"""Post-purchase deployed-score consistency audit over delivered real rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from export_package_fixture import plaintext_scores, row_digest


def package_digest(phi: np.ndarray, source: np.ndarray, members: list[int]) -> str:
    parts = [row_digest(phi[i], int(source[i])) for i in members]
    return hashlib.sha256("|".join(parts).encode("ascii")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--decoded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    buyer = json.loads(args.decoded.read_text(encoding="utf-8"))
    decoded = np.asarray(buyer["decoded_scores"], dtype=np.float64)
    with np.load(fixture["source_npz"], allow_pickle=True) as market:
        phi = np.asarray(market["phi"], dtype=np.float32)
        source = np.asarray(market["base_source_idx"], dtype=np.int64)
        rows: list[dict] = []
        all_members = [list(map(int, group)) for group in fixture["members"]]
        for pos, members in enumerate(all_members):
            registered_feature = np.asarray(fixture["features"][pos], dtype=np.float64)
            registered_digest = fixture["rows_digest"][pos]
            cases: list[tuple[str, list[int], np.ndarray]] = [("honest", members, phi.copy())]
            if len(all_members) > 1:
                other = all_members[(pos + 1) % len(all_members)]
            else:
                outside = next(i for i in range(len(phi)) if i not in set(members))
                other = [outside]
            replaced = list(members)
            if other:
                replaced[0] = other[0]
            cases.append(("replaced_row", replaced, phi.copy()))
            modified_phi = phi.copy()
            modified_phi[members[0], 0] += 0.05
            cases.append(("modified_text_feature", list(members), modified_phi))
            changed = list(members)
            if other:
                changed = sorted(set(changed + [other[0]]))
            cases.append(("changed_membership", changed, phi.copy()))
            for case, delivered_members, delivered_phi in cases:
                delivered_feature = np.mean(delivered_phi[delivered_members], axis=0, dtype=np.float64)
                recomputed_score = float(plaintext_scores(delivered_feature[None, :], int(fixture["landmarks"]))[0])
                digest = package_digest(delivered_phi, source, delivered_members)
                representation_diff = float(np.max(np.abs(delivered_feature - registered_feature)))
                score_diff = abs(recomputed_score - float(decoded[pos]))
                accept = digest == registered_digest and representation_diff <= args.tolerance and score_diff <= args.tolerance
                rows.append(
                    {
                        "dataset": fixture["dataset"],
                        "seed": fixture["seed"],
                        "package_id": fixture["package_ids"][pos],
                        "case": case,
                        "accepted": int(accept),
                        "detected": int(not accept),
                        "score_difference": score_diff,
                        "representation_max_abs_difference": representation_diff,
                        "registered_tolerance": args.tolerance,
                        "proves_deployed_score_consistency_only": 1,
                        "proves_package_cryptographic_identity": 0,
                        "proves_downstream_utility": 0,
                        "solves_transaction_fairness": 0,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"cases": len(rows), "honest_accepted": sum(r["accepted"] for r in rows if r["case"] == "honest"), "tampering_detected": sum(r["detected"] for r in rows if r["case"] != "honest")}, indent=2))


if __name__ == "__main__":
    main()
