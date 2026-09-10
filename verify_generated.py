"""Verify regenerated scientific results against the published dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DIMENSIONLESS_FIELDS = (
    "min_scaled_jacobian",
    "orthogonality_score",
    "center_rms_cosine",
    "area_cv",
    "edge_cv",
    "aspect_p95",
)
DIMENSIONAL_FIELDS = ("directional_length_difference", "min_center_jacobian")


def _read_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {(row["domain"], row["method"]): row for row in rows}


def _assert_close(label: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=5e-4, abs_tol=5e-8):
        raise AssertionError(f"{label}: {actual:.12g} != {expected:.12g}")


def verify(reference: Path, candidate: Path) -> None:
    expected_rows = _read_rows(reference / "results.csv")
    actual_rows = _read_rows(candidate / "results.csv")
    if actual_rows.keys() != expected_rows.keys():
        raise AssertionError("The regenerated domain/method rows do not match")

    for key, expected in expected_rows.items():
        actual = actual_rows[key]
        for field in ("n_xi", "n_eta", "converged", "inverted_cells"):
            if actual[field] != expected[field]:
                raise AssertionError(f"{key} {field}: {actual[field]} != {expected[field]}")
        for field in DIMENSIONLESS_FIELDS + DIMENSIONAL_FIELDS:
            _assert_close(
                f"{key} {field}", float(actual[field]), float(expected[field])
            )

        method = key[1]
        residual = float(actual["residual"])
        if method == "Метод упругих нитей":
            tolerance = 1e-10
        elif method == "Винслоу":
            tolerance = 2e-6
        else:
            tolerance = 2e-5
        if residual > tolerance:
            raise AssertionError(f"{key} residual {residual:.3g} exceeds {tolerance:.3g}")

    control = json.loads((candidate / "adaptive_mu0_control.json").read_text("utf-8"))
    metrics = control["metrics"]
    if control["converged"] or control["iterations"] != 5000:
        raise AssertionError("The mu=0 control must reach the 5000-iteration limit")
    if metrics["inverted_cells"] != 0:
        raise AssertionError("The mu=0 control unexpectedly contains inverted cells")
    if not 0.0 < float(metrics["min_scaled_jacobian"]) < 1e-6:
        raise AssertionError("The mu=0 control no longer approaches degeneracy")
    if float(control["residual"]) <= float(control["parameters"]["gradient_tolerance"]):
        raise AssertionError("The mu=0 control unexpectedly meets the gradient tolerance")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    arguments = parser.parse_args()
    verify(arguments.reference, arguments.candidate)
    print("Regenerated scientific metrics agree with the published dataset.")


if __name__ == "__main__":
    main()
