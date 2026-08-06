from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

FEATURE_KEYS = [
    "fact_accuracy",
    "contribution_consistency_rate",
    "taxonomy_structural_consistency",
    "argument_support_coverage",
    "gap_future_work_anchor_coverage",
    "contrast_topic_coverage",
    "synthesis_topic_coverage",
    "synthesis_density",
    "scope_methodology_existence",
    "scope_topic_coverage",
    "contribution_fulfillment_ratio",
    "citation_coverage",
    "reference_pool_coverage",
    "topic_coverage",
    "missing_topic_coverage",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def numeric_values(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            pass
    return out


def target_from_record(record: dict[str, Any], field: str) -> float | None:
    vals = []
    for review in record.get("reviews", []) or []:
        content = review.get("content", {}) if isinstance(review, dict) else {}
        vals.append(content.get(field))
    nums = numeric_values(vals)
    if not nums:
        return None
    raw = mean(nums)
    # OpenReview survey records are usually 1-10 for rating and 1-5 for sub-scores.
    scale_max = 10.0 if field == "rating" or raw > 5 else 5.0
    return max(0.0, min(100.0, 100.0 * (raw - 1.0) / (scale_max - 1.0)))


def golden_targets(golden_data_dir: Path, field: str) -> dict[str, float]:
    targets = {}
    for path in golden_data_dir.glob("*.json"):
        record = load_json(path)
        y = target_from_record(record, field)
        if y is None:
            continue
        keys = {path.stem, str(record.get("openreview_forum_id", "")), str(record.get("paper_title", ""))}
        match = re.match(r"^(\d+)_", path.name)
        if match:
            keys.add(str(int(match.group(1)) - 1))
            keys.add(match.group(1))
        for key in keys:
            if key:
                targets[key] = y
    return targets


def walk_metrics(value: Any) -> dict[str, float]:
    metrics = {}
    def visit(node: Any):
        if isinstance(node, dict):
            for key, item in node.items():
                if key in FEATURE_KEYS and isinstance(item, (int, float)) and math.isfinite(float(item)):
                    metrics[key] = float(item)
                if key in {"metrics", "adequacy_metrics", "caps", "aggregate_review", "evaluations"} or isinstance(item, (dict, list)):
                    visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)
    visit(value)
    return metrics


def run_key(path: Path, data: dict[str, Any]) -> str:
    for key in ("index", "original_index", "openreview_forum_id", "paper_id", "paper_title", "title"):
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    parent = path.parent.name
    return parent if parent != path.parent.parent.name else path.stem


def load_runs(runs_dir: Path) -> list[tuple[str, dict[str, float], Path]]:
    candidates = list(runs_dir.rglob("result.json")) + list(runs_dir.rglob("14_aggregate_review.json"))
    rows = []
    for path in candidates:
        try:
            data = load_json(path)
        except Exception:
            continue
        feats = walk_metrics(data)
        if not feats:
            continue
        rows.append((run_key(path, data), feats, path))
    return rows


def ridge_fit(x: list[list[float]], y: list[float], alpha: float) -> tuple[list[float], float]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("numpy is required for fitting") from exc
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Z = (X - mu) / sigma
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    b = Z.T @ Y
    coef_std = np.linalg.solve(A, b)
    intercept = float(Y.mean() - (mu / sigma) @ coef_std)
    coef = (coef_std / sigma).tolist()
    pred = X @ (coef_std / sigma) + intercept
    mae = float(np.mean(np.abs(pred - Y)))
    return [float(c) for c in coef], mae


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit TrustSurvey numeric adequacy/integrity score weights from cached runs.")
    parser.add_argument("--golden-data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--runs-dir", type=Path, required=True, help="Directory containing TrustSurvey evaluation outputs.")
    parser.add_argument("--target-field", default="rating", choices=["rating", "soundness", "presentation", "contribution"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "adequacy_score_fit.json")
    args = parser.parse_args()

    targets = golden_targets(args.golden_data, args.target_field)
    rows = []
    for key, feats, path in load_runs(args.runs_dir):
        y = targets.get(key)
        if y is None:
            continue
        rows.append((key, feats, y, path))
    if len(rows) < 5:
        raise SystemExit(f"not enough aligned rows: {len(rows)}")

    x = [[rows_i[1].get(key, 1.0) for key in FEATURE_KEYS] for rows_i in rows]
    y = [rows_i[2] for rows_i in rows]
    coef, mae = ridge_fit(x, y, args.alpha)
    payload = {
        "target_field": args.target_field,
        "n": len(rows),
        "features": FEATURE_KEYS,
        "weights": dict(zip(FEATURE_KEYS, coef)),
        "intercept": mean(y) - sum(mean(col) * w for col, w in zip(zip(*x), coef)),
        "train_mae": mae,
        "aligned_keys": [key for key, *_ in rows],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["target_field", "n", "train_mae", "output"] if k in payload}, ensure_ascii=False))


if __name__ == "__main__":
    main()
