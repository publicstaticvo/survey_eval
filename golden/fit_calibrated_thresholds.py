from __future__ import annotations

"""Fit calibrated risk thresholds from reviewer-observed concerns.

This script implements the paper's threshold-estimation calibration rather than
acceptance-score regression.  For each Calibrated category, it fits a low-value
risk trigger on development-set indicators and treats papers without an observed
comment as unlabeled comparators, not verified negatives.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold

from survey_eval.golden.fit_adequacy_thresholds import build_features, load_positive_papers


SPECS: dict[str, dict[str, Any]] = {
    "gap": {
        "features": ["gap_coverage", "gap_volume"],
        "rule": "any_below",
    },
    "contrast": {
        "features": ["contrast_coverage", "contrast_volume"],
        "rule": "any_below",
    },
    "synthesis": {
        "features": ["synthesis_coverage", "synthesis_volume"],
        "rule": "any_below",
    },
    "scope": {
        "features": ["scope_exists"],
        "rule": "any_below",
    },
    "contribution": {
        "features": ["contribution_exists"],
        "rule": "any_below",
    },
    "reference": {
        "features": ["reference_topic_coverage", "citations_per_topic"],
        "rule": "any_below",
    },
    "topic": {
        "features": ["topic_count"],
        "rule": "any_below",
    },
}


def threshold_grid(values: np.ndarray, grid_size: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.array([0.0])
    return np.unique(np.quantile(finite, np.linspace(0.0, 1.0, grid_size)))


def predict_any_below(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    pred = np.zeros(len(frame), dtype=bool)
    for column, threshold in thresholds.items():
        pred |= frame[column].to_numpy(float) < threshold
    return pred


def score_prediction(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    precision, recall, fbeta, _ = precision_recall_fscore_support(
        y,
        pred,
        beta=0.5,
        average="binary",
        zero_division=0,
    )
    return {
        "observed_precision_lb": float(precision),
        "observed_recall": float(recall),
        "f0_5": float(fbeta),
        "flag_rate": float(pred.mean()),
    }


def better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    key = (
        candidate["f0_5"],
        candidate["observed_precision_lb"],
        -abs(candidate["flag_rate"] - candidate["positive_rate"]),
    )
    old = (
        best["f0_5"],
        best["observed_precision_lb"],
        -abs(best["flag_rate"] - best["positive_rate"]),
    )
    return key > old


def fit_thresholds(frame: pd.DataFrame, y: np.ndarray, columns: list[str], grid_size: int, max_flag_rate: float) -> dict[str, Any]:
    grids = [threshold_grid(frame[column].to_numpy(float), grid_size) for column in columns]
    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None

    def visit(index: int, current: dict[str, float]) -> None:
        nonlocal best, fallback
        if index == len(columns):
            pred = predict_any_below(frame, current)
            metrics = score_prediction(y, pred)
            candidate = {
                "thresholds": dict(current),
                "positive_rate": float(y.mean()),
                **metrics,
            }
            if better(candidate, fallback):
                fallback = candidate
            if candidate["flag_rate"] <= max_flag_rate and better(candidate, best):
                best = candidate
            return
        column = columns[index]
        for threshold in grids[index]:
            current[column] = float(threshold)
            visit(index + 1, current)
        current.pop(column, None)

    visit(0, {})
    if best is None:
        best = fallback
    assert best is not None
    return best


def repeated_cv(
    features: pd.DataFrame,
    y: np.ndarray,
    columns: list[str],
    repeats: int,
    folds: int,
    grid_size: int,
    max_flag_rate: float,
    seed: int,
) -> dict[str, Any]:
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    n_splits = min(folds, positives, negatives)
    if n_splits < 2:
        return {"usable": False, "reason": "fewer than two positive or comparator examples"}

    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=seed)
    rows = []
    pred_sum = np.zeros(len(features), dtype=float)
    pred_count = np.zeros(len(features), dtype=float)
    fitted = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(features, y), start=1):
        train = features.iloc[train_idx]
        test = features.iloc[test_idx]
        model = fit_thresholds(train, y[train_idx], columns, grid_size, max_flag_rate)
        pred = predict_any_below(test, model["thresholds"])
        pred_sum[test_idx] += pred.astype(float)
        pred_count[test_idx] += 1.0
        fitted.append(model)
        rows.append(pd.DataFrame({"paper_id": test["paper_id"].to_numpy(), "fold": fold_id, "y": y[test_idx], "pred": pred.astype(int)}))

    all_rows = pd.concat(rows, ignore_index=True)
    metrics = score_prediction(all_rows["y"].to_numpy(int), all_rows["pred"].to_numpy(bool))
    risk_vote = pred_sum / np.maximum(pred_count, 1.0)
    doc_pred = risk_vote >= 0.5
    doc_metrics = score_prediction(y, doc_pred)
    threshold_summary = {}
    for column in columns:
        values = np.array([model["thresholds"][column] for model in fitted], dtype=float)
        threshold_summary[column] = {
            "median": float(np.median(values)),
            "iqr": [float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))],
        }
    return {
        "usable": True,
        "folds": int(n_splits),
        "repeats": int(repeats),
        "fold_level": metrics,
        "document_vote": doc_metrics,
        "threshold_summary": threshold_summary,
        "risk_votes": [float(v) for v in risk_vote],
    }


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, reps: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    n = len(y)
    for _ in range(reps):
        for _attempt in range(100):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) == 2:
                break
        else:
            continue
        m = score_prediction(y[idx], pred[idx])
        for key in ["observed_precision_lb", "observed_recall", "f0_5", "flag_rate"]:
            values[key].append(m[key])
    return {key: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for key, v in values.items()}


def stability_decision(category: str, y: np.ndarray, full_fit: dict[str, Any], cv: dict[str, Any]) -> tuple[bool, str]:
    pos_rate = float(y.mean())
    if pos_rate >= 0.80:
        return False, "observed concern is saturated; reviewer comments do not provide a useful threshold"
    if not cv.get("usable"):
        return False, str(cv.get("reason"))
    if cv["document_vote"]["observed_precision_lb"] < 0.45:
        return False, "held-out observed precision lower bound is too low"
    unstable = []
    for column, summary in cv["threshold_summary"].items():
        q1, q3 = summary["iqr"]
        scale = max(abs(summary["median"]), 1.0 if column.endswith("exists") else 1e-6)
        if (q3 - q1) / scale > 1.0 and column not in {"gap_volume", "contrast_volume", "synthesis_volume"}:
            unstable.append(column)
    if unstable:
        return False, "unstable threshold: " + ", ".join(unstable)
    return True, "stable enough for score-capping calibration"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-content", type=Path, default=Path(__file__).parent / "pdf_content")
    parser.add_argument("--features", type=Path, default=Path(__file__).parent / "adequacy_thresholds" / "features.csv")
    parser.add_argument("--review-analysis", type=Path, default=Path(__file__).parent / "review_analyze")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "calibrated_thresholds")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-flag-rate", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    if args.features.exists():
        features = pd.read_csv(args.features).sort_values("paper_id").reset_index(drop=True)
    else:
        features = build_features(args.pdf_content).sort_values("paper_id").reset_index(drop=True)
    labels = load_positive_papers(args.review_analysis)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_dir / "features.csv", index=False)

    output: dict[str, Any] = {
        "method": "per-category calibrated threshold estimation from reviewer-observed concerns",
        "n_papers": int(len(features)),
        "repeats": args.repeats,
        "folds": args.folds,
        "grid_size": args.grid_size,
        "max_flag_rate": args.max_flag_rate,
        "categories": {},
    }
    summary_rows = []
    prediction_rows = []
    for category, spec in SPECS.items():
        columns = spec["features"]
        y = features["paper_id"].isin(labels[category]).to_numpy(int)
        full_fit = fit_thresholds(features, y, columns, args.grid_size, args.max_flag_rate)
        full_pred = predict_any_below(features, full_fit["thresholds"])
        cv = repeated_cv(features, y, columns, args.repeats, args.folds, args.grid_size, args.max_flag_rate, args.seed)
        score_bearing, reason = stability_decision(category, y, full_fit, cv)
        ci = bootstrap_ci(y, full_pred, args.bootstrap, args.seed + len(summary_rows) + 1)
        output["categories"][category] = {
            "positive_papers": int(y.sum()),
            "positive_rate": float(y.mean()),
            "features": columns,
            "full_fit": full_fit,
            "cross_validation": cv,
            "bootstrap_95_ci_full_fit": ci,
            "score_bearing": bool(score_bearing),
            "decision_reason": reason,
        }
        cv_doc = cv.get("document_vote", {}) if cv.get("usable") else {}
        summary_rows.append({
            "category": category,
            "positive_papers": int(y.sum()),
            "positive_rate": float(y.mean()),
            "score_bearing": bool(score_bearing),
            "decision_reason": reason,
            "full_precision_lb": full_fit["observed_precision_lb"],
            "full_recall": full_fit["observed_recall"],
            "full_f0_5": full_fit["f0_5"],
            "cv_precision_lb": cv_doc.get("observed_precision_lb"),
            "cv_recall": cv_doc.get("observed_recall"),
            "cv_f0_5": cv_doc.get("f0_5"),
            "thresholds": json.dumps(full_fit["thresholds"], ensure_ascii=False),
        })
        for paper_id, label, vote in zip(features["paper_id"], y, cv.get("risk_votes", [np.nan] * len(features))):
            prediction_rows.append({"category": category, "paper_id": int(paper_id), "observed_concern": int(label), "risk_vote": float(vote)})

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "threshold_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "cross_validated_risk_votes.csv", index=False)
    (args.output_dir / "thresholds.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n_papers": output["n_papers"], "summary": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
