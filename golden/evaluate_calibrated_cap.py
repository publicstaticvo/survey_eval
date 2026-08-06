from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def deficit(row: pd.Series, thresholds: dict[str, float]) -> float:
    parts = []
    for column, threshold in thresholds.items():
        if threshold <= 0:
            parts.append(0.0)
        else:
            parts.append(max(0.0, 1.0 - float(row[column]) / threshold))
    return float(max(parts)) if parts else 0.0


def calibration_slope_intercept(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1 - 1e-6))
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    model.fit(logits.reshape(-1, 1), y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def bootstrap_ci(y: np.ndarray, p: np.ndarray, reps: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    values = defaultdict(list)
    n = len(y)
    for _ in range(reps):
        for _attempt in range(100):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) == 2:
                break
        else:
            continue
        yy = y[idx]
        pp = p[idx]
        values["auroc"].append(roc_auc_score(yy, pp))
        values["pr_auc"].append(average_precision_score(yy, pp))
        values["brier"].append(brier_score_loss(yy, pp))
        try:
            inter, slope = calibration_slope_intercept(yy, pp)
            values["calibration_intercept"].append(inter)
            values["calibration_slope"].append(slope)
        except Exception:
            pass
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in values.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=Path, default=Path(__file__).parent / "calibrated_thresholds" / "thresholds.json")
    parser.add_argument("--features", type=Path, default=Path(__file__).parent / "calibrated_thresholds" / "features.csv")
    parser.add_argument("--acceptance", type=Path, default=Path(__file__).parent / "acceptance_fit" / "training_features.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "calibrated_thresholds")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    threshold_data = json.loads(args.thresholds.read_text(encoding="utf-8"))
    features = pd.read_csv(args.features)
    acceptance = pd.read_csv(args.acceptance)[["paper_id", "accepted", "integrity_internal", "integrity_factual", "integrity_taxonomy", "integrity_argument"]]
    df = features.merge(acceptance, on="paper_id", how="inner")

    stable = {
        c: d["full_fit"]["thresholds"]
        for c, d in threshold_data["categories"].items()
        if d.get("score_bearing")
    }
    for category, thresholds in stable.items():
        df[f"risk_{category}"] = df.apply(lambda row: deficit(row, thresholds), axis=1)
    risk_columns = [f"risk_{category}" for category in stable]
    df["calibrated_cap"] = 100.0 * (1.0 - df[risk_columns].max(axis=1)) if risk_columns else 100.0
    # Simple monotone integrity display, not fitted as a regression target.
    df["integrity_count"] = df[["integrity_internal", "integrity_factual", "integrity_taxonomy", "integrity_argument"]].sum(axis=1)
    df["integrity_cap"] = np.maximum(0.0, 100.0 - 10.0 * df["integrity_count"])
    df["final_score"] = np.minimum(df["calibrated_cap"], df["integrity_cap"])
    y = df["accepted"].to_numpy(int)
    p = np.clip(df["final_score"].to_numpy(float) / 100.0, 1e-6, 1 - 1e-6)
    metrics = {
        "n": int(len(df)),
        "accepted": int(y.sum()),
        "rejected": int((1 - y).sum()),
        "score_bearing_categories": list(stable.keys()),
        "auroc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }
    inter, slope = calibration_slope_intercept(y, p)
    metrics["calibration_intercept"] = inter
    metrics["calibration_slope"] = slope
    metrics["bootstrap_95_ci"] = bootstrap_ci(y, p, args.bootstrap, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "acceptance_score_check.csv", index=False)
    (args.output_dir / "acceptance_score_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
