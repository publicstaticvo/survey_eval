from __future__ import annotations

"""Estimate observed-concern thresholds for adequacy indicators.

Thresholds are fitted per reviewer-concern category, not against editorial
acceptance.  A missing review comment is treated as unlabeled rather than proof
of adequacy; reported precision is therefore an observed-positive lower bound.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from survey_eval.golden.fit_acceptance_bottleneck import (
    CONTRIBUTION_PATTERN,
    all_sentences,
    iter_sections,
    load_papers,
    raw_features,
)


LABEL_FILES = {
    "gap": ["future_work_or_limitation_missing.jsonl"],
    "contrast": ["comparison_analysis_insufficient.jsonl"],
    "synthesis": ["synthesis_depth_insufficient.jsonl"],
    "scope": ["methodology_transparency_insufficient.jsonl"],
    "contribution": ["contribution_novelty_insufficient.jsonl"],
    "reference": ["missing_specific_references_labeled.jsonl", "references_insufficient.jsonl"],
    "topic": ["missing_specific_topics_labeled.jsonl", "coverage_insufficient.jsonl"],
}


def load_positive_papers(review_dir: Path) -> dict[str, set[int]]:
    labels: dict[str, set[int]] = defaultdict(set)
    for category, filenames in LABEL_FILES.items():
        for filename in filenames:
            path = review_dir / filename
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    labels[category].add(int(json.loads(line)["paper_id"]))
    return labels


def citation_count_per_content_topic(view: Any) -> tuple[float, float]:
    sections = list(iter_sections(view.paper.get("sections") or []))
    content = [
        section
        for section, depth in sections
        if depth == 1 and not re.search(
            r"\b(introduction|related work|background|preliminar|method|methodology|"
            r"search strategy|literature search|conclusion|discussion|future work|"
            r"limitation|references|appendix|acknowledg)\b",
            str(section.get("title") or ""),
            re.I,
        )
    ]
    if not content:
        content = [section for section, depth in sections if depth == 1]
    cited_topics = 0
    cited_keys: set[str] = set()
    for section in content:
        local = set()
        for paragraph in section.get("paragraphs") or []:
            for sentence in paragraph or []:
                for citation in sentence.get("citations") or []:
                    if isinstance(citation, dict):
                        key = citation.get("key") or citation.get("xml_id") or citation.get("title")
                    else:
                        key = citation
                    if key:
                        local.add(str(key))
        cited_topics += bool(local)
        cited_keys.update(local)
    n_topics = max(1, len(content))
    return cited_topics / n_topics, len(cited_keys) / n_topics


def build_features(pdf_content: Path) -> pd.DataFrame:
    rows = []
    for view in load_papers(pdf_content):
        raw = raw_features(view, {})
        citation_topic_coverage, citations_per_topic = citation_count_per_content_topic(view)
        contribution_exists = float(
            any(CONTRIBUTION_PATTERN.search(str(sentence.get("text") or "")) for sentence in all_sentences(view.paper))
        )
        rows.append(
            {
                "paper_id": view.paper_id,
                "title": view.title,
                "gap_coverage": raw["gap_section_coverage"],
                "gap_volume": raw["gap_volume_per_section"],
                "contrast_coverage": raw["contrast_section_coverage"],
                "contrast_volume": raw["contrast_volume_per_section"],
                "synthesis_coverage": raw["synthesis_section_coverage"],
                "synthesis_volume": raw["synthesis_volume_per_section"],
                "scope_exists": raw["scope_existence"],
                "contribution_exists": contribution_exists,
                "reference_topic_coverage": citation_topic_coverage,
                "citations_per_topic": citations_per_topic,
                "topic_count": raw["topic_breadth"],
            }
        )
    return pd.DataFrame(rows)


def threshold_grid(values: np.ndarray) -> np.ndarray:
    return np.unique(np.quantile(values, np.linspace(0.0, 1.0, 41)))


def fit_single_threshold(values: np.ndarray, y: np.ndarray, lower_is_risk: bool) -> dict[str, float]:
    best: dict[str, float] | None = None
    for threshold in threshold_grid(values):
        predicted = values < threshold if lower_is_risk else values > threshold
        precision, recall, fbeta, _ = precision_recall_fscore_support(
            y,
            predicted,
            beta=0.5,
            average="binary",
            zero_division=0,
        )
        candidate = {
            "threshold": float(threshold),
            "observed_precision_lb": float(precision),
            "observed_recall": float(recall),
            "f0_5": float(fbeta),
        }
        if best is None or (candidate["f0_5"], candidate["observed_precision_lb"]) > (
            best["f0_5"],
            best["observed_precision_lb"],
        ):
            best = candidate
    assert best is not None
    return best


def fit_pair_thresholds(coverage: np.ndarray, volume: np.ndarray, y: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    for coverage_threshold in threshold_grid(coverage):
        for volume_threshold in threshold_grid(volume):
            predicted = (coverage < coverage_threshold) | (volume < volume_threshold)
            precision, recall, fbeta, _ = precision_recall_fscore_support(
                y,
                predicted,
                beta=0.5,
                average="binary",
                zero_division=0,
            )
            candidate = {
                "coverage_threshold": float(coverage_threshold),
                "volume_threshold": float(volume_threshold),
                "observed_precision_lb": float(precision),
                "observed_recall": float(recall),
                "f0_5": float(fbeta),
            }
            if best is None or (candidate["f0_5"], candidate["observed_precision_lb"]) > (
                best["f0_5"],
                best["observed_precision_lb"],
            ):
                best = candidate
    assert best is not None
    return best


def cross_validate(
    features: pd.DataFrame,
    y: np.ndarray,
    fit_fn: Callable[..., dict[str, float]],
    columns: list[str],
    seed: int,
) -> dict[str, Any]:
    folds = min(5, int(y.sum()), int((1 - y).sum()))
    if folds < 2:
        return {"folds": 0}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    all_predictions = np.zeros(len(features), dtype=bool)
    parameters = []
    for train, test in splitter.split(features, y):
        values = [features.iloc[train][column].to_numpy(float) for column in columns]
        fitted = fit_fn(*values, y[train])
        test_values = [features.iloc[test][column].to_numpy(float) for column in columns]
        if len(columns) == 1:
            predicted = test_values[0] < fitted["threshold"]
        else:
            predicted = (test_values[0] < fitted["coverage_threshold"]) | (
                test_values[1] < fitted["volume_threshold"]
            )
        all_predictions[test] = predicted
        parameters.append(fitted)
    precision, recall, fbeta, _ = precision_recall_fscore_support(
        y,
        all_predictions,
        beta=0.5,
        average="binary",
        zero_division=0,
    )
    return {
        "folds": folds,
        "observed_precision_lb": float(precision),
        "observed_recall": float(recall),
        "f0_5": float(fbeta),
        "thresholds_by_fold": parameters,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-content", type=Path, default=Path(__file__).parent / "pdf_content")
    parser.add_argument("--review-analysis", type=Path, default=Path(__file__).parent / "review_analyze")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "adequacy_thresholds")
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    features = build_features(args.pdf_content)
    labels = load_positive_papers(args.review_analysis)
    specs = {
        "gap": ("pair", ["gap_coverage", "gap_volume"]),
        "contrast": ("pair", ["contrast_coverage", "contrast_volume"]),
        "synthesis": ("pair", ["synthesis_coverage", "synthesis_volume"]),
        "scope": ("single", ["scope_exists"]),
        "contribution": ("single", ["contribution_exists"]),
        "reference": ("pair", ["reference_topic_coverage", "citations_per_topic"]),
        "topic": ("single", ["topic_count"]),
    }
    output: dict[str, Any] = {"method": "per-category observed-concern threshold estimation", "categories": {}}
    for category, (kind, columns) in specs.items():
        y = features["paper_id"].isin(labels[category]).to_numpy(int)
        if kind == "pair":
            fitted = fit_pair_thresholds(*(features[column].to_numpy(float) for column in columns), y)
            validation = cross_validate(features, y, fit_pair_thresholds, columns, args.seed)
        else:
            fitted = fit_single_threshold(features[columns[0]].to_numpy(float), y, lower_is_risk=True)
            validation = cross_validate(
                features,
                y,
                lambda values, labels: fit_single_threshold(values, labels, lower_is_risk=True),
                columns,
                args.seed,
            )
        output["categories"][category] = {
            "positive_papers": int(y.sum()),
            "features": columns,
            "full_fit": fitted,
            "cross_validation": validation,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_dir / "features.csv", index=False)
    (args.output_dir / "thresholds.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
