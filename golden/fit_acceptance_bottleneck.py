from __future__ import annotations

"""Fit the acceptance-calibrated adequacy bottleneck without literature-pool calls.

The golden corpus has binary TMLR decisions for most papers but not a cardinal
review score. This script therefore uses only local paper skeletons and
review-analysis labels. It deliberately avoids the full evaluation agent:
constructing literature pools for 150 papers would consume far more external
API quota than is justified for a calibration pass.
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold


INTEGRITY_FILES = {
    "internal_inconsistency": "internal_inconsistency.jsonl",
    "factual_hallucination": "factual_hallucination_or_technical_error.jsonl",
    "taxonomy_partition": "taxonomy_framework_problem.jsonl",
    "argument_support": "evidence_support_insufficient.jsonl",
}

CONTENT_HEADING = re.compile(
    r"\b(introduction|related work|background|preliminar|method|methodology|"
    r"search strategy|literature search|conclusion|discussion|future work|"
    r"limitation|references|appendix|acknowledg)\b",
    re.IGNORECASE,
)
SCOPE_PATTERN = re.compile(
    r"\b(we (survey|review|cover|focus on|include|exclude|consider)|"
    r"inclusion|exclusion|search (strategy|database|query)|"
    r"systematic review|literature (search|selection)|"
    r"published (between|from|since)|time (span|range)|"
    r"out of scope|beyond the scope)\b",
    re.IGNORECASE,
)
GAP_PATTERN = re.compile(
    r"\b(future work|future research|open (problem|question|challenge)|"
    r"remains? (open|unclear|underexplored|unresolved)|"
    r"limitation|limitations|research gap|promising direction|"
    r"needs? further (study|investigation|research))\b",
    re.IGNORECASE,
)
COMPARISON_PATTERN = re.compile(
    r"\b(compared (with|to)|in contrast|whereas|while .*? (is|are)|"
    r"unlike|differ(s|ed|ent)? from|outperform(s|ed)?|"
    r"advantage|disadvantage|trade-?off|however|on the other hand)\b",
    re.IGNORECASE,
)
SYNTHESIS_PATTERN = re.compile(
    r"\b(overall|collectively|taken together|in summary|"
    r"we (observe|find|conclude|identify)|the literature (shows|suggests|reveals)|"
    r"a common (pattern|theme|trend)|across (these|the) (studies|works|methods)|"
    r"this suggests|these findings)\b",
    re.IGNORECASE,
)
CONTRIBUTION_PATTERN = re.compile(
    r"\b(we (provide|present|propose|offer|contribute|introduce)|"
    r"this (survey|review|work|paper) (provides|presents|offers|contributes)|"
    r"our (contribution|contributions))\b",
    re.IGNORECASE,
)
NOVELTY_BASIS_PATTERN = re.compile(
    r"\b(unlike|previous (surveys|reviews|work)|prior (surveys|reviews|work)|"
    r"existing (surveys|reviews)|to the best of our knowledge|"
    r"recent advances|since \d{4})\b",
    re.IGNORECASE,
)
STRONG_ARGUMENT_PATTERN = re.compile(
    r"\b(always|never|all|none|clearly|demonstrates?|proves?|"
    r"significantly|substantially|therefore|thus|hence|consequently)\b",
    re.IGNORECASE,
)
TAXONOMY_TERMS = re.compile(
    r"\b(taxonom|categor|classif|framework|overlap|sibling|axis|hierarch)\b",
    re.IGNORECASE,
)


@dataclass
class PaperView:
    paper_id: int
    forum_id: str
    title: str
    venue: str
    venue_id: str
    accepted: int | None
    paper: dict[str, Any]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_sections(sections: Iterable[dict[str, Any]], depth: int = 1):
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        yield section, depth
        yield from iter_sections(section.get("sections") or [], depth + 1)


def section_sentences(section: dict[str, Any]) -> list[dict[str, Any]]:
    sentences = []
    for paragraph in section.get("paragraphs") or []:
        for sentence in paragraph or []:
            if isinstance(sentence, dict) and sentence.get("environment_type", "text") == "text":
                text = str(sentence.get("text") or "").strip()
                if text:
                    sentences.append(sentence)
    return sentences


def all_sentences(paper: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    abstract = paper.get("abstract") or {}
    output.extend(section_sentences(abstract))
    for section, _ in iter_sections(paper.get("sections") or []):
        output.extend(section_sentences(section))
    return output


def unique_citation_keys(sentences: Iterable[dict[str, Any]]) -> set[str]:
    keys = set()
    for sentence in sentences:
        citations = sentence.get("citations") or []
        if isinstance(citations, dict):
            citations = citations.values()
        for citation in citations:
            if isinstance(citation, dict):
                key = citation.get("key") or citation.get("title") or citation.get("xml_id")
            else:
                key = str(citation)
            if key:
                keys.add(str(key))
    return keys


def acceptance_label(record: dict[str, Any]) -> int | None:
    venue = str(record.get("venue") or "")
    venue_id = str(record.get("venue_id") or "")
    if venue == "Accepted by TMLR" or venue_id == "TMLR":
        return 1
    if venue == "Rejected by TMLR" or venue_id == "TMLR/Rejected":
        return 0
    return None


def load_papers(pdf_content: Path) -> list[PaperView]:
    papers = []
    for path in sorted(pdf_content.glob("*.json")):
        record = read_json(path)
        match = re.match(r"^(\d+)_", path.name)
        if not match:
            continue
        papers.append(
            PaperView(
                paper_id=int(match.group(1)),
                forum_id=str(record.get("openreview_forum_id") or ""),
                title=str(record.get("paper_title") or ""),
                venue=str(record.get("venue") or ""),
                venue_id=str(record.get("venue_id") or ""),
                accepted=acceptance_label(record),
                paper=record.get("paper") or {},
            )
        )
    return papers


def taxonomy_record_is_structural(record: dict[str, Any]) -> bool:
    evidence = str(record.get("evidence") or "")
    return bool(TAXONOMY_TERMS.search(evidence))


def load_integrity_counts(review_dir: Path) -> dict[int, dict[str, int]]:
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for feature, filename in INTEGRITY_FILES.items():
        path = review_dir / filename
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if feature == "taxonomy_partition" and not taxonomy_record_is_structural(record):
                continue
            paper_id = int(record["paper_id"])
            counts[paper_id][feature] += 1
    return counts


def ratio(numerator: int | float, denominator: int | float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else default


def raw_features(view: PaperView, integrity: dict[str, int]) -> dict[str, float]:
    paper = view.paper
    sections = list(iter_sections(paper.get("sections") or []))
    top_level = [section for section, depth in sections if depth == 1]
    content_sections = [
        section
        for section in top_level
        if not CONTENT_HEADING.search(str(section.get("title") or ""))
    ]
    if not content_sections:
        content_sections = top_level

    all_text_sentences = all_sentences(paper)
    document_text = " ".join(str(s.get("text") or "") for s in all_text_sentences)
    section_stats = []
    for section in content_sections:
        sentences = section_sentences(section)
        text = " ".join(str(s.get("text") or "") for s in sentences)
        citations = unique_citation_keys(sentences)
        section_stats.append(
            {
                "sentences": sentences,
                "text": text,
                "citations": citations,
                "gap": sum(bool(GAP_PATTERN.search(str(s.get("text") or ""))) for s in sentences),
                "contrast": sum(bool(COMPARISON_PATTERN.search(str(s.get("text") or ""))) for s in sentences),
                "synthesis": sum(bool(SYNTHESIS_PATTERN.search(str(s.get("text") or ""))) for s in sentences),
            }
        )

    total_sentences = len(all_text_sentences)
    content_sentence_count = sum(len(item["sentences"]) for item in section_stats)
    total_citations = unique_citation_keys(all_text_sentences)
    content_citations = set().union(*(item["citations"] for item in section_stats)) if section_stats else set()
    n_sections = max(1, len(section_stats))

    contribution_context = " ".join(
        str(s.get("text") or "")
        for s in all_text_sentences[: min(len(all_text_sentences), 80)]
    )
    contribution_sentences = [
        str(s.get("text") or "")
        for s in all_text_sentences
        if CONTRIBUTION_PATTERN.search(str(s.get("text") or ""))
    ]
    contribution_basis = any(NOVELTY_BASIS_PATTERN.search(text) for text in contribution_sentences) or bool(
        NOVELTY_BASIS_PATTERN.search(contribution_context)
    )

    gap_sections = sum(item["gap"] > 0 for item in section_stats)
    contrast_sections = sum(item["contrast"] > 0 for item in section_stats)
    synthesis_sections = sum(item["synthesis"] > 0 for item in section_stats)
    gap_count = sum(item["gap"] for item in section_stats)
    contrast_count = sum(item["contrast"] for item in section_stats)
    synthesis_count = sum(item["synthesis"] for item in section_stats)
    summary_count = sum(
        bool(re.search(r"\b(describe|summari[sz]e|review|introduce|present)\b", str(s.get("text") or ""), re.I))
        for s in all_text_sentences
    )

    return {
        "scope_existence": float(bool(SCOPE_PATTERN.search(document_text))),
        "gap_section_coverage": ratio(gap_sections, n_sections),
        "gap_volume_per_section": ratio(gap_count, n_sections),
        "contrast_section_coverage": ratio(contrast_sections, n_sections),
        "contrast_volume_per_section": ratio(contrast_count, n_sections),
        "synthesis_section_coverage": ratio(synthesis_sections, n_sections),
        "synthesis_volume_per_section": ratio(synthesis_count, n_sections),
        "synthesis_to_summary_ratio": ratio(synthesis_count, synthesis_count + summary_count),
        "contribution_grounding": float(bool(contribution_sentences) and contribution_basis),
        "citation_density": ratio(len(total_citations), total_sentences),
        "citation_section_coverage": ratio(
            sum(bool(item["citations"]) for item in section_stats), n_sections
        ),
        "topic_breadth": float(len(section_stats)),
        "content_sentence_count": float(content_sentence_count),
        "integrity_internal": float(integrity.get("internal_inconsistency", 0)),
        "integrity_factual": float(integrity.get("factual_hallucination", 0)),
        "integrity_taxonomy": float(integrity.get("taxonomy_partition", 0)),
        "integrity_argument": float(integrity.get("argument_support", 0)),
    }


def empirical_cdf(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(train)
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def category_indices(train: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=target.index)
    output["scope"] = target["scope_existence"].astype(float)
    for category, components in {
        "gap": ["gap_section_coverage", "gap_volume_per_section"],
        "contrast": ["contrast_section_coverage", "contrast_volume_per_section"],
        "synthesis": ["synthesis_section_coverage", "synthesis_volume_per_section"],
        "contribution": ["contribution_grounding"],
        "reference_proxy": ["citation_density", "citation_section_coverage"],
        "topic_proxy": ["topic_breadth"],
    }.items():
        ranks = [
            empirical_cdf(train[column].to_numpy(float), target[column].to_numpy(float))
            for column in components
        ]
        output[category] = np.min(np.column_stack(ranks), axis=1)
    return output


def expit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


def firth_logistic_fit(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 200,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Bias-reduced logistic regression using the adjusted-score iteration."""

    beta = np.zeros(X.shape[1], dtype=float)
    for _ in range(max_iter):
        eta = X @ beta
        mu = expit(eta)
        weights = np.clip(mu * (1.0 - mu), 1e-8, None)
        information = X.T @ (weights[:, None] * X)
        information += np.eye(X.shape[1]) * 1e-8
        information_inv = np.linalg.inv(information)
        hat_diag = weights * np.einsum("ij,jk,ik->i", X, information_inv, X)
        adjusted_score = X.T @ (y - mu + hat_diag * (0.5 - mu))
        step = information_inv @ adjusted_score
        if np.max(np.abs(step)) < tolerance:
            return beta

        current_objective = float(
            np.sum(y * eta - np.logaddexp(0.0, eta)) + 0.5 * np.linalg.slogdet(information)[1]
        )
        scale = 1.0
        while scale >= 1e-6:
            candidate = beta + scale * step
            candidate_eta = X @ candidate
            candidate_mu = expit(candidate_eta)
            candidate_weights = np.clip(candidate_mu * (1.0 - candidate_mu), 1e-8, None)
            candidate_info = X.T @ (candidate_weights[:, None] * X) + np.eye(X.shape[1]) * 1e-8
            candidate_objective = float(
                np.sum(y * candidate_eta - np.logaddexp(0.0, candidate_eta))
                + 0.5 * np.linalg.slogdet(candidate_info)[1]
            )
            if candidate_objective >= current_objective:
                beta = candidate
                break
            scale *= 0.5
        else:
            return beta
    return beta


def calibration_slope_intercept(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1))
    X = np.column_stack([np.ones_like(logits), logits])
    beta = firth_logistic_fit(X, y)
    return float(beta[0]), float(beta[1])


def bootstrap_ci(
    y: np.ndarray,
    probabilities: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    metrics: dict[str, list[float]] = defaultdict(list)
    n = len(y)
    for _ in range(repetitions):
        for _attempt in range(100):
            indices = rng.integers(0, n, n)
            sample_y = y[indices]
            if len(np.unique(sample_y)) == 2:
                break
        else:
            continue
        sample_p = probabilities[indices]
        metrics["auroc"].append(roc_auc_score(sample_y, sample_p))
        metrics["pr_auc"].append(average_precision_score(sample_y, sample_p))
        metrics["brier"].append(brier_score_loss(sample_y, sample_p))
        intercept, slope = calibration_slope_intercept(sample_y, sample_p)
        metrics["calibration_intercept"].append(intercept)
        metrics["calibration_slope"].append(slope)
    return {
        name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for name, values in metrics.items()
    }


def evaluate(df: pd.DataFrame, repeats: int, folds: int, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    y = df["accepted"].to_numpy(int)
    predictions = np.full(len(df), np.nan)
    fold_rows = []
    splitter = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=repeats,
        random_state=seed,
    )

    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, y), start=1):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        train_indices = category_indices(train, train)
        test_indices = category_indices(train, test)
        train_b = train_indices.min(axis=1).to_numpy(float)
        test_b = test_indices.min(axis=1).to_numpy(float)
        integrity_columns = [
            "integrity_internal",
            "integrity_factual",
            "integrity_taxonomy",
            "integrity_argument",
        ]
        X_train = np.column_stack(
            [
                np.ones(len(train)),
                train_b,
                train[integrity_columns].to_numpy(float),
            ]
        )
        X_test = np.column_stack(
            [
                np.ones(len(test)),
                test_b,
                test[integrity_columns].to_numpy(float),
            ]
        )
        beta = firth_logistic_fit(X_train, train["accepted"].to_numpy(int))
        probabilities = expit(X_test @ beta)
        predictions[test_idx] = np.nan_to_num(predictions[test_idx], nan=0.0) + probabilities
        fold_rows.append(
            {
                "fold": fold,
                "beta_intercept": float(beta[0]),
                "beta_bottleneck": float(beta[1]),
                "accepted_test": int(test["accepted"].sum()),
                "n_test": int(len(test)),
            }
        )

    predictions /= repeats
    result = df[["paper_id", "forum_id", "title", "accepted"]].copy()
    result["predicted_acceptance"] = predictions
    result["predicted_acceptability_score"] = 100.0 * predictions
    metrics = {
        "n": int(len(df)),
        "accepted": int(y.sum()),
        "rejected": int((1 - y).sum()),
        "auroc": float(roc_auc_score(y, predictions)),
        "pr_auc": float(average_precision_score(y, predictions)),
        "brier": float(brier_score_loss(y, predictions)),
    }
    intercept, slope = calibration_slope_intercept(y, predictions)
    metrics["calibration_intercept"] = intercept
    metrics["calibration_slope"] = slope
    metrics["bootstrap_95_ci"] = bootstrap_ci(y, predictions, repetitions=1000, seed=seed + 19)
    metrics["folds"] = fold_rows
    return result, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-content", type=Path, default=Path(__file__).parent / "pdf_content")
    parser.add_argument("--review-analysis", type=Path, default=Path(__file__).parent / "review_analyze")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "acceptance_fit")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    integrity = load_integrity_counts(args.review_analysis)
    all_papers = load_papers(args.pdf_content)
    rows = []
    manifest = []
    for view in all_papers:
        row = {
            "paper_id": view.paper_id,
            "forum_id": view.forum_id,
            "title": view.title,
            "venue": view.venue,
            "venue_id": view.venue_id,
            "accepted": view.accepted,
            **raw_features(view, integrity.get(view.paper_id, {})),
        }
        manifest.append(row)
        if view.accepted is not None:
            rows.append(row)

    df = pd.DataFrame(rows).sort_values("paper_id").reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).sort_values("paper_id").to_csv(args.output_dir / "manifest.csv", index=False)
    df.to_csv(args.output_dir / "training_features.csv", index=False)
    predictions, metrics = evaluate(df, args.repeats, args.folds, args.seed)
    predictions.to_csv(args.output_dir / "cross_validated_predictions.csv", index=False)
    (args.output_dir / "fit_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: metrics[key] for key in metrics if key != "folds"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
