import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


DEFAULT_INPUT = Path("section_titles.txt")
DEFAULT_OUT_DIR = Path("section_title_clusters")

CANONICAL_PATTERNS = [
    ("background_introduction", r"\b(introduction|background|objective|objectives|aim|aims|overview)\b"),
    ("protocol_methods", r"\b(method|methods|methodology|materials and methods|protocol|registration|study design|research design)\b"),
    ("search_selection", r"\b(search|database|information sources|eligibility|inclusion|exclusion|selection criteria|study selection|screening|literature search)\b"),
    ("extraction_quality", r"\b(data extraction|data abstraction|data collection|data charting|quality assessment|risk of bias|bias assessment|methodological quality|quality appraisal|certainty|grade)\b"),
    ("analysis_synthesis", r"\b(statistical analysis|data analysis|data synthesis|meta analysis|meta analyses|heterogeneity|subgroup|sensitivity|publication bias|effect size)\b"),
    ("results_overview", r"\b(results|finding|findings|study characteristics|included studies|characteristics of included|search results|selection of studies|participants|outcomes)\b"),
    ("discussion_interpretation", r"\b(discussion|interpretation|implications|clinical implications|practical implications)\b"),
    ("limitations_strengths", r"\b(limitation|limitations|strengths|weaknesses)\b"),
    ("future_directions", r"\b(future|perspective|directions|recommendations|challenges|opportunities)\b"),
    ("conclusion", r"\b(conclusion|conclusions|summary|concluding remarks)\b"),
    ("domain_body_topic", r".*"),
]

NOISE_PATTERNS = [
    r"^\d+$",
    r"^(fig|figure|table|chart|box|supplement|appendix|scheme)\s*\d*\b",
    r"^\d+\s+\d+\s+[a-z].*",
    r"\b(open access|plos one|dovepress|publisher|springer nature|creative commons|license|copyright)\b",
    r"\b(author|authors|authorship|contributorship|contributions?|acknowledgments?|acknowledgements?|funding|funder|financial support|competing interests?|conflicts? of interests?|conflict of interests?)\b",
    r"\b(declaration|declarations|ethics statement|statement of ethics|ethical standards|ethics guidelines|ethical guidelines|ethics approval|ethical approval|ethics and dissemination|consent for publication|patient consent|human and animal rights)\b",
    r"\b(data availability|availability of data|data sharing|supplementary|supporting information|additional file)\b",
    r"\b(abbreviations|acronyms|correspondence|provenance|peer review|received|accepted|published)\b",
    r"\b(orcid|reviewer|biomedcentral|submissions)\b",
    r"^(low|medium|high|response|country|subgroup|yes|no|na|n a|unclear|not applicable|notes?|continued|comment|comments?|s\d+)$",
    r"^conflicts? of$",
    r"^[a-z]\s*[.-]?\s*[a-z]\s*[.-]?\s*$",
    r"^[a-z]\s+\d+$",
    r"^[a-z ]*et al\s+\d{4}.*",
    r"^flowchart\s*\d*$",
    r"\bunpublished\s+\d+\b",
]

NOISE_RE = [re.compile(pattern) for pattern in NOISE_PATTERNS]


def normalize_title(title: str) -> str:
    title = title.strip().lower()
    title = re.sub(r"[^\w\s+-]", " ", title, flags=re.ASCII)
    title = re.sub(r"[_\s]+", " ", title)
    return title.strip()


def token_count(title: str) -> int:
    return len(re.findall(r"[a-z0-9]+", title))


def noise_reason(title: str) -> str | None:
    if not title:
        return "empty"
    if len(title) < 2:
        return "too_short"
    words = token_count(title)
    if words == 0:
        return "no_words"
    if words > 18:
        return "too_long_for_section_title"
    if len(title) > 140:
        return "too_long_for_section_title"
    digit_chars = sum(ch.isdigit() for ch in title)
    alpha_chars = sum(ch.isalpha() for ch in title)
    if digit_chars > alpha_chars and words <= 8:
        return "mostly_numeric"
    for pattern in NOISE_RE:
        if pattern.search(title):
            return f"pattern:{pattern.pattern}"
    return None


def canonical_label(title: str) -> str:
    for label, pattern in CANONICAL_PATTERNS:
        if re.search(pattern, title):
            return label
    return "domain_body_topic"


def top_terms(vectorizer: TfidfVectorizer, centers: np.ndarray, n: int = 12) -> list[list[str]]:
    terms = np.asarray(vectorizer.get_feature_names_out())
    result = []
    for center in centers:
        result.append([str(term) for term in terms[np.argsort(center)[::-1][:n]]])
    return result


def choose_k(matrix, candidates: list[int], random_state: int) -> tuple[int, list[dict]]:
    scores = []
    sample_size = min(matrix.shape[0], 6000)
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(matrix.shape[0], size=sample_size, replace=False)
    sample = matrix[sample_idx]
    for k in candidates:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(matrix)
        score = silhouette_score(sample, labels[sample_idx], metric="cosine")
        scores.append({"k": k, "silhouette_cosine": float(score), "inertia": float(model.inertia_)})
    best = max(scores, key=lambda item: item["silhouette_cosine"])
    return int(best["k"]), scores


def cluster_titles(titles: list[str], k: int | None, random_state: int) -> dict:
    counts = Counter(titles)
    unique_titles = sorted(counts)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.35,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(unique_titles)
    lsa_dims = min(80, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    lsa = make_pipeline(TruncatedSVD(n_components=lsa_dims, random_state=random_state), Normalizer(copy=False))
    features = lsa.fit_transform(tfidf)
    if k is None:
        k, scores = choose_k(features, list(range(8, 19)), random_state)
    else:
        scores = []
    model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = model.fit_predict(features)

    raw_centers = np.zeros((k, tfidf.shape[1]))
    for cluster_id in range(k):
        members = np.where(labels == cluster_id)[0]
        raw_centers[cluster_id] = np.asarray(tfidf[members].mean(axis=0)).ravel()
    terms_by_cluster = top_terms(vectorizer, raw_centers)

    rows = []
    for title, cluster_id in zip(unique_titles, labels):
        rows.append(
            {
                "title": title,
                "count": counts[title],
                "cluster": int(cluster_id),
                "canonical_label": canonical_label(title),
            }
        )
    return {
        "rows": rows,
        "terms_by_cluster": terms_by_cluster,
        "chosen_k": k,
        "k_scores": scores,
    }


def summarize(rows: list[dict], terms_by_cluster: list[list[str]]) -> list[dict]:
    by_cluster = defaultdict(list)
    for row in rows:
        by_cluster[row["cluster"]].append(row)

    summary = []
    for cluster_id in sorted(by_cluster):
        members = by_cluster[cluster_id]
        label_counts = Counter()
        title_total = 0
        for row in members:
            label_counts[row["canonical_label"]] += row["count"]
            title_total += row["count"]
        examples = sorted(members, key=lambda item: (-item["count"], item["title"]))[:20]
        summary.append(
            {
                "cluster": cluster_id,
                "unique_titles": len(members),
                "title_occurrences": title_total,
                "dominant_function": label_counts.most_common(1)[0][0],
                "function_distribution": dict(label_counts.most_common()),
                "top_terms": terms_by_cluster[cluster_id],
                "examples": [{"title": item["title"], "count": item["count"]} for item in examples],
            }
        )
    return sorted(summary, key=lambda item: item["title_occurrences"], reverse=True)


def summarize_functions(rows: list[dict]) -> list[dict]:
    by_function = defaultdict(list)
    for row in rows:
        by_function[row["canonical_label"]].append(row)

    summary = []
    for label, members in by_function.items():
        title_total = sum(row["count"] for row in members)
        examples = sorted(members, key=lambda item: (-item["count"], item["title"]))[:30]
        summary.append(
            {
                "function": label,
                "unique_titles": len(members),
                "title_occurrences": title_total,
                "examples": [{"title": item["title"], "count": item["count"]} for item in examples],
            }
        )
    return sorted(summary, key=lambda item: item["title_occurrences"], reverse=True)


FUNCTION_DESCRIPTIONS = {
    "background_introduction": "Sets up scope, motivation, background, or review objectives.",
    "protocol_methods": "Describes the review protocol, design, registration, or general methods.",
    "search_selection": "Explains database search, eligibility criteria, inclusion/exclusion, and study screening.",
    "extraction_quality": "Describes extraction/charting, quality appraisal, certainty assessment, or risk-of-bias work.",
    "analysis_synthesis": "Describes statistical synthesis, meta-analysis, heterogeneity, subgroup, sensitivity, or publication-bias analysis.",
    "results_overview": "Reports included-study characteristics, search results, participants, outcomes, and main findings.",
    "domain_body_topic": "A substantive topical section within the review body, often organized by disease, method, mechanism, material, model, or application.",
    "discussion_interpretation": "Interprets findings and states practical, research, or clinical implications.",
    "limitations_strengths": "States limitations, strengths, weaknesses, or caveats.",
    "future_directions": "States challenges, opportunities, recommendations, perspectives, or future work.",
    "conclusion": "Closes the paper with a summary, conclusion, or concluding remarks.",
}


def write_taxonomy(path: Path, function_summary: list[dict], stats: dict) -> None:
    lines = [
        "# Section Function Taxonomy",
        "",
        f"Input titles: {stats['input_titles']}",
        f"Kept titles after noise filtering: {stats['kept_titles']}",
        f"Rejected as non-section-title noise: {stats['rejected_titles']}",
        f"Unique kept titles: {stats['unique_kept_titles']}",
        "",
        "Use these function labels to classify section headings in evaluated surveys. The unsupervised lexical clustering is kept in `cluster_summary.json`, but these function labels are the recommended target classes.",
        "",
    ]
    for item in function_summary:
        label = item["function"]
        lines.extend(
            [
                f"## {label}",
                "",
                FUNCTION_DESCRIPTIONS.get(label, ""),
                "",
                f"Occurrences: {item['title_occurrences']}; unique titles: {item['unique_titles']}.",
                "",
                "Typical titles:",
            ]
        )
        for example in item["examples"][:15]:
            lines.append(f"- {example['title']} ({example['count']})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["title", "count", "cluster", "canonical_label"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda item: (item["cluster"], -item["count"], item["title"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=13)
    args = parser.parse_args()

    raw_lines = args.input.read_text(encoding="utf-8").splitlines()
    normalized = [normalize_title(line) for line in raw_lines]
    kept = []
    rejected = []
    for original, title in zip(raw_lines, normalized):
        reason = noise_reason(title)
        if reason:
            rejected.append({"original": original, "normalized": title, "reason": reason})
        else:
            kept.append(title)

    clustering = cluster_titles(kept, args.k, args.random_state)
    summary = summarize(clustering["rows"], clustering["terms_by_cluster"])
    function_summary = summarize_functions(clustering["rows"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "clean_titles.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
    write_csv(args.out_dir / "title_clusters.csv", clustering["rows"])
    stats = {
        "input_titles": len(raw_lines),
        "kept_titles": len(kept),
        "rejected_titles": len(rejected),
        "unique_kept_titles": len(set(kept)),
        "chosen_k": clustering["chosen_k"],
    }
    (args.out_dir / "cluster_summary.json").write_text(
        json.dumps(
            {
                **stats,
                "k_scores": clustering["k_scores"],
                "function_summary": function_summary,
                "clusters": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.out_dir / "rejected_titles.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in rejected) + "\n",
        encoding="utf-8",
    )
    write_taxonomy(args.out_dir / "section_function_taxonomy.md", function_summary, stats)

    print(f"input_titles={len(raw_lines)}")
    print(f"kept_titles={len(kept)}")
    print(f"rejected_titles={len(rejected)}")
    print(f"unique_kept_titles={len(set(kept))}")
    print(f"chosen_k={clustering['chosen_k']}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
