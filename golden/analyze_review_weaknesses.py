from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_JSON = ROOT / "review.json"
OUTPUT_JSON = ROOT / "review_weaknesses_by_forum.json"
OUTPUT_MD = ROOT / "review_weakness_clusters.md"


NEGATIVE_FIELDS = [
    "weaknesses",
    "strengths_and_weaknesses",
    "requested_changes",
    "reasons_to_reject",
    "Reasons_to_reject",
    "questions",
    "Questions_for_the_Authors",
    "questions_to_authors",
    "additional_feedback",
    "additional_comments",
    "review",
    "metareview",
    "Relecture",
]


CLUSTERS = {
    "漏引用具体文献": {
        "source": "A",
        "description": "审稿人明确给出综述缺少的具体论文或作品标题，判断标准来自已有文献池。",
    },
    "未覆盖具体主题": {
        "source": "A",
        "description": "审稿人明确指出综述遗漏了某个具体 topic、任务、应用场景或子领域，判断标准来自已有文献池。",
    },
    "内容过时或未纳入近期进展": {
        "source": "A",
        "description": "审稿人认为综述与领域当前文献状态脱节、引用不是最新版，或未覆盖近期进展。",
    },
    "系统性与文献筛选方法不足": {
        "source": "B",
        "description": "检索策略、纳入排除标准、系统综述流程、复现性或文献池构建不够清楚。",
    },
    "覆盖浅显或深度不足": {
        "source": "B",
        "description": "综述停留在表层描述，关键技术、方法细节、实验设置或应用语境展开不够。",
    },
    "缺少批判性综合与洞见": {
        "source": "B",
        "description": "只是罗列已有工作，缺少比较、归纳、批判性评价、设计取舍或面向未来的实质洞见。",
    },
    "缺少比较证据或量化分析": {
        "source": "B",
        "description": "缺少 benchmark、表格、指标、数据集统计、复杂度分析、横向比较或实证支撑。",
    },
    "写作表达、格式或图表问题": {
        "source": "B",
        "description": "文字不清、排版错误、引用占位符、图表质量差、可视化不足、拼写或格式问题。",
    },
    "定义、术语或分类体系不清": {
        "source": "C",
        "description": "综述内部的核心定义、术语边界、taxonomy 或符号使用不清楚，影响自身论证一致性。",
    },
    "结构组织与章节衔接问题": {
        "source": "C",
        "description": "章节顺序、段落衔接、内容分配、标题命名或前后逻辑组织存在内部问题。",
    },
    "范围、标题或目标不一致": {
        "source": "C",
        "description": "标题、摘要、目标声明与正文实际覆盖范围不一致，或综述边界没有自洽界定。",
    },
    "内部论证、结论或建议支撑不足": {
        "source": "C",
        "description": "结论、建议或立场没有被综述自身的材料充分支撑，或前后论证链条断裂。",
    },
    "个人偏好、venue匹配或其他琐碎问题": {
        "source": "D",
        "description": "包含审稿人对 venue、创新性门槛、写作取向的个人偏好，以及无法稳定归入其他类别的零散意见。",
    },
}


RULES = {
    "内容过时或未纳入近期进展": [
        "outdated",
        "not up to date",
        "newest version",
        "latest work",
        "latest works",
        "recent work",
        "recent works",
        "current standard",
        "contemporary standard",
        "up to date",
        "published in conference proceedings",
    ],
    "系统性与文献筛选方法不足": [
        "systematic",
        "search strategy",
        "inclusion criteria",
        "exclusion criteria",
        "selection criteria",
        "screening",
        "literature search",
        "paper collection",
        "methodology",
        "reproducibility",
        "replicable",
        "comprehensive research",
        "not comprehensive",
    ],
    "覆盖浅显或深度不足": [
        "superficial",
        "too brief",
        "single paragraph",
        "lacks sufficient detail",
        "not enough detail",
        "more detailed",
        "in-depth",
        "depth",
        "high-level",
        "straightforward summary",
        "does not discuss",
        "not discussed",
        "not fully",
        "fails to address",
        "limited discussion",
        "briefly",
        "more explanation",
    ],
    "缺少批判性综合与洞见": [
        "critical review",
        "critical analysis",
        "critical evaluation",
        "new insights",
        "meaningful contribution",
        "novel insights",
        "synthesize",
        "synthesis",
        "insight",
        "take a stance",
        "actionable",
        "future directions",
        "recommendations",
        "merely lists",
        "listing",
        "taxonomy is not",
    ],
    "缺少比较证据或量化分析": [
        "comparison",
        "comparative",
        "benchmark",
        "benchmarking",
        "evaluation",
        "metric",
        "quantitative",
        "statistics",
        "dataset size",
        "complexity",
        "ablation",
        "empirical",
        "table",
        "compare",
        "baseline",
    ],
    "写作表达、格式或图表问题": [
        "writing",
        "presentation",
        "unclear",
        "not clear",
        "confusing",
        "typo",
        "grammar",
        "format",
        "citation",
        "wrongly cited",
        "vague citations",
        "question marks",
        "\"?\"",
        "figure",
        "visualization",
        "diagram",
        "caption",
        "readability",
        "hard to read",
        "poorly written",
        "redundant",
    ],
    "定义、术语或分类体系不清": [
        "definition",
        "terminology",
        "taxonomy",
        "categorization",
        "classification",
        "what is",
        "define",
        "not accurate",
        "concept",
        "term",
        "notation",
        "distinction",
        "boundary",
    ],
    "结构组织与章节衔接问题": [
        "structure",
        "organization",
        "organisation",
        "organized",
        "section",
        "subsection",
        "paragraph",
        "flow",
        "connection",
        "connected",
        "bridge",
        "order",
        "layout",
        "title of section",
        "move",
    ],
    "范围、标题或目标不一致": [
        "scope",
        "title claims",
        "paper title",
        "objective",
        "goal",
        "target",
        "focus",
        "too broad",
        "too narrow",
        "does not fit",
        "purports",
        "claims",
        "range of topics",
    ],
    "内部论证、结论或建议支撑不足": [
        "not supported",
        "disconnect",
        "does not follow",
        "stem from the findings",
        "motivated by first principles",
        "without using the results",
        "bridging",
        "support the definition",
        "conclusion",
        "argument",
        "evidence",
        "justified",
        "valid",
    ],
    "个人偏好、venue匹配或其他琐碎问题": [
        "not the right venue",
        "venue",
        "conference",
        "journal",
        "novelty",
        "technical novelty",
        "original research",
        "blog post",
        "position paper",
        "i would recommend",
        "i suggest",
        "minor",
        "nit",
        "preference",
        "not appropriate for",
    ],
}


REFERENCE_CONTEXT = re.compile(
    r"(?i)(missing references?|references? (?:are )?missing|should cite|should include|"
    r"not cite|did not cite|additional papers?|potential additional papers?|"
    r"add missing works|missing works|include .*related works?|discussing related works?)"
)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_strengths(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"(?is)\*\*\s*weaknesses?\s*:\s*\*\*|weaknesses?\s*:", text, maxsplit=1)
    if len(parts) == 2:
        return parts[1]
    text = re.sub(r"(?is)\*\*\s*strengths?\s*:\s*\*\*.*?(?=\*\*\s*weaknesses?\s*:\s*\*\*)", "", text)
    return text


def negative_text(review: dict) -> str:
    chunks = []
    for field in NEGATIVE_FIELDS:
        value = review.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        if field == "strengths_and_weaknesses":
            value = strip_strengths(value)
        chunks.append(f"[{field}] {value.strip()}")
    return "\n\n".join(chunks)


def has_any(text: str, needles: list[str]) -> bool:
    lower = text.lower()
    return any(needle in lower for needle in needles)


def classify_text(text: str) -> list[str]:
    labels = [label for label, needles in RULES.items() if has_any(text, needles)]
    return list(dict.fromkeys(labels))


def clean_title(candidate: str) -> str | None:
    candidate = normalize_space(candidate)
    candidate = re.sub(r"^\[[0-9]+\]\s*", "", candidate)
    candidate = re.sub(r"\s+(?:arXiv preprint|Proceedings of|Transactions on|The \w+ International Conference|[0-9]{4}\b).*$", "", candidate)
    candidate = candidate.strip(" .;:,()[]")
    if len(candidate) < 8 or len(candidate) > 180:
        return None
    if "?" in candidate:
        return None
    if candidate.lower().startswith(("http", "section", "line", "recommendation")):
        return None
    if re.search(
        r"(?i)(conference|journal|survey track|call for|doi|arxiv\.org|openreview|dl\.acm|https?|"
        r"we did not|accounting for|recommendation|related work|limitations|section|figure|table|"
        r"et al$|^[A-Z][a-z]+,\s+[A-Z][a-z]+|^as seen\b|^in\s+[?,]|many followup|cookbook|"
        r"semantic mapping in\b|^at training$)",
        candidate,
    ):
        return None
    if len(candidate.split()) < 3:
        return None
    if not (":" in candidate or re.search(r"\b(of|for|with|through|from|against|under|based|learning|survey|model|models|reinforcement|generation|constraints?|method|methods|analysis|framework|networks?)\b", candidate, re.I)):
        return None
    if not re.search(r"[a-z]{3,}", candidate):
        return None
    if sum(ch.isalpha() for ch in candidate) < 6:
        return None
    return candidate


def extract_reference_titles(text: str) -> list[str]:
    if not REFERENCE_CONTEXT.search(text):
        return []

    titles: list[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        is_context_line = bool(REFERENCE_CONTEXT.search(line))
        is_bibliography_line = bool(re.match(r"^\s*(?:[-*]\s*)?\[[0-9]+\]\s+", line))
        if not is_context_line and not is_bibliography_line:
            continue

        quoted_titles = []
        for quoted in re.findall(r'"([^"]{8,180})"', line):
            title = clean_title(quoted)
            if title:
                quoted_titles.append(title)
        titles.extend(quoted_titles)

        bracket_title = re.match(
            r"^\s*[-*]?\s*\[[0-9]+\]\s*(?:.+?\([0-9]{4}\)\.?\s*)?(.+?)(?:\.\s+(?:Proceedings|arXiv|The|Transactions)|$)",
            line,
        )
        if bracket_title and not quoted_titles:
            title = clean_title(bracket_title.group(1))
            if title:
                titles.append(title)

    seen = set()
    result = []
    for title in titles:
        key = title.casefold()
        if key not in seen:
            seen.add(key)
            result.append(title)
    return result


def extract_missing_topics(text: str) -> list[str]:
    topics: list[str] = []
    patterns = [
        r"(?i)(?:missing|omits?|does not cover|fails to cover|should cover|should discuss|include|address)\s+(?:the\s+)?(?:topic|area|field|domain|application|aspect|direction|task)s?\s+(?:of|on|about)?\s*([A-Za-z0-9][A-Za-z0-9 /+_-]{3,90})",
    ]
    stop = re.compile(
        r"(?i)\b(the paper|this paper|authors?|section|line|figure|table|reference|references|there|what|"
        r"more|some|several|many|any|discussion|details?|examples?|works?|papers?|citations?)\b"
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            candidate = normalize_space(match.group(1))
            candidate = re.split(r"[.;:\n]|\s+(?:but|and|or|which|where|because|with|in section)\b", candidate)[0]
            candidate = candidate.strip(" -,:;()[]")
            if 3 <= len(candidate) <= 80 and not stop.fullmatch(candidate):
                words = candidate.split()
                if len(words) <= 10 and not candidate.lower().startswith(("more ", "some ", "several ", "many ")):
                    topics.append(candidate)

    known_topic_phrases = [
        "human-in-the-loop",
        "human in the loop",
        "informal theorem proving via natural language explanation",
        "automated theorem generation",
        "theorem prover feedback",
        "proof search",
        "causal RL",
        "spurious correlation",
        "generalizability",
        "robotics",
        "multimodal dialogue",
        "AI creation",
        "language-modeling related content",
        "DPO",
        "constitutional AI",
        "consistency models",
        "foundation models",
        "future applications of LLMs",
        "emerging challenges",
        "real-world engagement",
        "user studies",
        "substantive equality of opportunity",
        "preference elicitation",
        "interactive algorithmic recourse",
    ]
    lower = text.lower()
    coverage_cue = re.search(r"(?i)(missing|omits?|does not cover|fails to address|should discuss|should include|not discussed|not mention)", text)
    if coverage_cue:
        for phrase in known_topic_phrases:
            if phrase.lower() in lower:
                topics.append(phrase)

    seen = set()
    result = []
    for topic in topics:
        topic = normalize_space(topic).strip(" .")
        key = topic.casefold()
        if key not in seen and not re.fullmatch(r"(?i)(this|that|it|them|the topic|new fields)", topic):
            seen.add(key)
            result.append(topic)
    return result


def analyze() -> tuple[dict, list[dict], Counter, Counter]:
    payload = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    by_forum = {}
    review_rows = []

    for forum_id, reviews in payload.items():
        paper_labels = []
        paper_refs = []
        paper_topics = []
        for idx, review in enumerate(reviews, start=1):
            text = negative_text(review)
            labels = classify_text(text)
            refs = extract_reference_titles(text)
            topics = extract_missing_topics(text)

            if refs and "漏引用具体文献" not in labels:
                labels.append("漏引用具体文献")
            if topics and "未覆盖具体主题" not in labels:
                labels.append("未覆盖具体主题")
            if not labels and normalize_space(text):
                labels.append("个人偏好、venue匹配或其他琐碎问题")

            paper_labels.extend(labels)
            paper_refs.extend(refs)
            paper_topics.extend(topics)
            review_rows.append(
                {
                    "forum_id": forum_id,
                    "review_index": idx,
                    "reviewer_id": review.get("reviewer_id", ""),
                    "weaknesses": list(dict.fromkeys(labels)),
                    "missed_references": refs,
                    "missed_topics": topics,
                }
            )

        by_forum[forum_id] = {
            "weaknesses": sorted(set(paper_labels), key=list(CLUSTERS).index),
            "missed_references": list(dict.fromkeys(paper_refs)),
            "missed_topics": list(dict.fromkeys(paper_topics)),
        }

    review_counts = Counter()
    paper_counts = Counter()
    for row in review_rows:
        review_counts.update(row["weaknesses"])
    for item in by_forum.values():
        paper_counts.update(item["weaknesses"])
    return by_forum, review_rows, review_counts, paper_counts


def build_report(by_forum: dict, review_rows: list[dict], review_counts: Counter, paper_counts: Counter) -> str:
    source_names = {
        "A": "已有文献池",
        "B": "学术共识（在已有文献池未提及）",
        "C": "综述自身",
        "D": "审稿人自己的喜好或专有知识",
    }
    lines = [
        "# Review Weakness Cluster Analysis",
        "",
        f"- 输入文件: `{INPUT_JSON.name}`",
        f"- 论文数: {len(by_forum)}",
        f"- Review 数: {len(review_rows)}",
        "- 统计口径: 一个 review 命中同一类别最多计 1 次；一篇文章只要任一 review 命中该类别，则该文章计 1 次。",
        "- 提取范围: 排除 `strengths`、`reasons_to_accept`，并从 `strengths_and_weaknesses` 中只保留 Weaknesses 段；其余审稿字段中指出的问题均纳入。",
        "",
        "## 聚类结果",
        "",
        "| Weakness 类别 | 标准来源 | Review 频次 | 文章频次 | 说明 |",
        "|---|---|---:|---:|---|",
    ]

    for label, info in CLUSTERS.items():
        if review_counts[label] == 0 and paper_counts[label] == 0:
            continue
        source = f"{info['source']}. {source_names[info['source']]}"
        lines.append(
            f"| {label} | {source} | {review_counts[label]} | {paper_counts[label]} | {info['description']} |"
        )

    lines.extend(
        [
            "",
            "## 漏引用与未覆盖 Topic",
            "",
            "只有当审稿人给出可识别的具体论文标题时，才记录为 `漏引用具体文献`；只有当审稿人给出具体 topic、任务、应用场景或子领域名称时，才记录为 `未覆盖具体主题`。",
            "",
            "### 高频缺失文献标题",
            "",
        ]
    )
    ref_counts = Counter(ref for item in by_forum.values() for ref in item["missed_references"])
    if ref_counts:
        for title, count in ref_counts.most_common(30):
            lines.append(f"- {title}: {count} 篇文章")
    else:
        lines.append("- 未抽取到符合严格条件的具体论文标题。")

    lines.extend(["", "### 高频缺失 Topic", ""])
    topic_counts = Counter(topic for item in by_forum.values() for topic in item["missed_topics"])
    if topic_counts:
        for topic, count in topic_counts.most_common(30):
            lines.append(f"- {topic}: {count} 篇文章")
    else:
        lines.append("- 未抽取到符合严格条件的具体 topic 名称。")

    isolated = [
        label
        for label, info in CLUSTERS.items()
        if info["source"] != "D" and review_counts[label] <= 2 and review_counts[label] > 0
    ]
    lines.extend(["", "## 孤立点说明", ""])
    if isolated:
        for label in isolated:
            lines.append(f"- `{label}` 频次较低但不属于 D 类，因此保留为独立类别。")
    else:
        lines.append("- 本次规则聚类中没有需要单开的低频非 D 类孤立点。")

    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- 逐篇文章 JSON: `{OUTPUT_JSON.name}`",
            f"- 本报告: `{OUTPUT_MD.name}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    by_forum, review_rows, review_counts, paper_counts = analyze()
    OUTPUT_JSON.write_text(json.dumps(by_forum, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(build_report(by_forum, review_rows, review_counts, paper_counts), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Reviews: {len(review_rows)} Papers: {len(by_forum)}")


if __name__ == "__main__":
    main()
