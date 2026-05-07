from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_JSON = Path(__file__).resolve().parent / "review_weaknesses.json"
OUTPUT_MD = Path(__file__).resolve().parent / "review_weakness_clusters.md"


NEGATIVE_FIELDS = [
    "weaknesses",
    "strengths_and_weaknesses",
    "requested_changes",
    "Reasons_to_reject",
    "reasons_to_reject",
    "opportunities_for_improvement",
    "Missing_References",
    "questions_to_authors",
    "Questions_for_the_Authors",
    "additional_feedback",
    "additional_comments",
    "review",
    "metareview",
]


CLUSTER_INFO = {
    "文章结构问题": {
        "standard": "B",
        "description": "章节组织失衡、段落衔接差、结构松散、分类层次混乱等，标准来自被评审综述自身是否形成清晰一致的内部结构。",
    },
    "漏引用": {
        "standard": "A",
        "description": "审稿人明确指出缺失某些已有工作或建议加入具体论文，标准来自领域内已有综述或文献集合。",
    },
    "未提及某个子领域": {
        "standard": "A",
        "description": "审稿人明确指出某个应用方向、任务方向或子领域未被覆盖，标准来自领域内已有综述或文献集合。",
    },
    "定义/术语不清": {
        "standard": "B",
        "description": "核心定义、术语边界、taxononomy 命名或符号表述不清，标准来自文章自身概念系统是否前后一致。",
    },
    "覆盖浅显/深度不足": {
        "standard": "C",
        "description": "虽然覆盖了主题，但解释太浅、细节不足、关键训练/技术细节未展开，标准来自普遍的学术共识：综述应提供足够深度。",
    },
    "缺乏批判性分析或综合洞见": {
        "standard": "C",
        "description": "只是罗列工作，没有形成批判性比较、总结或更高层次洞见，标准来自普遍的学术共识。",
    },
    "内容过时": {
        "standard": "C",
        "description": "遗漏近期进展、举例陈旧、与当前领域状态脱节，标准来自普遍的学术共识。",
    },
    "证据/比较不足": {
        "standard": "B",
        "description": "缺少对方法、数据集、结果或复杂度的表格、定量比较、系统对照等，标准来自文章内部是否支撑自己的综述目标。",
    },
    "写作与表述问题": {
        "standard": "B",
        "description": "语句不清、错别字、图表引用有误、占位符或问号引用等，标准来自文章内部一致性和基本表达质量。",
    },
    "范围界定不清或与标题不符": {
        "standard": "B",
        "description": "标题、摘要、正文实际覆盖范围不一致，或 paper 的 scope 没有说清楚，标准来自被评审综述的内部一致性。",
    },
    "缺乏新意或技术贡献": {
        "standard": "D",
        "description": "这类批评通常依赖审稿 venue 对“创新性”的期待，而不是综述内部一个可程序验证的正确/错误标准。",
    },
    "与投稿 venue 不匹配": {
        "standard": "D",
        "description": "认为不适合某个会议/期刊，本质上是 venue 偏好和评价框架，不是综述内容本身的客观真伪。",
    },
    "图表/可视化不足": {
        "standard": "B",
        "description": "要求增加示意图、综述表、对比图等，标准来自文章内部信息组织是否足够支持读者理解。",
    },
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def iter_review_records():
    for path in sorted(DATA_DIR.glob("*.json")):
        if path.name == "_index.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for review in payload.get("reviews", []):
            review_id = review.get("review_id")
            content = review.get("content", {})
            text_chunks = []
            for field in NEGATIVE_FIELDS:
                value = content.get(field, "")
                if isinstance(value, str) and value.strip():
                    text_chunks.append(f"[{field}] {value}")
            yield {
                "file": path.name,
                "paper_title": payload.get("paper_title", ""),
                "review_id": review_id,
                "review": review,
                "content": content,
                "text": "\n\n".join(text_chunks),
            }


def contains_any(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def classify_weaknesses(text: str) -> list[str]:
    weaknesses = []
    lowered = text.lower()

    if contains_any(lowered, [
        "structure", "organization", "section is too brief", "section 3 is too brief", "unbalanced",
        "not connected", "complicated reading", "dilute the main theme", "framing and organization",
        "structure of the paper", "organization is very poor", "the paper feels empty",
    ]):
        weaknesses.append("文章结构问题")

    if contains_any(lowered, [
        "definition", "terminology", "what is a framework", "needs to discuss it", "clarify",
        "not clear what", "scope needs to be clarified", "take a stance on how", "endorsed",
        "definition of multimodality", "taxonomy is not accurate",
    ]):
        weaknesses.append("定义/术语不清")

    if contains_any(lowered, [
        "superficial", "lacks sufficient detail", "too brief", "not fully elucidate", "single paragraph",
        "more detailed explanation", "deeper", "in-depth critical review", "straightforward summary",
        "without offering new insights", "does not provide insights", "lacks in-depth", "felt very general",
    ]):
        weaknesses.append("覆盖浅显/深度不足")

    if contains_any(lowered, [
        "critical review", "more insights", "provide insights", "critical of", "stronger stance",
        "synthesize", "general recommendations", "actionable domain-specific recommendations",
        "not totally clear about the extent", "future revision could provide",
    ]):
        weaknesses.append("缺乏批判性分析或综合洞见")

    if contains_any(lowered, [
        "outdated", "recent works", "more popular neural networks in 2024", "current standards", "after our initial submission",
    ]):
        weaknesses.append("内容过时")

    if contains_any(lowered, [
        "quantitative analysis", "comparison", "comparative table", "table compares", "dataset size per domain",
        "complexity", "table 1", "no table comparing", "should provide a table", "benchmarking",
    ]):
        weaknesses.append("证据/比较不足")

    if contains_any(lowered, [
        "poorly written", "typos", "question marks", "?", "presentation", "stylegan ?", "pointnet should be",
        "proper description of figure", "redundant", "vague citations", "wrongly cited references",
    ]):
        weaknesses.append("写作与表述问题")

    if contains_any(lowered, [
        "title claims", "scope", "out of scope", "title of scientific rigor", "does not fit requirement",
        "focus of the paper is", "fails to address this topic adequately", "the paper title claims",
        "range of topics", "scope of this work", "scope to papers within 5 years",
    ]):
        weaknesses.append("范围界定不清或与标题不符")

    if contains_any(lowered, [
        "novelty", "no novel", "not provide any new", "technical novelty", "new and original research",
        "meaningful contribution", "does not present any novel", "question their novelty",
    ]):
        weaknesses.append("缺乏新意或技术贡献")

    if contains_any(lowered, [
        "not the right venue", "other venues", "not fit the requirement", "suitable venue",
    ]):
        weaknesses.append("与投稿 venue 不匹配")

    if contains_any(lowered, [
        "diagram", "schematics", "figure", "table", "visualization", "appendix for better visualization",
    ]):
        weaknesses.append("图表/可视化不足")

    return list(dict.fromkeys(weaknesses))


def extract_reference_titles(text: str) -> list[str]:
    titles = []

    if not re.search(r"(?i)(missing reference|missing references|additional papers|should cite|consider adding|incorporate these references|did not cite|more works|potential additional papers)", text):
        return titles

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if not re.search(r"(?i)(missing reference|additional papers|should cite|consider adding|incorporate|related work|more works|paper of|line .* reference|point out some potential additional papers|\[[0-9]+\])", line):
            continue

        for quoted in re.findall(r'"([^"]{8,220})"', line):
            candidate = normalize_space(quoted)
            if (
                "http" not in candidate
                and len(candidate.split()) >= 2
                and "?" not in candidate
                and candidate[:1].isupper()
                and not re.search(r"(?i)(we did not|accounting for|recommendation|line \\d+|section \\d+)", candidate)
            ):
                titles.append(candidate.rstrip("."))

        bullet = re.sub(r"^[\-\*\d\.\)\[\]\s]+", "", line)
        bullet = re.sub(r"https?://\S+", "", bullet).strip()
        if re.match(r"^[A-Z][A-Za-z0-9][A-Za-z0-9:,\-\' ]{8,200}$", bullet):
            if not re.search(r"(?i)(survey track|conference|journal|line \d+|section \d+|recommendation \d+)", bullet):
                if (
                    (":" in bullet or len(bullet.split()) >= 4)
                    and "?" not in bullet
                    and bullet[:1].isupper()
                    and not re.search(r"(?i)(authors may|would like|related work|missing references|more works|there is|please|see strengths|na$)", bullet)
                ):
                    titles.append(bullet.rstrip("."))

    clean = []
    seen = set()
    for title in titles:
        title = normalize_space(title)
        if not title or title.lower() == "na":
            continue
        if title.lower().startswith(("line ", "section ", "recommendation ")):
            continue
        if "?" in title or not title[:1].isupper():
            continue
        if any(title != other and title in other for other in titles):
            continue
        if title not in seen:
            seen.add(title)
            clean.append(title)
    return clean


def extract_missing_topics(text: str) -> list[str]:
    topics = []

    lowered = text.lower()

    if "robotics" in lowered:
        topics.append("robotics")
    if "multimodal dialogue" in lowered:
        topics.append("multimodal dialogue")
    if "ai creation" in lowered:
        topics.append("AI creation")
    if "language-modeling related content" in lowered or "rlhf to language models" in lowered:
        topics.append("language-modeling related content")
    if "informal theorem proving via natural language explanation" in lowered:
        topics.append("informal theorem proving via natural language explanation")
    if "broader set of topics with regards to how ml researchers conduct their work" in lowered:
        topics.append("scientific rigor beyond reproducibility")
    if "dpo" in lowered and "would be mentioned" in lowered:
        topics.append("DPO")
    if "constitutional ai" in lowered:
        topics.append("constitutional AI")
    if "consistency models are not very well discussed" in lowered:
        topics.append("consistency models")

    clean = []
    seen = set()
    for topic in topics:
        topic = normalize_space(topic)
        if not topic:
            continue
        if topic.lower() in {"others", "etc", "new fields", "this topic", "the field"}:
            continue
        if topic not in seen:
            seen.add(topic)
            clean.append(topic)
    return clean


def finalize_review_entry(record: dict) -> dict:
    text = record["text"]
    weaknesses = classify_weaknesses(text)
    missing_references = extract_reference_titles(text)
    missing_topics = extract_missing_topics(text)

    if missing_references:
        if "漏引用" not in weaknesses:
            weaknesses.append("漏引用")
    else:
        weaknesses = [item for item in weaknesses if item != "漏引用"]

    if missing_topics:
        if "未提及某个子领域" not in weaknesses:
            weaknesses.append("未提及某个子领域")
    else:
        weaknesses = [item for item in weaknesses if item != "未提及某个子领域"]

    return {
        "weaknesses": weaknesses,
        "missing_references": missing_references if "漏引用" in weaknesses else [],
        "missing_topics": missing_topics if "未提及某个子领域" in weaknesses else [],
    }


def build_markdown_report(results: dict[str, dict]) -> str:
    counter = Counter()
    for item in results.values():
        counter.update(item["weaknesses"])

    lines = [
        "# Review Weakness Clusters",
        "",
        f"- Review 数量: {len(results)}",
        "",
        "## 聚类结果",
        "",
        "| Weakness | Count | 标准来源 | 说明 |",
        "|---|---:|---|---|",
    ]
    for weakness, count in counter.most_common():
        info = CLUSTER_INFO.get(
            weakness,
            {"standard": "C", "description": "该类问题主要依赖通行的学术写作和综述写作共识。"},
        )
        lines.append(f"| {weakness} | {count} | {info['standard']} | {info['description']} |")

    lines.extend(
        [
            "",
            "## 标准来源图例",
            "",
            "- A. 已有的其他文献综述",
            "- B. 被评审综述的内部一致性",
            "- C. 普遍的学术共识",
            "- D. 审稿人自己的喜好或专有知识",
            "",
            "## 说明",
            "",
            "- `漏引用` 和 `未提及某个子领域` 只在 review 文本中明确出现具体论文标题或具体 topic 时才填充相应字段。",
            "- 如果 review 只是泛泛提到“有漏引”“应补充相关工作”但没有给出可识别的标题，则不会把该 review 标成 `漏引用`。",
            "- 同样地，如果 review 只说“覆盖不全”但没有给出明确子领域名称，则不会把该 review 标成 `未提及某个子领域`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    results = {}
    for record in iter_review_records():
        review_id = record["review_id"]
        if not review_id:
            continue
        results[review_id] = finalize_review_entry(record)

    OUTPUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(build_markdown_report(results), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
