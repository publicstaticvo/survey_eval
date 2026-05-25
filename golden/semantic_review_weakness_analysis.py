from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INPUT_JSON = ROOT / "review.json"
ANNOTATIONS_JSON = ROOT / "review_weakness_semantic_annotations.json"
OUTPUT_JSON = ROOT / "review_weaknesses_by_forum_semantic.json"
OUTPUT_MD = ROOT / "review_weakness_clusters_semantic.md"


CATEGORIES = {
    "漏引用具体文献": {
        "source": "A",
        "description": "审稿人明确指出缺少一篇或多篇具体论文、书籍、报告或命名文献。",
    },
    "未覆盖具体主题": {
        "source": "A",
        "description": "审稿人明确指出综述遗漏了具体 topic、任务、应用场景、方法族或子领域。",
    },
    "内容过时或未纳入近期进展": {
        "source": "A",
        "description": "审稿人指出引用、覆盖范围或论述没有反映最新/当前文献状态。",
    },
    "文献检索、纳入标准或系统性不足": {
        "source": "B",
        "description": "检索流程、纳入排除标准、系统综述方法、文献池构建或复现性不足。",
    },
    "覆盖浅显或技术深度不足": {
        "source": "B",
        "description": "已有内容过浅、缺少关键细节、解释不充分，或没有给出足够技术/实践深度。",
    },
    "缺少批判性综合、比较或洞见": {
        "source": "B",
        "description": "综述像罗列，缺少归纳、批判、比较、洞见、立场或面向未来的综合。",
    },
    "缺少实证、量化或基准比较支撑": {
        "source": "B",
        "description": "缺少实验/benchmark、数据集统计、复杂度、表格、指标或横向比较证据。",
    },
    "写作表达、格式、引用或图表呈现问题": {
        "source": "B",
        "description": "文字、排版、图表、可视化、引用格式、拼写、符号或展示质量问题。",
    },
    "定义、术语或分类体系不清": {
        "source": "C",
        "description": "综述自身的概念定义、术语边界、taxonomy、分类逻辑或符号不清。",
    },
    "结构组织、章节衔接或叙事逻辑问题": {
        "source": "C",
        "description": "章节安排、内容顺序、段落衔接、标题命名、叙事主线或组织结构有问题。",
    },
    "范围、标题、目标或贡献不一致": {
        "source": "C",
        "description": "题目/摘要/目标/贡献与正文实际范围不一致，或 scope 过大过小且未自洽。",
    },
    "内部论证、结论或建议支撑不足": {
        "source": "C",
        "description": "结论、建议、主张或 framing 没有被综述自身材料充分支撑，或存在内在张力。",
    },
    "个人偏好、venue匹配或其他琐碎问题": {
        "source": "D",
        "description": "审稿人个人偏好、venue 适配、创新性门槛、建议投其他 venue、或其他零散问题。",
    },
}


NEGATIVE_FIELDS = [
    "weaknesses",
    "strengths_and_weaknesses",
    "requested_changes",
    "reasons_to_reject",
    "Reasons_to_reject",
    "questions",
    "Questions_for_the_Authors",
    "questions_for_the_authors",
    "additional_feedback",
    "additional_comments",
    "review",
    "metareview",
    "Relecture",
]


def strip_strengths(text: str) -> str:
    match = re.search(r"(?is)(?:\*\*)?\s*weaknesses?\s*:?\s*(?:\*\*)?", text or "")
    if match:
        return text[match.end() :].strip()
    return text.strip()


def negative_text(review: dict[str, Any]) -> str:
    chunks = []
    for field in NEGATIVE_FIELDS:
        value = review.get(field, "")
        if not isinstance(value, str) or not value.strip():
            continue
        if field == "strengths_and_weaknesses":
            value = strip_strengths(value)
        chunks.append(f"[{field}]\n{value.strip()}")
    return "\n\n".join(chunks).strip()


def build_prompt(forum_id: str, review: dict[str, Any]) -> str:
    category_lines = "\n".join(
        f"- {name}: 来源{info['source']}；{info['description']}" for name, info in CATEGORIES.items()
    )
    review_block = f"REVIEW_INDEX={review['review_index']}\nREVIEWER_ID={review['reviewer_id']}\n{review['text']}"
    return f"""你是一个严谨的学术审稿意见分析员。请阅读下面这条综述论文 review，只基于语义理解分类，不要使用关键词匹配。

任务：
1. 提取审稿人指出的 weakness。排除 strengths、reasons_to_accept，以及 strengths_and_weaknesses 中的 strengths 部分。
2. 每条 review 可以有多个 weakness 类别。
3. “漏引用具体文献”只有在审稿人明确给出具体缺失文献标题、书名、报告名或明确命名作品时才标注，并把标题写入 missed_references。
4. “未覆盖具体主题”只有在审稿人明确给出具体 topic、任务、方法族、应用场景或子领域名称时才标注，并把名称写入 missed_topics。
5. 个人偏好、venue 匹配、创新性门槛、零散小问题归入 D 类“个人偏好、venue匹配或其他琐碎问题”。
6. 不要为了覆盖而过度标注。没有明确问题的 review 可以 weaknesses=[]。

类别集合：
{category_lines}

输出必须是合法 JSON，且只输出 JSON：
{{
  "forum_id": "{forum_id}",
  "review_index": {review['review_index']},
  "weaknesses": ["类别名"],
  "missed_references": ["具体文献标题"],
  "missed_topics": ["具体topic名称"],
  "rationale": "一句话说明主要判断依据"
}}

待分析 review：
{review_block}
"""


def request_json(prompt: str, model: str, base_url: str, api_key: str, retries: int = 4) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", content, flags=re.S)
            return json.loads(match.group(0) if match else content)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            if attempt == retries:
                raise RuntimeError(f"LLM request failed after {retries} attempts: {exc}") from exc
            time.sleep(min(2**attempt, 20))
    raise AssertionError("unreachable")


def validate_review_annotation(raw: dict[str, Any], forum_id: str, review_index: int) -> dict[str, Any]:
    assert raw["forum_id"] == forum_id
    valid = set(CATEGORIES)
    idx = int(raw["review_index"])
    assert idx == review_index
    labels = [label for label in raw.get("weaknesses", []) if label in valid]
    refs = [str(item).strip() for item in raw.get("missed_references", []) if str(item).strip()]
    topics = [str(item).strip() for item in raw.get("missed_topics", []) if str(item).strip()]
    if refs and "漏引用具体文献" not in labels:
        labels.append("漏引用具体文献")
    if topics and "未覆盖具体主题" not in labels:
        labels.append("未覆盖具体主题")
    if "漏引用具体文献" not in labels:
        refs = []
    if "未覆盖具体主题" not in labels:
        topics = []
    return {
        "review_index": idx,
        "weaknesses": list(dict.fromkeys(labels)),
        "missed_references": list(dict.fromkeys(refs)),
        "missed_topics": list(dict.fromkeys(topics)),
        "rationale": str(raw.get("rationale", "")).strip(),
    }


def collect_review_inputs() -> dict[str, list[dict[str, Any]]]:
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    inputs = {}
    for forum_id, reviews in data.items():
        rows = []
        for index, review in enumerate(reviews, start=1):
            text = negative_text(review)
            rows.append({"review_index": index, "reviewer_id": review.get("reviewer_id", ""), "text": text})
        inputs[forum_id] = rows
    return inputs


def annotate(args: argparse.Namespace) -> dict[str, Any]:
    inputs = collect_review_inputs()
    annotations = {}
    if ANNOTATIONS_JSON.exists() and not args.overwrite:
        annotations = json.loads(ANNOTATIONS_JSON.read_text(encoding="utf-8"))

    api_key = args.api_key or os.environ.get("REVIEW_LLM_API_KEY")
    base_url = args.base_url or os.environ.get("REVIEW_LLM_BASE_URL", "https://api.deepseek.com")
    model = args.model or os.environ.get("REVIEW_LLM_MODEL", "deepseek-v4-flash")
    if not api_key:
        try:
            import sys

            sys.path.append(str(ROOT.parent))
            from agent.tools.utility.tool_config import ToolConfig

            config = ToolConfig()
            api_key = config.llm_server_info.api_key
            base_url = args.base_url or os.environ.get("REVIEW_LLM_BASE_URL") or config.llm_server_info.base_url
            model = args.model or os.environ.get("REVIEW_LLM_MODEL") or config.llm_server_info.model
        except Exception:
            api_key = ""
    if not api_key:
        raise SystemExit("Set REVIEW_LLM_API_KEY or pass --api-key.")

    total_reviews = sum(len(reviews) for reviews in inputs.values())
    done = 0
    for forum_order, (forum_id, reviews) in enumerate(inputs.items(), start=1):
        item = annotations.setdefault(forum_id, {"forum_id": forum_id, "review_annotations": []})
        existing = {
            row["review_index"]: row for row in item.get("review_annotations", [])
        }
        for review in reviews:
            if review["review_index"] in existing and not args.overwrite:
                continue
            prompt = build_prompt(forum_id, review)
            raw = request_json(prompt, model=model, base_url=base_url, api_key=api_key)
            existing[review["review_index"]] = validate_review_annotation(raw, forum_id, review["review_index"])
            item["review_annotations"] = [existing[index] for index in sorted(existing)]
            annotations[forum_id] = item
            ANNOTATIONS_JSON.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
            done += 1
            print(f"[forum {forum_order}/{len(inputs)} review {review['review_index']}] annotated {forum_id} ({done}/{total_reviews})")
            if args.limit and done >= args.limit:
                return annotations
    return annotations


def load_or_annotate(args: argparse.Namespace) -> dict[str, Any]:
    if args.annotate:
        return annotate(args)
    if not ANNOTATIONS_JSON.exists():
        raise SystemExit(f"Missing {ANNOTATIONS_JSON}. Run with --annotate first.")
    return json.loads(ANNOTATIONS_JSON.read_text(encoding="utf-8"))


def aggregate(annotations: dict[str, Any]) -> tuple[dict[str, Any], Counter, Counter]:
    by_forum = {}
    review_counts = Counter()
    paper_counts = Counter()
    for forum_id, item in annotations.items():
        labels = []
        refs = []
        topics = []
        for row in item["review_annotations"]:
            row_labels = list(dict.fromkeys(row["weaknesses"]))
            labels.extend(row_labels)
            refs.extend(row["missed_references"])
            topics.extend(row["missed_topics"])
            review_counts.update(row_labels)
        paper_labels = [name for name in CATEGORIES if name in set(labels)]
        paper_counts.update(paper_labels)
        by_forum[forum_id] = {
            "weaknesses": paper_labels,
            "missed_references": list(dict.fromkeys(refs)),
            "missed_topics": list(dict.fromkeys(topics)),
        }
    return by_forum, review_counts, paper_counts


def write_report(by_forum: dict[str, Any], review_counts: Counter, paper_counts: Counter) -> None:
    source_names = {
        "A": "已有文献池",
        "B": "学术共识（在已有文献池未提及）",
        "C": "综述自身",
        "D": "审稿人自己的喜好或专有知识",
    }
    lines = [
        "# Review Weakness Semantic Cluster Analysis",
        "",
        f"- 输入文件: `{INPUT_JSON.name}`",
        f"- 论文数: {len(by_forum)}",
        f"- Review 数: {sum(review_counts.values())}（多标签计数；下表每类为命中该类的 review 数）",
        "- 方法: 类别由语义理解得到；程序只负责 JSON 校验、聚合和计数，不用关键词规则决定类别。",
        "- 统计口径: 一个 review 命中同一类别最多计 1 次；一篇文章只要任一 review 命中该类别，则该文章计 1 次。",
        "",
        "## 聚类结果",
        "",
        "| Weakness 类别 | 标准来源 | Review 频次 | 文章频次 | 说明 |",
        "|---|---|---:|---:|---|",
    ]
    for label, info in CATEGORIES.items():
        lines.append(
            f"| {label} | {info['source']}. {source_names[info['source']]} | {review_counts[label]} | {paper_counts[label]} | {info['description']} |"
        )

    lines.extend(["", "## 缺失文献标题", ""])
    ref_counts = Counter(ref for item in by_forum.values() for ref in item["missed_references"])
    lines.extend([f"- {ref}: {count} 篇文章" for ref, count in ref_counts.most_common()] or ["- 无"])

    lines.extend(["", "## 缺失 Topic", ""])
    topic_counts = Counter(topic for item in by_forum.values() for topic in item["missed_topics"])
    lines.extend([f"- {topic}: {count} 篇文章" for topic, count in topic_counts.most_common()] or ["- 无"])

    isolated = [
        label
        for label, info in CATEGORIES.items()
        if info["source"] != "D" and 0 < review_counts[label] <= 2
    ]
    lines.extend(["", "## 孤立点说明", ""])
    if isolated:
        lines.extend([f"- `{label}` 频次较低但不属于 D 类，保留为独立类别。" for label in isolated])
    else:
        lines.append("- 没有需要单开的低频非 D 类孤立点。")

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", action="store_true", help="Call an OpenAI-compatible LLM to create semantic annotations.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    annotations = load_or_annotate(args)
    by_forum, review_counts, paper_counts = aggregate(annotations)
    OUTPUT_JSON.write_text(json.dumps(by_forum, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(by_forum, review_counts, paper_counts)
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
