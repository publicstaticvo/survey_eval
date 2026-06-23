from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.tools.preprocess.literature_pool import BuildLiteraturePool
from agent.tools.utility.academic_engine import get_academic_engine
from agent.tools.utility.request_utils import SessionManager
from agent.tools.utility.s2 import S2_DEFAULT_FIELDS
from agent.tools.utility.tool_config import ToolConfig
from agent.tools.utility.utils import valid_check


GOLDEN_DIR = Path(r"P:\AI4S\survey_eval\golden")
LABELED_PATH = GOLDEN_DIR / "missing_specific_references_labeled.jsonl"
PDF_CLASS_DIR = GOLDEN_DIR / "pdf_class"
OUT_PATH = GOLDEN_DIR / "R5.jsonl"
OVERALL_RECORD_TYPE = "overall"
ITEM_RECORD_TYPE = "item"


QUERY_BY_PAPER_ID = {
    4: "causal reinforcement learning",
    5: "verifiable cross-silo federated learning",
    9: "retrieval augmented multimodal generation",
    10: "AI retrosynthetic planning",
    12: "missing modality multimodal learning",
    13: "transformers in reinforcement learning",
    17: "safe reinforcement learning",
    19: "information retrieval model architectures",
    21: "LLM GUI agents benchmark",
    26: "knowledge distillation",
    28: "reinforcement learning from human feedback",
    31: "memorisation in machine learning",
    35: "conditional image synthesis diffusion models",
    38: "metaheuristic algorithm design",
    41: "efficient diffusion models",
    42: "efficient reasoning models",
    43: "efficient large language models",
    45: "uncertainty in graph neural networks",
    51: "vision-language-action robotics",
    53: "deep learning optimization convergence generalization",
    54: "LLM scientific idea generation",
    56: "backdoor attacks image recognition",
    58: "over-smoothing over-squashing graph neural networks",
    59: "image self-supervised learning",
    60: "machine learning with physics knowledge",
    61: "large language models societal domains",
    64: "adversarial attacks multimodal large language models",
    67: "brain encoding decoding neural networks",
    68: "collaborative learning",
    69: "graph neural networks graph types",
    74: "tabular data generation",
    76: "weather prediction foundation models",
    77: "self-play reinforcement learning",
    81: "large language model robustness",
    82: "imbalanced learning SMOTE",
    85: "medical image data drift bias assessment",
    88: "large language model honesty",
    89: "implicit neural representations",
    90: "multimodal token compression",
    91: "inverse constrained reinforcement learning",
    92: "KV cache management LLM acceleration",
    93: "concept bottleneck models",
    96: "formal methods robot policy learning",
    97: "in-context learning retrieved demonstrations",
    102: "computational pathology foundation models",
    105: "large language models tabular data",
    106: "real-time object detection networks",
    108: "causal discovery time series",
    112: "data contamination detection large language models",
    113: "vision language navigation foundation models",
    119: "end-to-end task-oriented dialogue",
    121: "transfer learning natural language processing",
    128: "sentiment analysis measurement bias",
    129: "reinforcement learning clinical decision support",
    130: "culture artificial intelligence",
    132: "combinatorial Bayesian optimization",
    133: "self-training representation learning",
    134: "LLM reasoning evaluation",
    135: "world models autonomous driving",
    138: "scientific survey generation",
    144: "autoregressive vision models",
    145: "trustworthy AI",
    148: "discrete diffusion language models",
    149: "prompt-based adaptation vision models",
    150: "state representation learning reinforcement learning",
    153: "parameter-efficient fine-tuning",
    154: "future frame synthesis",
    155: "role-playing language agents",
    158: "compositional learning AI models",
    163: "open set recognition",
    164: "model merging",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(path)


def write_output(path: Path, items: list[dict[str, Any]], overall: Counter) -> None:
    processed_paper_ids = sorted({item["paper_id"] for item in items})
    records = [
        {
            "record_type": OVERALL_RECORD_TYPE,
            "overall_label_frequency": dict(sorted(overall.items())),
            "processed_paper_ids": processed_paper_ids,
            "num_processed_papers": len(processed_paper_ids),
            "num_items": len(items),
        }
    ]
    records.extend(items)
    write_jsonl(path, records)


def write_literature_pool(path: Path, literature_pool: dict[str, dict[str, Any]]) -> None:
    records = [
        {
            "pool_key": key,
            "label": item.get("label", ""),
            "paper": item.get("paper") or {},
        }
        for key, item in literature_pool.items()
    ]
    write_jsonl(path, records)


def load_existing_output(path: Path) -> tuple[list[dict[str, Any]], Counter, set[int]]:
    if not path.exists():
        return [], Counter(), set()

    existing_items = []
    overall = Counter()
    saw_overall = False
    for record in read_jsonl(path):
        record_type = record.get("record_type")
        if record_type == OVERALL_RECORD_TYPE:
            overall.update(record.get("overall_label_frequency") or {})
            saw_overall = True
            continue
        if record_type in {None, ITEM_RECORD_TYPE} and record.get("reason_label") == "R5":
            item = dict(record)
            item["record_type"] = ITEM_RECORD_TYPE
            existing_items.append(item)

    processed_paper_ids = {item["paper_id"] for item in existing_items}
    if not saw_overall:
        per_paper_frequency = {}
        for item in existing_items:
            per_paper_frequency[item["paper_id"]] = item.get("paper_label_frequency") or {}
        for frequency in per_paper_frequency.values():
            overall.update(frequency)
    return existing_items, overall, processed_paper_ids


def load_paper(paper_id: int) -> tuple[Path, dict[str, Any]]:
    matches = list(PDF_CLASS_DIR.glob(f"{paper_id:03d}_*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one pdf_class file for paper_id={paper_id}, found {len(matches)}")
    data = json.loads(matches[0].read_text(encoding="utf-8"))
    paper = {
        "title": data.get("title") or "",
        "author": data.get("author") or "",
        "abstract": data.get("abstract"),
        "paragraphs": data.get("paragraphs") or [],
        "sections": data.get("sections") or [],
        "citations": data.get("citations") or {},
        "submission_cdate": data.get("submission_cdate"),
        "submission_date": data.get("submission_date") or "",
    }
    return matches[0], paper


def config_for_paper(base_config: ToolConfig, paper: dict[str, Any]) -> ToolConfig:
    submission_cdate = paper.get("submission_cdate")
    if submission_cdate:
        eval_date = datetime.fromtimestamp(submission_cdate / 1000, tz=timezone.utc).replace(tzinfo=None)
        return replace(base_config, evaluation_date=eval_date)
    return base_config


async def missing_paper_result(
    paper_id: int,
    query: str,
    reason: str,
    items: list[dict[str, Any]],
    engine: Any,
) -> dict[str, Any]:
    missing_references = []
    frequency = Counter()
    for item in items:
        for reference in item.get("missed_references", []):
            if not reference_needs_label(reference):
                continue
            ref_out = {k: v for k, v in reference.items() if k != "unrecognized"}
            paper = await resolve_reference_paper(reference, engine)
            title = reference_paper_title(reference, paper)
            if title and not ref_out.get("title"):
                ref_out["title"] = title
            ref_out["literature_pool_label"] = "unseen"
            missing_references.append(ref_out)
            frequency["unseen"] += 1
    return {
        "paper_id": paper_id,
        "paper_title": "",
        "query": query,
        "missing_references": missing_references,
        "label_frequency": dict(sorted(frequency.items())),
        "literature_pool_size": 0,
        "literature_pool_error": reason,
    }


async def build_paper_content_map(citations: dict[str, Any], engine, source_name: str) -> dict[str, Any]:
    async def resolve_one(key: str, value: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        reference = citation_to_reference(value)
        if not reference_needs_label(reference):
            return None
        try:
            paper = await resolve_reference_paper(reference, engine)
        except Exception:
            paper = None
        if not paper:
            return None
        return key, {"metadata": {source_name: paper}}

    tasks = [
        asyncio.create_task(resolve_one(key, value))
        for key, value in (citations or {}).items()
        if isinstance(value, dict)
    ]
    normalized = {}
    for result in await asyncio.gather(*tasks):
        if result is None:
            continue
        key, metadata = result
        normalized[key] = metadata
    return normalized


def reference_needs_label(reference: dict[str, Any]) -> bool:
    location = reference.get("location") or {}
    return bool(reference.get("title")) or location.get("type") in {"arxiv", "doi"}


def normalize_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def normalize_doi(doi: str) -> str:
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:", "", doi)
    return doi


def doi_url(doi: str) -> str:
    return f"https://doi.org/{normalize_doi(doi)}"


def arxiv_doi(arxiv_id: str) -> str:
    return f"10.48550/arxiv.{normalize_arxiv_id(arxiv_id).lower()}"


def is_semantic_scholar_engine(engine: Any) -> bool:
    return engine.__class__.__name__.lower() == "semanticscholar"


def is_openalex_engine(engine: Any) -> bool:
    return engine.__class__.__name__.lower() == "openalex"


def citation_to_reference(citation: dict[str, Any]) -> dict[str, Any]:
    title = citation.get("title") or citation.get("raw_text") or ""
    reference: dict[str, Any] = {"title": title} if title else {}
    doi = citation.get("doi") or citation.get("DOI")
    arxiv = citation.get("arxiv") or citation.get("arxiv_id") or citation.get("arXiv")
    if doi:
        reference["location"] = {"type": "doi", "link": str(doi)}
    elif arxiv:
        reference["location"] = {"type": "arxiv", "link": str(arxiv)}
    return reference


async def paper_by_title(title: str, engine: Any) -> dict[str, Any] | None:
    if not title:
        return None
    try:
        return await engine.find_work_by_title(title)
    except Exception:
        return None


async def first_search_result(engine: Any, **kwargs) -> dict[str, Any] | None:
    try:
        payload = await engine.search_works(per_page=1, **kwargs)
    except Exception:
        return None
    results = payload.get("results") or []
    return results[0] if results else None


async def semantic_scholar_paper_by_location(loc_type: str, link: str, engine: Any) -> dict[str, Any] | None:
    if loc_type == "arxiv":
        query = f"ARXIV:{normalize_arxiv_id(link)}"
    elif loc_type == "doi":
        query = f"DOI:{normalize_doi(link)}"
    else:
        return None
    try:
        return await engine.get_entity(query, entity_type="paper", select=S2_DEFAULT_FIELDS)
    except Exception:
        return await first_search_result(engine, search=query, select=S2_DEFAULT_FIELDS)


async def openalex_paper_by_location(loc_type: str, link: str, engine: Any) -> dict[str, Any] | None:
    if loc_type == "doi":
        normalized = normalize_doi(link)
        for doi_value in (doi_url(normalized), normalized):
            paper = await first_search_result(engine, search="", filter={"doi": doi_value})
            if paper:
                return paper
        return None

    if loc_type == "arxiv":
        arxiv_id = normalize_arxiv_id(link)
        paper = await openalex_paper_by_location("doi", arxiv_doi(arxiv_id), engine)
        if paper:
            return paper
        return await first_search_result(engine, search=f"arxiv {arxiv_id}")

    return None


async def resolve_reference_paper(reference: dict[str, Any], engine: Any) -> dict[str, Any] | None:
    if reference.get("title"):
        paper = await paper_by_title(reference["title"], engine)
        return paper or {"title": reference["title"]}

    location = reference.get("location") or {}
    loc_type = location.get("type")
    link = location.get("link") or ""
    if not link:
        return None
    if is_semantic_scholar_engine(engine):
        return await semantic_scholar_paper_by_location(loc_type, link, engine)
    if is_openalex_engine(engine):
        return await openalex_paper_by_location(loc_type, link, engine)
    paper = await semantic_scholar_paper_by_location(loc_type, link, engine)
    return paper or await openalex_paper_by_location(loc_type, link, engine)


def reference_paper_title(reference: dict[str, Any], paper: dict[str, Any] | None) -> str:
    return (paper or {}).get("title") or reference.get("title") or ""


def pool_label_for_title(title: str, literature_pool: dict[str, dict[str, Any]]) -> str:
    for item in literature_pool.values():
        paper = item.get("paper") or {}
        if valid_check(title, paper.get("title", "")):
            return item.get("label") or "unseen"
    return "unseen"


def collapsed_label(label: str) -> str:
    if label in {"cited_papers", "unseen", "cites", "cited_by"}:
        return label
    return "section_keyword"


async def build_paper_summary(
    paper_id: int,
    items: list[dict[str, Any]],
    query: str,
    base_config: ToolConfig,
) -> dict[str, Any] | None:
    try:
        input_path, paper = load_paper(paper_id)
    except FileNotFoundError as exc:
        print(f"skip paper_id={paper_id}: {exc}", flush=True)
        return None

    try:
        paper_config = config_for_paper(base_config, paper)
        engine = get_academic_engine(paper_config)
        builder = BuildLiteraturePool(paper_config)
        paper_content_map = await build_paper_content_map(paper.get("citations") or {}, engine, builder._source_name())
        pool_result = await builder(query=query, paper=paper, paper_content_map=paper_content_map)
        literature_pool = pool_result["literature_pool"]
    except Exception as exc:
        print(f"skip paper_id={paper_id}: literature pool failed: {exc}", flush=True)
        return None

    if not literature_pool:
        print(f"skip paper_id={paper_id}: empty literature pool", flush=True)
        return None

    missing_references = []
    frequency = Counter()
    for item in items:
        for reference in item.get("missed_references", []):
            if not reference_needs_label(reference): continue
            paper = await resolve_reference_paper(reference, engine)
            title = reference_paper_title(reference, paper)
            label = pool_label_for_title(title, literature_pool) if title else "unseen"
            label = collapsed_label(label)
            ref_out = {k: v for k, v in reference.items() if k != "unrecognized"}
            if title and not ref_out.get("title"):
                ref_out["title"] = title
            ref_out["literature_pool_label"] = label
            missing_references.append(ref_out)
            frequency[label] += 1

    literature_pool_dir = GOLDEN_DIR / f"{base_config.default_academic_search_engine}_literature_pools"
    literature_pool_dir.mkdir(parents=True, exist_ok=True)
    write_literature_pool(literature_pool_dir / f"{input_path.name[:3]}.jsonl", literature_pool)

    return {
        "paper_id": paper_id,
        "paper_title": paper.get("title") or "",
        "submission_date": paper.get("submission_date") or "",
        "submission_cdate": paper.get("submission_cdate"),
        "query": query,
        "missing_references": missing_references,
        "label_frequency": dict(sorted(frequency.items())),
        "literature_pool_size": len(literature_pool),
    }


def attach_paper_summary(item: dict[str, Any], paper_summary: dict[str, Any]) -> dict[str, Any]:
    output = dict(item)
    output["record_type"] = ITEM_RECORD_TYPE
    output["query"] = paper_summary["query"]
    output["submission_date"] = paper_summary.get("submission_date") or ""
    output["submission_cdate"] = paper_summary.get("submission_cdate")
    output["paper_label_frequency"] = paper_summary["label_frequency"]
    output["paper_missing_references"] = paper_summary["missing_references"]
    output["literature_pool_size"] = paper_summary["literature_pool_size"]
    if paper_summary.get("literature_pool_error"):
        output["literature_pool_error"] = paper_summary["literature_pool_error"]
    return output


async def build_items(config: ToolConfig, max_papers: int | None = None) -> tuple[list[dict[str, Any]], Counter]:
    r5_items = [item for item in read_jsonl(LABELED_PATH) if item["reason_label"] == "R5"]
    grouped = defaultdict(list)
    for item in r5_items:
        grouped[item["paper_id"]].append(item)

    output_items, overall, processed_paper_ids = load_existing_output(OUT_PATH)
    paper_ids = [paper_id for paper_id in sorted(grouped) if paper_id not in processed_paper_ids]
    if max_papers is not None:
        paper_ids = paper_ids[:max_papers]

    if processed_paper_ids:
        print(f"resume: loaded {len(processed_paper_ids)} processed papers from {OUT_PATH}", flush=True)

    for pos, paper_id in enumerate(paper_ids, 1):
        query = QUERY_BY_PAPER_ID[paper_id]
        print(f"[{pos}/{len(paper_ids)}] paper_id={paper_id} query={query}", flush=True)
        paper_summary = await build_paper_summary(paper_id, grouped[paper_id], query, config)
        if paper_summary is None:
            continue
        output_items.extend(attach_paper_summary(item, paper_summary) for item in grouped[paper_id])
        overall.update(paper_summary["label_frequency"])
        write_output(OUT_PATH, output_items, overall)
        print(f"checkpoint: wrote {OUT_PATH} after paper_id={paper_id}", flush=True)

    return output_items, overall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="semantic_scholar", choices=["semantic_scholar", "openalex"])
    parser.add_argument("--max-papers", type=int, default=None)
    args = parser.parse_args()
    config = ToolConfig(
        default_academic_search_engine=args.engine,
        openalex_api_keys=[
            "NXd77zSxqdt2XLfu14Npp2",
            "v8Fl7dmrRk2ERkT3npPapC",
            "xnaKKdDHuqcXQPY1Crplwu",
            "OKsOaFG3SbaxrRoYSIUBfx",
            "YFl8EWRMHmmZvEd9cljGXt",
        ],
        proxy_url=""
    )
    async def runner():
        await SessionManager.init()
        try:
            return await build_items(config, max_papers=args.max_papers)
        finally:
            await SessionManager.close()

    output_items, overall = asyncio.run(runner())
    write_output(OUT_PATH, output_items, overall)
    print(f"wrote {OUT_PATH} items={len(output_items)}")
    print("overall_label_frequency", json.dumps(dict(sorted(overall.items())), ensure_ascii=False))


if __name__ == "__main__":
    main()
