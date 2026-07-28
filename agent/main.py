from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from .agent import evaluate_survey, evaluate_survey_with_session
    from .tools.utility.latex_parser import LatexPaperParser
    from .tools.utility.paper_elements import Paper
    from .tools.utility.request_utils import SessionManager
    from .tools.utility.tool_config import ToolConfig
except ImportError:
    from agent import evaluate_survey, evaluate_survey_with_session
    from tools.utility.latex_parser import LatexPaperParser
    from tools.utility.paper_elements import Paper
    from tools.utility.request_utils import SessionManager
    from tools.utility.tool_config import ToolConfig

DEFAULT_OUTPUT_DIR = "test_output"


@dataclass(frozen=True)
class BatchPaper:
    source_path: Path
    output_dir: Path
    kind: str


def _to_yamlable(value: Any):
    if is_dataclass(value):
        return {key: _to_yamlable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_yamlable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_yamlable(item) for item in value]
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return value


def write_agent_yaml(path: str | Path):
    path = Path(path)
    config = ToolConfig()
    data = {
        "general": {"evaluation_date": config.evaluation_date.strftime("%Y-%m-%d")},
        "agent": _to_yamlable(config.agent_info) | {"max_tokens": config.agent_max_tokens},
        "external_llm": _to_yamlable(config.llm_server_info),
        "rerank": _to_yamlable(config.rerank_server_info) | {"num_documents": config.rerank_n_documents},
        "sbert": {"base_url": config.sbert_server_url},
        "dynamic_oracle": {
            "num_oracle_papers": getattr(config, "num_oracle_papers", 1000),
            "letor_path": getattr(config, "letor_path", "backup/ranker.txt"),
        },
        "citation_parser": {
            "grobid_url": config.grobid_url,
            "grobid_parse_mode": config.grobid_parse_mode,
            "proxy_url": config.proxy_url,
        },
        "source_selection": {
            "topn": config.topn,
            "new_paper_reference_overlap_threshold": config.new_paper_reference_overlap_threshold,
            "citation_velocity_keep_ratio": config.citation_velocity_keep_ratio,
            "minimum_reference_survey_citations": config.minimum_reference_survey_citations,
            "use_openalex_count_by_year": config.use_openalex_count_by_year,
        },
        "topic_coverage": {
            "topic_weak_sim_threshold": config.topic_weak_sim_threshold,
            "topic_sim_threshold": config.topic_sim_threshold,
            "new_paper_topic_similarity_threshold": config.new_paper_topic_similarity_threshold,
            "search_limit": config.topic_coverage_search_limit,
        },
        "topic_papers": {
            "search_limit": config.topic_papers_search_limit,
            "missing_topic_min_community_size": config.missing_topic_min_community_size,
        },
        "fact_check": {"background_reference_similarity_threshold": config.background_reference_similarity_threshold},
        "quality": {
            "sentence_similarity_threshold": config.sentence_similarity_threshold,
            "paragraph_similarity_threshold": config.paragraph_similarity_threshold,
            "redundancy_ngram": config.redundancy_ngram,
        },
        "websearch": {"url": config.websearch_url, "api_key": config.websearch_apikey},
        "openalex": {
            "rate_limit_enabled": config.openalex_rate_limit_enabled,
            "requests_per_second": config.openalex_requests_per_second,
            "api_keys": config.openalex_api_keys,
        },
        "academic_search": {"default_engine": config.default_academic_search_engine},
        "semantic_scholar": {
            "api_key": config.semantic_scholar_api_key,
            "retry_count": config.semantic_scholar_retry_count,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def load_json_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON paper must be an object: {path}")
    return data


def load_paper(path: str | Path) -> Paper:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = load_json_record(path)
        full_content = data.get("full_content", data.get("full_text", data.get("paper")))
        if not isinstance(full_content, dict):
            raise ValueError(f"JSON paper has an invalid full_content field: {path}")
        return Paper.from_skeleton(full_content)
    paper = LatexPaperParser().parse(path)
    if paper is None:
        raise RuntimeError(f"Failed to parse paper from {path}")
    return paper


def _split_queries(query: str | list[str]) -> list[str]:
    if isinstance(query, list):
        queries = [str(item).strip() for item in query if str(item).strip()]
    else:
        queries = [item.strip() for item in str(query or "").split(",") if item.strip()]
    if not queries:
        raise ValueError("query must contain at least one non-empty item")
    return queries


def _query_text(queries: list[str]) -> str:
    return " ".join(queries)


def _query_from_json(path: str | Path) -> list[str]:
    query = load_json_record(path)["query"]
    if not isinstance(query, (str, list)):
        raise ValueError(f"JSON paper has an invalid query field: {path}")
    try:
        return _split_queries(query)
    except ValueError as exc:
        raise ValueError(f"JSON paper has an invalid query field: {path}") from exc


def _config_for_json_publication_date(config: ToolConfig, path: str | Path) -> ToolConfig:
    path = Path(path)
    if path.suffix.lower() != ".json":
        return config
    publication_date = load_json_record(path)["publication_date"]
    if not isinstance(publication_date, str) or len(publication_date.strip()) < 10:
        raise ValueError(f"JSON paper has an invalid publication_date field: {path}")
    return replace(config, evaluation_date=datetime.strptime(publication_date.strip()[:10], "%Y-%m-%d"))


def _has_direct_tex(path: Path) -> bool:
    return any(child.is_file() and child.suffix.lower() == ".tex" for child in path.iterdir())


def _batch_relative_path(source_path: Path, batch_root: Path) -> Path:
    source_path = source_path.resolve()
    batch_root = batch_root.resolve()
    parts = source_path.parts
    test_input_indexes = [idx for idx, part in enumerate(parts) if part.lower() in {"test_input", "test_inputs"}]
    if test_input_indexes:
        rel_parts = parts[test_input_indexes[-1] + 1 :]
        relative = Path(*rel_parts) if rel_parts else Path(source_path.name)
    else:
        relative = source_path.relative_to(batch_root)
    if source_path.is_file() and source_path.suffix.lower() == ".json":
        relative = relative.with_suffix("")
    return relative


def discover_batch_papers(input_dir: str | Path, output_root: str | Path = "test_output") -> list[BatchPaper]:
    input_dir = Path(input_dir)
    output_root = Path(output_root)
    if not input_dir.is_dir():
        raise ValueError(f"Batch input must be a directory: {input_dir}")

    papers: list[BatchPaper] = []
    for child in sorted(input_dir.iterdir(), key=lambda item: item.name.lower()):
        if child.is_file() and child.suffix.lower() == ".json":
            kind = "json"
        elif child.is_dir() and _has_direct_tex(child):
            kind = "latex_dir"
        else:
            continue
        papers.append(BatchPaper(child, output_root / _batch_relative_path(child, input_dir), kind))
    return papers


def _looks_like_batch_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        (child.is_file() and child.suffix.lower() == ".json") or (child.is_dir() and _has_direct_tex(child))
        for child in path.iterdir()
    )


def _query_for_batch_paper(batch_paper: BatchPaper, paper: Paper) -> list[str]:
    if batch_paper.kind == "json":
        return _query_from_json(batch_paper.source_path)
    return _split_queries(batch_paper.source_path.name.lower().replace("_", " "))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate survey papers with SurveyEvaluationAgent.")
    parser.add_argument("--paper", required=False, help="Path to a parsed JSON paper or a LaTeX source directory/file.")
    parser.add_argument("--query", required=False, help="Survey topic/query, comma-separated list allowed, e.g. 'causal reinforcement learning,low-resource learning'.")
    parser.add_argument("--batch-dir", required=False, help="Directory whose first-level JSON files and TeX subdirectories should be evaluated in parallel.")
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run batch evaluation in parallel. Use --no-parallel for serial evaluation.",
    )
    parser.add_argument("--batch-concurrency", type=int, default=2, help="Maximum number of papers evaluated concurrently in batch mode.")
    parser.add_argument("--tool-config", default="agent.yaml", help="Path to ToolConfig yaml.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for module outputs and final result. Batch mode treats this as the output root.")
    parser.add_argument("--write-default-config", action="store_true", help="Write a default ToolConfig yaml and exit.")
    parser.add_argument("--run-modules", default="", help="Comma-separated module numbers to run, e.g. '0,1,3,5,8,12'. Use 2.1-2.5 for 02 substeps. Empty means all modules.")
    parser.add_argument("--force", action="store_true", help="Re-run selected modules even if cached outputs exist.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


async def _run_one_batch_paper(batch_paper: BatchPaper, config: ToolConfig, args: argparse.Namespace) -> dict[str, Any]:
    paper = load_paper(batch_paper.source_path)
    query = _query_for_batch_paper(batch_paper, paper)
    paper_config = _config_for_json_publication_date(config, batch_paper.source_path)
    result = await evaluate_survey_with_session(query, paper, config=paper_config, output_dir=batch_paper.output_dir, run_modules=args.run_modules, force=args.force)
    return {
        "status": "ok",
        "paper": str(batch_paper.source_path),
        "query": _query_text(query),
        "output_dir": str(batch_paper.output_dir),
        "result": result,
    }


async def _run_batch(args: argparse.Namespace, input_dir: str | Path):
    output_root = Path(args.output_dir or "test_output")
    papers = discover_batch_papers(input_dir, output_root=output_root)
    if not papers:
        raise SystemExit(f"No first-level JSON files or TeX-containing subdirectories found in {input_dir}")

    config = ToolConfig.from_yaml(args.tool_config)
    semaphore = asyncio.Semaphore(max(1, args.batch_concurrency))
    logger = logging.getLogger(__name__)

    async def worker(batch_paper: BatchPaper) -> dict[str, Any]:
        async with semaphore:
            logger.info("start batch paper: %s -> %s", batch_paper.source_path, batch_paper.output_dir)
            try:
                return await _run_one_batch_paper(batch_paper, config, args)
            except Exception as exc:
                logger.exception("batch paper failed: %s", batch_paper.source_path)
                return {
                    "status": "error",
                    "paper": str(batch_paper.source_path),
                    "output_dir": str(batch_paper.output_dir),
                    "error": repr(exc),
                }

    output_root.mkdir(parents=True, exist_ok=True)
    await SessionManager.init()
    try:
        if args.parallel:
            results = await asyncio.gather(*(worker(batch_paper) for batch_paper in papers))
        else:
            results = []
            for batch_paper in papers:
                results.append(await worker(batch_paper))
    finally:
        await SessionManager.close()

    summary = {
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "parallel": args.parallel,
        "batch_concurrency": max(1, args.batch_concurrency) if args.parallel else 1,
        "total": len(results),
        "succeeded": sum(1 for item in results if item["status"] == "ok"),
        "failed": sum(1 for item in results if item["status"] == "error"),
        "results": results,
    }
    summary_path = output_root / "batch_results.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Saved batch summary to {summary_path}")
    return summary


async def _run(args: argparse.Namespace):
    if args.write_default_config:
        path = write_agent_yaml(args.tool_config)
        print(f"Wrote default ToolConfig yaml to {path}")
        return

    batch_input = Path(args.batch_dir) if args.batch_dir else None
    paper_path = Path(args.paper) if args.paper else None
    if batch_input is None and paper_path is not None and args.query is None and _looks_like_batch_dir(paper_path):
        batch_input = paper_path
    if batch_input is not None:
        return await _run_batch(args, batch_input)

    if not args.paper:
        raise SystemExit("--paper is required unless --write-default-config or --batch-dir is set")
    config = ToolConfig.from_yaml(args.tool_config)
    paper_path = Path(args.paper)
    paper = load_paper(paper_path)
    query = _split_queries(args.query) if args.query else (_query_from_json(paper_path) if paper_path.suffix.lower() == ".json" else None)
    if not query:
        raise SystemExit("--query is required for non-JSON papers")
    config = _config_for_json_publication_date(config, paper_path)
    result = await evaluate_survey(query, paper, config=config, output_dir=args.output_dir, run_modules=args.run_modules, force=args.force)
    if args.output_dir:
        output_path = Path(args.output_dir) / "result.json"
        print(f"Saved result to {output_path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
