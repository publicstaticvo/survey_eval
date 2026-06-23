from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from .agent import evaluate_survey
    from .tools.utility.latex_parser import LatexPaperParser
    from .tools.utility.tool_config import ToolConfig
except ImportError:
    from agent import evaluate_survey
    from tools.utility.latex_parser import LatexPaperParser
    from tools.utility.tool_config import ToolConfig


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
        "dynamic_oracle": {"num_oracle_papers": config.num_oracle_papers, "letor_path": config.letor_path},
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
        "topic_papers": {"search_limit": config.topic_papers_search_limit},
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
            "max_concurrency": config.openalex_max_concurrency,
            "api_keys": config.openalex_api_keys,
        },
        "academic_search": {"default_engine": config.default_academic_search_engine},
        "semantic_scholar": {"api_key": config.semantic_scholar_api_key},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def load_paper(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    paper = LatexPaperParser().parse(path)
    if paper is None:
        raise RuntimeError(f"Failed to parse paper from {path}")
    return paper.get_skeleton()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one survey paper with SurveyEvaluationAgent.")
    parser.add_argument("--paper", required=False, help="Path to a parsed JSON paper or a LaTeX source directory/file.")
    parser.add_argument("--query", required=False, help="Survey topic/query, e.g. 'causal reinforcement learning'.")
    parser.add_argument("--tool-config", default="agent.yaml", help="Path to ToolConfig yaml.")
    parser.add_argument("--output-dir", default=None, help="Directory for module outputs and final result.")
    parser.add_argument("--write-default-config", action="store_true", help="Write a default ToolConfig yaml and exit.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


async def _run(args: argparse.Namespace):
    if args.write_default_config:
        path = write_agent_yaml(args.tool_config)
        print(f"Wrote default ToolConfig yaml to {path}")
        return None
    if not args.paper or not args.query:
        raise SystemExit("--paper and --query are required unless --write-default-config is set")
    config = ToolConfig.from_yaml(args.tool_config)
    paper = load_paper(args.paper)
    result = await evaluate_survey(args.query, paper, config=config, output_dir=args.output_dir)
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