from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from tools.scope.family_coverage import FamilyFrontierCoverage
from tools.utility.latex_parser.tex_parser import LatexPaperParser
from tools.utility.request_utils import SessionManager
from tools.utility.tool_config import ToolConfig


DEFAULT_CASES = {
    "active_learning": "active learning machine learning",
    "aspect_based_sentiment_analysis": "aspect based sentiment analysis natural language processing",
    "continual_learning": "continual learning machine learning",
}


async def evaluate_case(config: ToolConfig, root: Path, name: str, query: str, pool_size: int, assess_topics: bool) -> dict:
    case_dir = root / name
    paper = LatexPaperParser().parse(case_dir / "main.tex", base_path=case_dir)
    assert paper is not None, f"Could not parse {case_dir}"
    result = await FamilyFrontierCoverage(config, pool_size=pool_size, assess_topics=assess_topics)(query, paper)
    output = result.to_dict()
    output["case"] = name
    output["survey_title"] = paper.title
    output["reference_count"] = len(paper.references)
    return output


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--pool-size", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("family_coverage_debug.json"))
    parser.add_argument("--assess-topics", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    config = ToolConfig.from_yaml(repo / "agent.yaml")
    root = repo / "test_inputs" / "cc_surveys"
    outputs = []
    await SessionManager.init()
    try:
        for name in args.cases:
            query = DEFAULT_CASES[name]
            print(f"[family-coverage] {name}: {query}")
            outputs.append(await evaluate_case(config, root, name, query, args.pool_size, args.assess_topics))
        args.output.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
        for item in outputs:
            coverage = item["coverage"]
            print(
                f"{item['case']}: pool={item['pool_size']} graph={item['graph']['edges']} "
                f"citation(macro/micro)={coverage['citation_macro']:.3f}/{coverage['citation_micro']:.3f} "
                f"topic(macro/micro)={coverage['topic_macro'] if coverage['topic_macro'] is not None else 'NA'}/{coverage['topic_micro'] if coverage['topic_micro'] is not None else 'NA'}"
            )
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    asyncio.run(main())
