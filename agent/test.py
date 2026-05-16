import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import tqdm


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdf"
DEFAULT_PAPER_FILE = "Transformer.json"
DEBUG_DIR = BASE_DIR / "debug" / DEFAULT_PAPER_FILE.replace(' ', '_').split('.')[0]
DEFAULT_QUERY = "Transformers Natural Language Processing"
DEFAULT_SURVEY_TITLE = "Transformer models in Natural Language Processing: A Survey"


def _debug_path(name: str) -> Path:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    return DEBUG_DIR / name


def _load_json(name: str, default: Any = None):
    path = _debug_path(name)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Missing debug input: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(name: str, data):
    with _debug_path(name).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_jsonl(name: str):
    path = _debug_path(name)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(name: str, rows):
    with _debug_path(name).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _references_dir() -> Path:
    path = _debug_path("references")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _reference_file_name(citation_key: str) -> str:
    safe_key = re.sub(r"[^0-9A-Za-z._-]+", "_", str(citation_key)).strip("._")
    safe_key = safe_key[:80] or "citation"
    digest = hashlib.sha1(str(citation_key).encode("utf-8")).hexdigest()[:10]
    return f"{safe_key}_{digest}.json"


def _clear_reference_outputs():
    for path in _references_dir().glob("*.json"): path.unlink()


def _write_reference(citation_key: str, info: dict[str, Any]):
    payload = {"citation_key": citation_key, "info": info}
    with (_references_dir() / _reference_file_name(citation_key)).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_references() -> dict[str, Any]:
    citations = {}
    for path in sorted(_references_dir().glob("*.json")):
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        citation_key = payload["citation_key"]
        citations[citation_key] = payload["info"]
    return citations


def _load_paper(name: str = DEFAULT_PAPER_FILE):
    with (PDF_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


def _status_counter(paper_content_map: dict[str, Any]) -> dict[str, int]:
    counts = {"0": 0, "1": 0, "2": 0, "3": 0}
    for item in paper_content_map.values():
        status = str(item.get("status", 3))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _minimum_details(minimum_result: dict[str, Any]) -> dict[str, Any]:
    details = minimum_result.get("minimum_check", {})
    return {"discussion_section_candidates": details.get("discussion_candidates", [])}


if __package__:
    from .agent import SurveyEvaluationAgent
    from .tools.eval.argument_eval import ArgumentStructureEvaluator
    from .tools.eval.fact_check import FactualCorrectnessCritic
    from .tools.eval.minimum_completion import minimum_completion
    from .tools.eval.structure_eval import StructureCheck
    from .tools.preprocess.golden_topics import GoldenTopicGenerator
    from .tools.eval.missing_papers import MissingPaperCheck
    from .tools.preprocess.citation_parser import CitationParser
    from .tools.preprocess.claim_segmentation import ClaimSegmentation
    from .tools.preprocess.extract_scope import ScopeClaimExtract
    from .tools.eval.topic_coverage import TopicCoverageCritic
    from .tools.utility.request_utils import SessionManager
    from .tools.utility.tool_config import ToolConfig
else:
    from agent import SurveyEvaluationAgent
    from tools.eval.argument_eval import ArgumentStructureEvaluator
    from tools.eval.fact_check import FactualCorrectnessCritic
    from tools.eval.minimum_completion import minimum_completion
    from tools.eval.structure_eval import StructureCheck
    from tools.preprocess.golden_topics import GoldenTopicGenerator
    from tools.eval.missing_papers import MissingPaperCheck
    from tools.preprocess.citation_parser import CitationParser
    from tools.preprocess.claim_segmentation import ClaimSegmentation
    from tools.preprocess.extract_scope import ScopeClaimExtract
    from tools.eval.topic_coverage import TopicCoverageCritic
    from tools.utility.request_utils import SessionManager
    from tools.utility.tool_config import ToolConfig


async def testGoldenTopicGenerator(config, query, paper):
    generator = GoldenTopicGenerator(config)
    generator.source.debug_dir = DEBUG_DIR
    anchor_data_path = _debug_path("anchor_data.json")
    anchor_survey_path = _debug_path("anchor_survey.json")
    if anchor_data_path.exists() and anchor_survey_path.exists():
        reference_data = {
            "reference_papers": _load_json("anchor_data.json"),
            "reference_surveys": _load_json("anchor_survey.json"),
        }
    else:
        reference_data = await generator.source(query)
        _write_json("anchor_data.json", reference_data.get("reference_papers", {}))
        _write_json("anchor_survey.json", reference_data.get("reference_surveys", {}))

    reference_surveys = reference_data.get("reference_surveys", {}) or {}
    reference_topics = []
    if len(reference_surveys) >= 2:
        raw_topics = await generator.llm.call(inputs={"query": query, "anchor_surveys": reference_surveys})
        reference_topics = raw_topics["topics"]

    result = {
        "reference_data": reference_data,
        "reference_topics": reference_topics,
        "self_topics": await generator._self_scope(paper),
    }
    _write_json("topics.json", {"reference": result['reference_topics'], "self": result['self_topics']})
    print(
        "GoldenTopicGenerator: "
        f"{len(result.get('reference_topics', []))} reference topics, "
        f"{len((result.get('self_topics', {}) or {}).get('aspect_list', []))} self aspects"
    )
    return result


async def testScopeClaimExtract(config):
    extractor = ScopeClaimExtract(config)
    survey_dir = _debug_path("surveys")
    results = {}
    for path in sorted(survey_dir.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        print(f"Load {path}")
        paper = (payload.get("paper") or {}).get("full_content") or payload.get("paper") or {}
        title = payload.get("title") or path.stem
        try:
            result = await extractor(paper, candidate_types=["introduction", "conclusion"])
        except Exception as exc:
            print(f"testScopeClaimExtract {title} {exc}")
            result = {"section_map": {}, "aspect_list": [], "evidence_records": [], "errors": 1}
        results[title] = result
    _write_json("scope_claims.json", results)
    print(f"ScopeClaimExtract: {len(results)} surveys")
    return results


async def testMinimumCompletion(paper):
    result = minimum_completion(paper)
    _write_json("minimum_check.json", result["minimum_check"])


async def testCitationParser(config, paper):
    parser = CitationParser(config)
    citations = paper.get("citations", {})
    # _clear_reference_outputs()
    tasks = [
        asyncio.create_task(parser._parse_single(citation_key, citation_info))
        for citation_key, citation_info in citations.items()
        if not (_references_dir() / _reference_file_name(citation_key)).exists()
    ]
    paper_content_map = {}
    try:        
        for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            try:
                citation_key, info = await task
                paper_content_map[citation_key] = info
                _write_reference(citation_key, info)
            except Exception as exc:
                print(f"testCitationParser {exc}")
    finally:
        print(f"CitationParser: statuses={_status_counter(paper_content_map)}")


async def testClaimSegmentation(config, paper):
    result = await ClaimSegmentation(config)(paper)
    _write_jsonl("claims.jsonl", result['claims'])
    print(f"ClaimSegmentation: {len(result['claims'])} claims, {len(result['errors'])} errors")


async def testFactualCorrectnessCritic(config):
    claims = _load_jsonl("claims.jsonl")
    citations = _load_references()

    async def _single_fact_check(index: int, claim: dict[str, Any]):
        citation_key = claim.get("citation_key")
        cited_paper = citations.get(citation_key)
        if not cited_paper:
            return {
                "id": index,
                "claim": claim.get("claim_text", ""),
                "citation_key": citation_key,
                "judgment": "NEUTRAL",
                "reason": "missing citation metadata",
                "score": 0.0,
                "material": "missing",
            }
        if cited_paper.get("status", 3) == 3:
            return {
                "id": index,
                "claim": claim.get("claim_text", ""),
                "citation_key": citation_key,
                "judgment": "NEUTRAL",
                "reason": "unresolved citation",
                "score": 0.0,
                "material": "missing",
            }
        try:
            result = await FactualCorrectnessCritic(config)(claim.get("claim_text", ""), cited_paper)
            return {"id": index, "citation_key": citation_key, **result.get("fact_check", {})}
        except Exception as exc:
            return {
                "id": index,
                "claim": claim.get("claim_text", ""),
                "citation_key": citation_key,
                "judgment": "NEUTRAL",
                "reason": f"fact check error: {exc}",
                "score": 0.0,
                "material": "error",
            }

    results = await asyncio.gather(*[_single_fact_check(i, claim) for i, claim in enumerate(claims)])
    _write_jsonl("fact_check.jsonl", results)
    print(f"FactualCorrectnessCritic: {len(results)} checks")


async def testTopicCoverageCritic(config, paper):
    topics = _load_json("topics.json")
    result = await TopicCoverageCritic(config)(topics, paper)
    topic_evals = result.get("topic_evals", {})
    _write_json("topic_coverage.json", topic_evals)
    print(
        "TopicCoverageCritic: "
        f"{len(topic_evals['covered_topics'])} covered, "
        f"{len(topic_evals['missing_topics'])} missing, "
        f"{len(topic_evals['self_consistency'])} self-inconsistent"
    )


async def testMissingPaperCheck(config, query):
    citations = _load_references()
    topics = _load_json("topics.json")
    topic_eval = _load_json("topic_coverage.json")
    result = (await MissingPaperCheck(config)(query, citations, topics, topic_eval))["source_evals"]
    missing_references = [x['title'] for x in result['missing_reference_papers']]
    missing_new = [x['title'] for x in result['missing_new_papers']]
    _write_json("missing_paper_check.json", {"old": missing_references, "new": missing_new})
    print(
        "MissingPaperCheck: "
        f"{len(missing_references)} old, "
        f"{len(missing_references)} new"
    )


async def testStructureCheck(config, paper):
    result = await StructureCheck(config)(paper)['structure_evals']
    print(f"StructureCheck: {result}")


async def testArgumentStructureEvaluator(config, paper):
    minimum_result = _load_json("minimum_check.json")
    result = await ArgumentStructureEvaluator(config)(paper, _minimum_details(minimum_result))["argument_evals"]
    _write_json("argument_eval.json", result)


async def testSurveyEvaluationAgent(config, query, paper):
    result = await SurveyEvaluationAgent(config).evaluate(query, paper)
    _write_json("agent_eval.json", result)
    print("SurveyEvaluationAgent: done")


async def main():
    await SessionManager.init()
    try:
        config = ToolConfig()
        query = DEFAULT_QUERY
        survey_title = DEFAULT_SURVEY_TITLE
        paper = _load_paper(DEFAULT_PAPER_FILE)

        # await testMinimumCompletion(paper)
        # await testGoldenTopicGenerator(config, query, paper)
        await testScopeClaimExtract(config)
        # await testCitationParser(config, paper)
        # await testClaimSegmentation(config, paper)
        # await testFactualCorrectnessCritic(config)
        # await testTopicCoverageCritic(config, paper)
        # await testMissingPaperCheck(config, query)
        # await testStructureCheck(config, paper)
        # await testArgumentStructureEvaluator(config, paper)
        # await testSurveyEvaluationAgent(config, query, paper)
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    asyncio.run(main())
