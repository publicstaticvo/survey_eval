import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import tqdm


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PAPER_FILE = "golden/0.json"
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
    with Path(name).open(encoding="utf-8") as f:
        paper = json.load(f)
    return paper.get('full_content', paper.get('paper', paper))


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
    from .tools.fact.citation_check import CitationCorrectnessCheck
    from .tools.preprocess.claim_segmentation import ClaimSegmentation
    from .tools.fact.fact_check_single import FactualCorrectnessCritic
    from .tools.preprocess.minimum_completion import minimum_completion
    from .tools.scope.missing_papers import MissingPaperCheck
    from .tools.preprocess.citation_parser import CitationParser
    from .tools.preprocess.contribution_classify import ContributionClassification
    from .tools.preprocess.section_classify import SectionClassification
    from .tools.preprocess.sentences import SentenceClassification
    from .tools.scope.topic_coverage import TopicCoverageCritic
    from .tools.utility.request_utils import SessionManager
    from .tools.utility.tool_config import ToolConfig
else:
    from agent import SurveyEvaluationAgent
    from tools.fact.citation_check import CitationCorrectnessCheck
    from tools.preprocess.claim_segmentation import ClaimSegmentation
    from tools.fact.fact_check_single import FactualCorrectnessCritic
    from tools.preprocess.minimum_completion import minimum_completion
    from tools.scope.missing_papers import MissingPaperCheck
    from tools.preprocess.citation_parser import CitationParser
    from tools.preprocess.contribution_classify import ContributionClassification
    from tools.preprocess.section_classify import SectionClassification
    from tools.preprocess.sentences import SentenceClassification
    from tools.scope.topic_coverage import TopicCoverageCritic
    from tools.utility.request_utils import SessionManager
    from tools.utility.tool_config import ToolConfig


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


async def testSentenceClassification(config, paper):
    result = await SentenceClassification(config)(paper)
    _write_json("classified_paper.json", result)
    print("SentenceClassification: wrote classified_paper.json")


async def testClaimSegmentation(config, paper=None):
    if paper is None:
        paper = _load_json("classified_paper.json")
    result = await ClaimSegmentation(config)(paper)
    _write_json("claim_segmentation.json", result)
    _write_jsonl("claims.jsonl", result["claims"])
    print(
        "ClaimSegmentation: "
        f"{len(result['claims'])} claims, {len(result['errors'])} errors"
    )


async def testCitationCorrectnessCheck(config, paper):
    citation_data = _load_references()
    result = await CitationCorrectnessCheck()(paper.get("citations", {}), citation_data)
    _write_json("citation_correctness.json", result["citation_evals"])
    print(
        "CitationCorrectnessCheck: "
        f"{result['citation_evals']['failed_count']}/{result['citation_evals']['checked_count']} failed"
    )


async def testClassification(config, paper, output_file: str | os.PathLike = "classified_paper.json"):
    result = await SectionClassification(config)(paper)
    result = await SentenceClassification(config)(result)
    result = await ContributionClassification(config)(result)
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = _debug_path(str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Classification: wrote {output_path}")
    return result


async def batch_testClassification(
    config,
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    paper_paths = sorted(path for path in input_root.rglob("*.json") if path.is_file())

    for paper_path in tqdm.tqdm(paper_paths, desc="Classification"):
        paper = _load_paper(paper_path)
        result = await SectionClassification(config)(paper)
        result = await SentenceClassification(config)(result)
        # result = await ContributionClassification(config)(result)
        output_path = output_root / paper_path.relative_to(input_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"batch_testClassification: wrote {len(paper_paths)} files to {output_root}")

    # async def _single(paper_path):
    #     with paper_path.open(encoding="utf-8") as f:
    #         paper = json.load(f)['full_content']
    #     paper = await SentenceClassification(config)(await SectionClassification(config)(paper))
    #     output_path = output_root / paper_path.relative_to(input_root)
    #     return output_path, paper

    # tasks = [asyncio.create_task(_single(x)) for x in paper_paths]
    # success = 0
    # for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Classification"):
    #     try:
    #         output_path, result = await task
    #         output_path.parent.mkdir(parents=True, exist_ok=True)
    #         with output_path.open("w", encoding="utf-8") as f:
    #             json.dump(result, f, ensure_ascii=False, indent=2)
    #         success += 1
    #     except Exception as e:
    #         print(f"testClassification {e}")
    # print(f"batch_testClassification: wrote {success}/{len(paper_paths)} files to {output_root}")


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
    paper = _load_json("classified_paper.json")
    await MissingPaperCheck(config)(paper, citations)
    print("MissingPaperCheck: expansion logic completed")


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
        # paper = _load_paper(BASE_DIR / DEFAULT_PAPER_FILE)
        # await testClassification(config, paper, "class.json")
        await batch_testClassification(config, "../golden/pdf_content", "../golden/pdf_class")
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    asyncio.run(main())
