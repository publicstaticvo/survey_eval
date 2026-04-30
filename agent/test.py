import asyncio
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEBUG_DIR = BASE_DIR / "debug"
PDF_DIR = BASE_DIR / "pdf"


def _load_json(name: str):
    with (DEBUG_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(name: str, data):
    with (DEBUG_DIR / name).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_jsonl(name: str):
    path = DEBUG_DIR / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __package__:
    from .tools.eval.argument_eval import ArgumentStructureEvaluator
    from .tools.eval.fact_check import FactualCorrectnessCritic
    from .tools.eval.minimum_completion import minimum_completion
    from .tools.eval.missing_papers import MissingPaperCheck
    from .tools.eval.structure_eval import StructureCheck
    from .tools.eval.topic_coverage import TopicCoverageCritic
    from .tools.preprocess.build_sources import AnchorSurveyFetch
    from .tools.preprocess.citation_parser import CitationParser
    from .tools.preprocess.claim_segmentation import ClaimSegmentation
    from .tools.preprocess.dynamic_candidate_pool import DynamicCandidatePool
    from .tools.preprocess.golden_topics import GoldenTopicGenerator
    from .tools.preprocess.query_expand import QueryExpand
    from .tools.utility.request_utils import SessionManager
    from .tools.utility.tool_config import ToolConfig
else:
    from tools.eval.argument_eval import ArgumentStructureEvaluator
    from tools.eval.fact_check import FactualCorrectnessCritic
    from tools.eval.minimum_completion import minimum_completion
    from tools.eval.missing_papers import MissingPaperCheck
    from tools.eval.structure_eval import StructureCheck
    from tools.eval.topic_coverage import TopicCoverageCritic
    from tools.preprocess.build_sources import AnchorSurveyFetch
    from tools.preprocess.citation_parser import CitationParser
    from tools.preprocess.claim_segmentation import ClaimSegmentation
    from tools.preprocess.dynamic_candidate_pool import DynamicCandidatePool
    from tools.preprocess.golden_topics import GoldenTopicGenerator
    from tools.preprocess.query_expand import QueryExpand
    from tools.utility.request_utils import SessionManager
    from tools.utility.tool_config import ToolConfig


async def testQueryExpand(config, survey_title):
    qe = await QueryExpand(config)(survey_title)
    _write_json("qe.json", qe)


async def testAnchorSurveyFetch(config, survey_title):
    results = await AnchorSurveyFetch(config)(survey_title)
    _write_json("anchor_papers.json", results["anchor_papers"])
    _write_json("anchor_surveys.json", results["anchor_surveys"])


async def testDynamicOracleGenerator(config, survey_title):
    anchor_papers = _load_json("anchor_papers.json")
    anchor_surveys = _load_json("anchor_surveys.json")
    oracles = await DynamicCandidatePool(config)(
        survey_title,
        anchor_data={"anchor_papers": anchor_papers, "anchor_surveys": anchor_surveys},
    )
    _write_json("oracles.json", oracles["candidate_pool"])


async def testGoldenTopicGenerator(config, query):
    qe = _load_json("qe.json")
    anchor_surveys = _load_json("anchor_surveys.json")
    results = await GoldenTopicGenerator(config)(query, anchor_surveys, qe["library"])
    _write_json("topics.json", results)


async def testCitationParser(config, paper):
    task = await CitationParser(config)(paper["citations"])
    paper_map = task["paper_content_map"]
    _write_json("cites.json", paper_map)
    count = [0, 0, 0, 0]
    for item in paper_map.values():
        count[item["status"]] += 1
    print(count)


async def testClaimSegmentation(config, paper):
    task = await ClaimSegmentation(config)(paper)
    with (DEBUG_DIR / "claims.jsonl").open("w", encoding="utf-8") as f:
        for claim in task["claims"]:
            f.write(json.dumps(claim, ensure_ascii=False) + "\n")
    print(f"We have {len(task['claims'])} claims and {len(task['errors'])} errors.")


async def testFactualCorrectnessCritic(config):
    claims = _load_jsonl("claims.jsonl")
    citations = _load_json("cites.json")
    title_to_id = {v['title']: k for k, v in citations.items()}
    for i, claim in enumerate(claims): 
        claim['id'] = i
        claim['citation_key'] = title_to_id[claim['citation_key']]

    async def _single_fact_check(claim):
        cited_paper = citations.get(claim['citation_key'])
        if not cited_paper:
            print(f"insufficient info (status == 3) for {claim['id']}")
            return
        try:
            result = await FactualCorrectnessCritic(config)(claim.get("claim_text", ""), cited_paper)
            return {"id": claim['id'], **result['fact_check']}
        except Exception as exc:
            print(f"FactCheck {exc} {claim['id']}")

    results = await asyncio.gather(*[_single_fact_check(claim) for claim in claims], return_exceptions=True)
    with (DEBUG_DIR / "fact_check.jsonl").open("w", encoding="utf-8") as f:
        for item in results:
            if isinstance(item, dict) and results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


async def testMissingPaperCheck(config):
    citations = _load_json("cites.json")
    oracle_papers = _load_json("oracles.json")
    anchor_papers = _load_json("anchor_papers.json")
    topics = _load_json("topics.json")
    result = MissingPaperCheck(config)(
        citations=citations,
        oracle_data=oracle_papers,
        anchor_papers=anchor_papers,
        topics=topics.get("golden_topics", []),
    )
    _write_json("missing_paper_check.json", result['source_evals'])


async def testTopicCoverageCritic(config, paper):
    topics = _load_json("topics.json")
    result = await TopicCoverageCritic(config)(topics.get("golden_topics", []), paper)
    _write_json("topic_coverage.json", result)


async def testStructureCheck(config, paper):
    topic_coverage = _load_json("topic_coverage.json")
    missing_topics = [
        item["topic"]
        for item in topic_coverage.get("topic_evals", {}).get("topic_coverage", [])
        if item.get("status") == "missing"
    ]
    result = await StructureCheck(config)(paper, missing_topics)
    _write_json("structure_eval.json", result)


async def testArgumentStructureEvaluator(config, paper):
    minimum_result = minimum_completion(paper)
    discussion_candidates = minimum_result.get("minimum_check", {}).get("discussion_candidates", [])
    result = await ArgumentStructureEvaluator(config)(
        paper,
        minimum_check_details={"discussion_section_candidates": discussion_candidates},
    )
    _write_json("argument_eval.json", result)


async def main():
    try:
        await SessionManager.init()
        config = ToolConfig()
        query = "Transformers Natural Language Processing"
        survey_title = "Transformer models in Natural Language Processing: A Survey"
        with (PDF_DIR / "Transformer.json").open(encoding="utf-8") as f: paper = json.load(f)

        # print(minimum_completion(paper))
        # await testQueryExpand(config, survey_title)  # 不需要sbert不需要grobid
        # print("testQueryExpand passed")
        # await testAnchorSurveyFetch(config, survey_title)  # 需要sbert需要grobid
        # print("testAnchorSurveyFetch passed")
        # await testDynamicOracleGenerator(config, survey_title)  # 需要sbert不需要grobid
        # print("testDynamicOracleGenerator passed")
        await testGoldenTopicGenerator(config, query)  # 需要sbert不需要grobid
        print("testGoldenTopicGenerator passed")
        # await testClaimSegmentation(config, paper)  # 不需要sbert不需要grobid
        # print("testClaimSegmentation passed")
        # await testCitationParser(config, paper)  # 不需要sbert需要grobid
        # print("testCitationParser passed")
        # await testFactualCorrectnessCritic(config)  # 不需要sbert不需要grobid
        # print("testFactualCorrectnessCritic passed")
        # await testMissingPaperCheck(config)  # 需要sbert不需要grobid
        # print("testMissingPaperCheck passed")
        # await testTopicCoverageCritic(config, paper)  # 需要sbert不需要grobid
        # print("testTopicCoverageCritic passed")
        # await testStructureCheck(config, paper)  # 不需要sbert不需要grobid
        # print("testStructureCheck passed")
        # await testArgumentStructureEvaluator(config, paper)  # 不需要sbert不需要grobid
        # print("testArgumentStructureEvaluator passed")
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    asyncio.run(main())
