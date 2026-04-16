from tools import *
import asyncio
import json


if __package__:
    from .tools.anchor_surveys import AnchorSurveyFetch
    from .tools.argument_eval import ArgumentStructureEvaluator
    from .tools.citation_parser import CitationParser
    from .tools.claim_segmentation import ClaimSegmentation
    from .tools.dynamic_oracle_generator import DynamicOracleGenerator
    from .tools.fact_check import FactualCorrectnessCritic
    from .tools.golden_topics import GoldenTopicGenerator
    from .tools.minimum_completion import minimum_completion
    from .tools.programmatic_quality import QualityCritic
    from .tools.query_expand import QueryExpand
    from .tools.request_utils import SessionManager
    from .tools.source_critic import MissingPaperCheck
    from .tools.structure_eval import StructureCheck
    from .tools.tool_config import ToolConfig
    from .tools.topic_coverage import TopicCoverageCritic
else:
    from tools.anchor_surveys import AnchorSurveyFetch
    from tools.argument_eval import ArgumentStructureEvaluator
    from tools.citation_parser import CitationParser
    from tools.claim_segmentation import ClaimSegmentation
    from tools.dynamic_oracle_generator import DynamicOracleGenerator
    from tools.fact_check import FactualCorrectnessCritic
    from tools.golden_topics import GoldenTopicGenerator
    from tools.minimum_completion import minimum_completion
    from tools.programmatic_quality import QualityCritic
    from tools.query_expand import QueryExpand
    from tools.request_utils import SessionManager
    from tools.source_critic import MissingPaperCheck
    from tools.structure_eval import StructureCheck
    from tools.tool_config import ToolConfig
    from tools.topic_coverage import TopicCoverageCritic


async def testQueryExpand(config, survey_title):
    qe = await QueryExpand(config)(survey_title)
    with open("debug/qe.json", "w", encoding='utf-8') as f:
        json.dump(qe, f, ensure_ascii=False, indent=2)


async def testAnchorSurveyFetch(config, survey_title):
    with open("debug/qe.json", encoding='utf-8') as f: qe = json.load(f)
    results = await AnchorSurveyFetch(config)(qe['core'], survey_title)
    with open("debug/anchor_papers.json", "w", encoding='utf-8') as f: 
        json.dump(results['anchor_papers'], f, ensure_ascii=False, indent=2)
    with open("debug/anchor_surveys.json", "w", encoding='utf-8') as f: 
        json.dump(results['anchor_surveys'], f, ensure_ascii=False, indent=2)


async def testDynamicOracleGenerator(config, survey_title):
    with open("debug/qe.json", encoding='utf-8') as f: qe = json.load(f)
    oracles = await DynamicOracleGenerator(config)(survey_title, qe['library'])
    assert len(oracles['oracle_papers']) == 1000, len(oracles['oracle_papers'])
    with open("debug/oracles.json", "w", encoding='utf-8') as f: 
        json.dump(oracles['oracle_papers'], f, ensure_ascii=False, indent=2)


async def testGoldenTopics(config, query):
    with open("debug/qe.json", encoding='utf-8') as f: qe = json.load(f)
    with open("debug/anchor_surveys.json", encoding='utf-8') as f: anchor_surveys = json.load(f)
    results = await GoldenTopicGenerator(config)(query, anchor_surveys, qe['library'])
    with open("debug/topics.json", "w", encoding='utf-8') as f: 
        json.dump(results, f, ensure_ascii=False, indent=2)


async def testCitationParser(config, paper):
    task = await CitationParser(config)(paper['citations'])
    paper_map = task['paper_content_map']
    with open("debug/cites.json", "w", encoding='utf-8') as f: 
        json.dump(paper_map, f, ensure_ascii=False, indent=2)
    count = [0, 0, 0, 0]
    for x in paper_map.values(): count[x['status']] += 1
    print(count)


async def testClaimSegmentation(config, paper):
    task = await ClaimSegmentation(config)(paper)
    with open("debug/claims.jsonl", "w", encoding='utf-8') as f: 
        for claim in task['claims']: f.write(json.dumps(claim, ensure_ascii=False) + "\n")
    print(f"We have {len(task['claims'])} claims and {task['errors']} errors.")


async def main():
    try:
        await SessionManager.init()
        config = ToolConfig()
        query = "Transformers Natural Language Processing"
        survey_title = "Transformer models in Natural Language Processing: A Survey"
        with open("pdf/Transformer.json", encoding='utf-8') as f: paper = json.load(f)
        # print(await minimum_completion(paper))
        # await testQueryExpand(config, survey_title)
        # await testAnchorSurveyFetch(config, survey_title)
        # print("testAnchorSurveyFetch passed")
        # await testGoldenTopics(config, query)
        # print("testGoldenTopics passed")
        # await testDynamicOracleGenerator(config, survey_title)
        # print("testDynamicOracleGenerator passed")
        # await testClaimSegmentation(config, paper)
        # print("testClaimSegmentation passed")
        # await testCitationParser(config, paper)
        # print("testCitationParser passed")
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    asyncio.run(main())
