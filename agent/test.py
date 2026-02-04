from tools import *
import asyncio
import json
import sys


async def testQueryExpand(config, survey_title):
    qe = await QueryExpand(config)(survey_title)
    with open("debug/qe.json", "w") as f: json.dump(qe, f, ensure_ascii=False)


async def testAnchorSurveyFetch(config, survey_title):
    with open("debug/qe.json") as f: qe = json.load(f)
    results = await AnchorSurveyFetch(config)(qe['core'], qe['library'], survey_title)
    with open("debug/anchor_papers.json", "w") as f: json.dump(results['anchor_papers'], f, ensure_ascii=False)
    with open("debug/downloaded.json", "w") as f: json.dump(results['downloaded'], f, ensure_ascii=False)
    with open("debug/surveys.json", "w") as f: json.dump(results['surveys'], f, ensure_ascii=False)
    with open("debug/topics.txt", "w") as f: f.write("\n".join(results['golden_topics']))


async def testDynamicOracleGenerator(config, survey_title):
    with open("debug/qe.json") as f: qe = json.load(f)
    oracles = await DynamicOracleGenerator(config)(survey_title, qe['library'])
    with open("debug/oracles.json", "w") as f: json.dump(oracles['oracle_papers'], f, ensure_ascii=False)


async def testCitationParser(config, paper):
    task = await CitationParser(config)(paper['citations'])
    paper_map = task['paper_content_map']
    with open("debug/cites.json", "w") as f: json.dump(paper_map, f, ensure_ascii=False)
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
        survey_title = "Transformer models in Natural Language Processing: A Survey"
        with open("pdf/Transformer.json", encoding='utf-8') as f: paper = json.load(f)
        if sys.argv[1] == "query": await testAnchorSurveyFetch(config, survey_title)
        else: await testClaimSegmentation(config, paper)
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    asyncio.run(main())
