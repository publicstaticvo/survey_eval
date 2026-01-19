from tools import *
import asyncio
import json
import sys


async def testAnchorSurveyFetch(config, query, survey_title):
    results = await AnchorSurveyFetch(config)(query, survey_title)
    with open("debug/anchor_papers.json", "w") as f: json.dump(results['anchor_papers'], f, ensure_ascii=False)
    with open("debug/downloaded.json", "w") as f: json.dump(results['downloaded'], f, ensure_ascii=False)
    with open("debug/surveys.json", "w") as f: json.dump(results['surveys'], f, ensure_ascii=False)
    with open("debug/topics.txt", "w") as f: f.write("\n".join(results['golden_topics']))


async def testDynamicOracleGenerator(config, query):
    oracles = await DynamicOracleGenerator(config)(query)
    with open("debug/oracles.json", "w") as f: 
        json.dump(oracles['oracle_papers'], f, ensure_ascii=False)


async def testCitationParser(config, paper):
    task = await CitationParser(config)(paper['citations'])
    paper_map = task['paper_content_map']
    with open("debug/cites.json", "w") as f: json.dump(paper_map, f, ensure_ascii=False)
    count = [0, 0, 0, 0]
    for x in paper_map.values(): count[x['status']] += 1
    print(count)


async def testClaimSegmentation(config, paper):
    task = await ClaimSegmentation(config)(paper)
    with open("debug/claims.json", "w") as f: json.dump(task['claims'], f, ensure_ascii=False)
    print(f"We have {len(task['claims'])} claims and {task['errors']} errors.")


async def main():
    try:
        await SessionManager.init()
        config = ToolConfig()
        query = "Transformers Natural Language Processing"
        survey_title = "Transformer models in Natural Language Processing: A Survey"
        with open("pdf/Transformer.json") as f: paper = json.load(f)
        if sys.argv[1] == "query": await testDynamicOracleGenerator(config, query)
        else: await testClaimSegmentation(config, paper)
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    asyncio.run(main())
