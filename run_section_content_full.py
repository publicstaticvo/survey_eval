import asyncio
import json
import logging
from pathlib import Path

from agent.tools.preprocess.paper_content_classify import PaperContentClassification
from agent.tools.utility.paper_elements import Paper
from agent.tools.utility.request_utils import RateLimit, SessionManager
from agent.tools.utility.tool_config import ToolConfig


INPUT = Path("agent/test_output/sgen_surveys/dialogue_systems/02_classified_paper.json")
OUTPUT = Path("agent/test_output/sgen_surveys/dialogue_systems_section_content_full_v3")


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paper = Paper.from_skeleton(json.loads(INPUT.read_text(encoding="utf-8")))
    RateLimit.AGENT_SEMAPHORE = asyncio.Semaphore(8)
    await SessionManager.init()
    try:
        classifier = PaperContentClassification(ToolConfig())
        paper = await classifier.run_steps("dialogue systems", paper, ["section", "content"], only_missing=False)
    finally:
        await SessionManager.close()
    (OUTPUT / "02_classified_paper.json").write_text(
        json.dumps(paper.get_skeleton(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "02_error_report.json").write_text(
        json.dumps(classifier.get_last_report(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(classifier.get_last_report(), ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
