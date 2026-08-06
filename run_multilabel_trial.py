import asyncio
import json
import logging
from pathlib import Path

from agent.tools.preprocess.paper_content_classify import PaperContentClassification
from agent.tools.utility.latex_parser import LatexPaperParser
from agent.tools.utility.request_utils import SessionManager
from agent.tools.utility.tool_config import ToolConfig

SOURCE = Path("agent/test_inputs/sgen_surveys/dialogue_systems/final_survey_refined.tex")
OUTPUT = Path("agent/test_output/sgen_surveys/dialogue_systems_six_label_trial")

async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paper = LatexPaperParser().parse(SOURCE)
    if paper is None:
        raise RuntimeError("LaTeX parser returned no paper")
    await SessionManager.init()
    try:
        classifier = PaperContentClassification(ToolConfig())
        classifier.multilabel_extraction.checkpoint_path = OUTPUT / "partial_classified_paper.json"
        paper = await classifier.run_steps("dialogue systems", paper, ["multilabel"], only_missing=False)
    finally:
        await SessionManager.close()
    with (OUTPUT / "02_classified_paper.json").open("w", encoding="utf-8") as handle:
        json.dump(paper.get_skeleton(), handle, ensure_ascii=False, indent=2, default=str)
    with (OUTPUT / "02_error_report.json").open("w", encoding="utf-8") as handle:
        json.dump(classifier.get_last_report(), handle, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(classifier.get_last_report(), ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())