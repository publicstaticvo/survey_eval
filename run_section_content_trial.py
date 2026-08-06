import asyncio
import copy
import json
import logging
import time
from pathlib import Path

from agent.tools.preprocess.content_parser_paragraph import ParagraphContentParser
from agent.tools.preprocess.section_classify_hierarchical import HierarchicalSectionClassification
from agent.tools.utility.paper_elements import Paper
from agent.tools.utility.request_utils import SessionManager
from agent.tools.utility.tool_config import ToolConfig


INPUT = Path("agent/test_output/sgen_surveys/dialogue_systems/02_classified_paper.json")
OUTPUT = Path("agent/test_output/sgen_surveys/dialogue_systems_section_content_trial")


def sample_paper(paper: Paper) -> Paper:
    sample = copy.deepcopy(paper)
    selected = []
    for section in sample.children:
        if section.name in {"Introduction", "Foundations of Dialogue System Architectures", "Emerging Trends, Open Challenges, and Future Directions"}:
            section.children = section.children[:1]
            selected.append(section)
    sample.children = selected
    return sample


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paper = sample_paper(Paper.from_skeleton(json.loads(INPUT.read_text(encoding="utf-8"))))
    started = time.perf_counter()
    await SessionManager.init()
    try:
        section_detector = HierarchicalSectionClassification(ToolConfig())
        paper = await section_detector(paper, only_missing=False)
        content_detector = ParagraphContentParser(ToolConfig())
        paper = await content_detector(paper, only_missing=False)
    finally:
        await SessionManager.close()
    (OUTPUT / "02_section_content_sample.json").write_text(
        json.dumps(paper.get_skeleton(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {"elapsed_seconds": time.perf_counter() - started, "section": section_detector.last_report, "content": content_detector.last_report}
    (OUTPUT / "02_error_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
