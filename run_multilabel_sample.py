import asyncio
import json
import time
from pathlib import Path

from agent.tools.preprocess.multilabel_extract import MultiLabelExtraction
from agent.tools.utility.latex_parser import LatexPaperParser
from agent.tools.utility.request_utils import SessionManager
from agent.tools.utility.tool_config import ToolConfig

SOURCE = Path("agent/test_inputs/sgen_surveys/dialogue_systems/dialogue_systems.tex")
if not SOURCE.exists():
    SOURCE = Path("agent/test_inputs/sgen_surveys/dialogue_systems/final_survey_refined.tex")
OUTPUT = Path("agent/test_output/sgen_surveys/dialogue_systems_six_label_sample.json")

def first_paragraph(section):
    for paragraph in section.paragraphs:
        if any(sentence.environment_type == "text" for sentence in paragraph.sentences):
            return paragraph
    for child in section.children:
        found = first_paragraph(child)
        if found is not None:
            return found
    return None

async def main():
    paper = LatexPaperParser().parse(SOURCE)
    selected = [(section.name, first_paragraph(section)) for section in paper.children]
    selected = [(name, paragraph) for name, paragraph in selected if paragraph is not None]
    extractor = MultiLabelExtraction(ToolConfig())
    await SessionManager.init()
    started = time.perf_counter()
    try:
        results = await asyncio.gather(
            *(extractor._run_paragraph(paragraph.sentences, index) for index, (_, paragraph) in enumerate(selected))
        )
    finally:
        await SessionManager.close()
    output = []
    for (name, paragraph), result in zip(selected, results):
        index, matches, errors = result
        output.append({
            "section": name,
            "paragraph_index": index,
            "text": "\n".join(sentence.text for sentence in paragraph.sentences),
            "matches": matches,
            "errors": errors,
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"elapsed_seconds": time.perf_counter() - started, "samples": output}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"elapsed_seconds": time.perf_counter() - started, "sections": len(output), "errors": sum(len(item["errors"]) for item in output), "output": str(OUTPUT)}, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
