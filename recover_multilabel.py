import asyncio
import json
from pathlib import Path

from agent.tools.preprocess.multilabel_extract import MultiLabelExtraction
from agent.tools.utility.content_walk import split_content_to_paragraph
from agent.tools.utility.latex_parser import LatexPaperParser
from agent.tools.utility.request_utils import SessionManager
from agent.tools.utility.tool_config import ToolConfig

SOURCE = Path("agent/test_inputs/sgen_surveys/dialogue_systems/dialogue_systems.tex")
if not SOURCE.exists():
    SOURCE = Path("agent/test_inputs/sgen_surveys/dialogue_systems/final_survey_refined.tex")
OUT = Path("agent/test_output/sgen_surveys/dialogue_systems_six_label_trial")
FAILED = [0, 23, 39, 57, 68, 70, 140, 182]

async def main():
    paper = LatexPaperParser().parse(SOURCE)
    paragraphs = split_content_to_paragraph(paper, include_abstract=True, include_appendix=False)
    extractor = MultiLabelExtraction(ToolConfig())
    await SessionManager.init()
    try:
        results = await asyncio.gather(*(extractor._run_paragraph(paragraphs[i], i) for i in FAILED))
    finally:
        await SessionManager.close()
    errors = []
    recovered = []
    for index, result in zip(FAILED, results):
        paragraph_index, extracted, paragraph_errors = result
        errors.extend(paragraph_errors)
        paragraph = paragraphs[paragraph_index]
        by_id = {f"p{paragraph_index}-s{j}": sentence for j, sentence in enumerate(paragraph)}
        for label, matches in extracted.items():
            for match in matches:
                targets = [by_id[item_id] for item_id in match["item_id"] if item_id in by_id]
                for sentence in targets:
                    sentence.label_extractions.setdefault(label, []).append(match)
                recovered.append({"paragraph": paragraph_index, "label": label, "item_id": match["item_id"]})
    (OUT / "recovery_report.json").write_text(json.dumps({"failed_indices": FAILED, "recovered": recovered, "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not errors:
        (OUT / "02_classified_paper_recovered.json").write_text(json.dumps(paper.get_skeleton(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"errors": len(errors), "recovered_matches": len(recovered), "output": str(OUT / "02_classified_paper_recovered.json")}, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
