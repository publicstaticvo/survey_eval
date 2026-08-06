from __future__ import annotations

"""Materialize section--topic--paper mappings for the golden survey corpus."""

import argparse
import asyncio
import json
import re
from pathlib import Path

from survey_eval.agent.tools.preprocess.paper_content_classify import PaperContentClassification
from survey_eval.agent.tools.utility.paper_elements import Paper
from survey_eval.agent.tools.utility.request_utils import SessionManager
from survey_eval.agent.tools.utility.tool_config import ToolConfig


async def classify_one(
    classifier: PaperContentClassification,
    input_path: Path,
    output_path: Path,
) -> dict[str, object]:
    record = json.loads(input_path.read_text(encoding="utf-8"))
    paper = Paper.from_skeleton(record["paper"])
    query = str(record.get("paper_title") or paper.title or "")
    paper = await classifier.run_steps(query, paper, ["sentence", "section", "content"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "paper_id": int(re.match(r"^(\d+)_", input_path.name).group(1)),
                "paper_title": record.get("paper_title", ""),
                "paper": paper.get_skeleton(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    targets = []
    for section in paper.children:
        stack = [section]
        while stack:
            current = stack.pop()
            if current.functional_type == "CONTENT":
                targets.append(current)
            stack.extend(current.children)
    return {
        "paper_id": int(re.match(r"^(\d+)_", input_path.name).group(1)),
        "content_sections": len(targets),
        "parsed_sections": sum(bool(section.parsed_contents) for section in targets),
    }


async def main_async(args: argparse.Namespace) -> None:
    config = ToolConfig.from_yaml(args.config)
    classifier = PaperContentClassification(config)
    inputs = sorted(args.pdf_content.glob("*.json"))
    if args.limit:
        inputs = inputs[:args.limit]
    await SessionManager.init()
    try:
        for input_path in inputs:
            paper_id = int(re.match(r"^(\d+)_", input_path.name).group(1))
            if args.agent_cache_layout:
                stem = input_path.stem
                output_path = args.output_dir / stem / "02_classified_paper.json"
            else:
                output_path = args.output_dir / f"{paper_id:03d}_content.json"
            if output_path.exists() and not args.refresh:
                print(json.dumps({"paper_id": paper_id, "status": "cached"}), flush=True)
                continue
            try:
                result = await classify_one(classifier, input_path, output_path)
                result["status"] = "ok"
            except Exception as exc:
                result = {"paper_id": paper_id, "status": f"error: {exc}"}
            print(json.dumps(result, ensure_ascii=True), flush=True)
    finally:
        await SessionManager.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-content", type=Path, default=Path(__file__).parent / "pdf_content")
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "agent.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "classified_papers")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--agent-cache-layout", action="store_true", default=True)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
