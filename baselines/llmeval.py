from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from survey_eval.agent.tools.preprocess.utils import extract_json
from survey_eval.agent.tools.utility.latex_parser import LatexPaperParser
from survey_eval.agent.tools.utility.latex_parser.paper_elements import (
    LatexEnvironment,
    LatexParagraph,
    LatexPaper,
    LatexSection,
    LatexSentence,
    LatexSubSection,
    LatexSubSubSection,
)
from survey_eval.agent.tools.utility.llmclient import AsyncChat
from survey_eval.agent.tools.utility.request_utils import SessionManager
from survey_eval.agent.tools.utility.tool_config import ToolConfig
from survey_eval.baselines.prompts import CC_PROMPT, SYSTEM, USER_ARISE, USER_PLAIN, USER_TRUSTSURVEY


INPUT_DIR = Path("data/latex_surveys")
OUTPUT_ROOT = REPO_ROOT / "baselines" / "llm"
GRAPH_ENVIRONMENTS = {"figure", "figure*", "table", "table*", "tabular", "longtable"}


class EvalMode(str, Enum):
    PLAIN = "PLAIN"
    ARISE = "ARISE"
    TRUSTSURVEY = "TRUSTSURVEY"


USER_PROMPTS = {
    EvalMode.PLAIN: USER_PLAIN,
    EvalMode.ARISE: USER_ARISE,
    EvalMode.TRUSTSURVEY: USER_TRUSTSURVEY,
}


class SurveyLLMEvalClient(AsyncChat):
    def __init__(self, config: ToolConfig, mode: EvalMode):
        super().__init__(config.llm_server_info, sampling_params={})
        self.mode = mode

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        if not result:
            raise ValueError(f"Failed to extract JSON from LLM response: {response[:500]}")
        return result

    def _organize_inputs(self, inputs):
        user_prompt = self._format_user_prompt(USER_PROMPTS[self.mode], inputs)
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ], {}

    def _format_user_prompt(self, template: str, inputs: str) -> str:
        try:
            return template.format(SURVEY_FULL_TEXT=inputs)
        except (KeyError, ValueError):
            return template.replace("{SURVEY_FULL_TEXT}", inputs)


def _clean_text(text: Any) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _paper_title(paper: LatexPaper | dict[str, Any], fallback: str = "untitled") -> str:
    if isinstance(paper, dict):
        return _clean_text(paper.get("title")) or fallback
    return _clean_text(paper.title) or fallback


def _slugify_title(title: str) -> str:
    slug = re.sub(r"\s+", "_", title.strip().lower())
    slug = re.sub(r"[^a-z0-9_\-]+", "", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "untitled"


def _caption_block(item: LatexEnvironment | dict[str, Any], counters: dict[str, int]) -> str:
    if isinstance(item, dict):
        env_name = item.get("environment_type", "")
        caption = _clean_text(item.get("caption") or item.get("text"))
    else:
        env_name = item.environment_name
        caption = _clean_text(item.caption or item.text)
    if not caption:
        return ""
    counter_key = "table" if env_name in {"table", "table*", "tabular", "longtable"} else "figure"
    counters[counter_key] += 1
    label = "Table" if counter_key == "table" else "Figure"
    return f"<{label} {counters[counter_key]}: {caption}>"


def _render_paragraph(paragraph: LatexParagraph | list[dict[str, Any]], counters: dict[str, int]) -> list[str]:
    if isinstance(paragraph, list):
        items = paragraph
    else:
        items = paragraph.sentences

    text_parts = []
    blocks = []
    for item in items:
        if isinstance(item, dict):
            env_type = item.get("environment_type", "text")
            if env_type in GRAPH_ENVIRONMENTS:
                caption = _caption_block(item, counters)
                if caption:
                    if text_parts:
                        blocks.append(" ".join(text_parts).strip())
                        text_parts = []
                    blocks.append(caption)
                continue
            text = _clean_text(item.get("text"))
        elif isinstance(item, LatexEnvironment):
            if item.environment_name in GRAPH_ENVIRONMENTS:
                caption = _caption_block(item, counters)
                if caption:
                    if text_parts:
                        blocks.append(" ".join(text_parts).strip())
                        text_parts = []
                    blocks.append(caption)
                continue
            text = _clean_text(item.text)
        elif isinstance(item, LatexSentence):
            text = _clean_text(item.text)
        else:
            text = _clean_text(getattr(item, "text", ""))

        if text:
            text_parts.append(text)

    if text_parts:
        blocks.append(" ".join(text_parts).strip())
    return [block for block in blocks if block]


def _render_object_section(
    section: LatexSection | LatexSubSection | LatexSubSubSection,
    depth: int,
    counters: dict[str, int],
) -> list[str]:
    lines = [f"{'#' * depth} {_clean_text(section.name)}"]
    for child in section.children:
        if isinstance(child, LatexParagraph):
            lines.extend(_render_paragraph(child, counters))
        elif isinstance(child, (LatexSection, LatexSubSection, LatexSubSubSection)):
            lines.extend(_render_object_section(child, min(depth + 1, 4), counters))
    return lines


def _render_dict_section(section: dict[str, Any], depth: int, counters: dict[str, int]) -> list[str]:
    title = _clean_text(section.get("title") or section.get("name"))
    lines = [f"{'#' * depth} {title}"] if title else []
    for paragraph in section.get("paragraphs", []) or []:
        if isinstance(paragraph, dict):
            paragraph = paragraph.get("sentences", [])
        lines.extend(_render_paragraph(paragraph, counters))
    for child in section.get("sections", []) or []:
        if isinstance(child, dict):
            lines.extend(_render_dict_section(child, min(depth + 1, 4), counters))
    return lines


def _reference_field(entry: dict[str, Any], *names: str) -> str:
    for name in names:
        value = _clean_text(entry.get(name))
        if value:
            return value
    return ""


def _format_authors(entry: dict[str, Any]) -> str:
    authors = entry.get("author") or entry.get("authors") or ""
    if isinstance(authors, list):
        return ", ".join(_clean_text(author) for author in authors if _clean_text(author))
    return _clean_text(authors)


def _format_reference(key: str, entry: Any) -> str:
    if not isinstance(entry, dict):
        return f"[{key}] {_clean_text(entry)}"
    authors = _format_authors(entry) or "Unknown authors"
    title = _reference_field(entry, "title") or "Untitled"
    venue = _reference_field(entry, "journal", "booktitle", "venue", "publisher", "school")
    year = _reference_field(entry, "year", "date")
    tail = ", ".join(part for part in [venue, year] if part)
    return f"[{key}] {authors}. {title}. {tail}.".rstrip()


def render_paper_markdown(paper: LatexPaper | dict[str, Any]) -> str:
    counters = {"figure": 0, "table": 0}
    lines = [f"# {_paper_title(paper)}"]

    abstract = paper.get("abstract") if isinstance(paper, dict) else paper.abstract
    if abstract:
        abstract_blocks: list[str] = []
        if isinstance(abstract, dict):
            for paragraph in abstract.get("paragraphs", []) or []:
                abstract_blocks.extend(_render_paragraph(paragraph, counters))
        elif isinstance(abstract, LatexSubSubSection):
            for child in abstract.children:
                if isinstance(child, LatexParagraph):
                    abstract_blocks.extend(_render_paragraph(child, counters))
        if abstract_blocks:
            lines.extend(["## Abstract", *abstract_blocks])

    if isinstance(paper, dict):
        section_groups: Iterable[Any] = [
            *(paper.get("sections", []) or []),
            *(paper.get("limitation", []) or []),
            *(paper.get("appendix", []) or []),
        ]
        for section in section_groups:
            if isinstance(section, dict):
                lines.extend(_render_dict_section(section, 2, counters))
        bibliography = paper.get("citations", {}) or paper.get("bibliography", {})
    else:
        for section in [*paper.sections, *paper.limitation, *paper.appendix]:
            lines.extend(_render_object_section(section, 2, counters))
        bibliography = paper.bibliography

    if bibliography:
        lines.append("## References")
        for key, entry in bibliography.items():
            lines.append(_format_reference(str(key), entry))

    return "\n\n".join(line for line in lines if _clean_text(line))


def parse_paper(path: Path) -> LatexPaper:
    paper = LatexPaperParser().parse(path)
    if paper is None:
        raise RuntimeError(f"Failed to parse paper from {path}")
    return paper


def iter_input_paths(input_dir: Path) -> list[Path]:
    if input_dir.is_file():
        return [input_dir]
    children = sorted(input_dir.iterdir())
    tex_files = [path for path in children if path.is_file() and path.suffix.lower() == ".tex"]
    if tex_files:
        return [input_dir]
    return [path for path in children if path.is_dir()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Output file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def resolve_output_path(
    output_root: Path,
    mode: EvalMode,
    input_path: Path,
    paper: LatexPaper | dict[str, Any] | None,
    output_file: str | None,
) -> Path:
    if output_file:
        requested_path = Path(output_file)
        suffix = requested_path.suffix or ".json"
        filename = f"{requested_path.stem}_{mode.value.lower()}{suffix}"
        if requested_path.is_absolute():
            return requested_path.with_name(filename)
        return output_root / requested_path.parent / filename

    if paper is None:
        raise ValueError("paper is required when --output-file is not provided")
    title_slug = _slugify_title(_paper_title(paper, input_path.stem))
    return output_root / title_slug / f"{mode.value.lower()}.json"


def _claude_code_requirements(mode: EvalMode) -> str:
    prompt = USER_PROMPTS[mode]
    survey_end = prompt.index("</survey>") + len("</survey>")
    return prompt[survey_end:].strip().replace("{{", "{").replace("}}", "}")


def build_claude_code_prompt(args: argparse.Namespace) -> str:
    mode = EvalMode(args.mode.upper())
    return CC_PROMPT.format(
        input_dir=str(Path(args.input_dir).resolve()),
        requirements=_claude_code_requirements(mode),
        output_file=args.output_file,
    )


def run_claude_code_eval(args: argparse.Namespace) -> None:
    if not args.output_file:
        raise SystemExit("--output-file is required when --backend claude-code")
    output_path = resolve_output_path(Path(args.output_root), EvalMode(args.mode.upper()), Path(args.input_dir), None, args.output_file)
    if output_path.exists():
        raise SystemExit(f"Output file already exists: {output_path}")
    args.output_file = str(output_path)
    prompt = build_claude_code_prompt(args)
    subprocess.run(["claude", "-p", prompt], check=True)


async def evaluate_one(path: Path, client: SurveyLLMEvalClient, output_root: Path, output_file: str | None) -> Path:
    paper = parse_paper(path)
    output_path = resolve_output_path(output_root, client.mode, path, paper, output_file)
    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")
    survey_full_text = render_paper_markdown(paper)
    print("Start!")
    result = await client.call(inputs=survey_full_text)
    print("Finish!")
    write_json(output_path, result)
    return output_path


def output_file_for_path(output_file: str | None, input_path: Path, include_input_name: bool) -> str | None:
    if not output_file or not include_input_name:
        return output_file
    requested_path = Path(output_file)
    suffix = requested_path.suffix or ".json"
    filename = f"{requested_path.stem}_{input_path.stem}{suffix}"
    return str(requested_path.with_name(filename))


async def main_async(args: argparse.Namespace) -> list[Path]:
    config = ToolConfig.from_yaml(args.tool_config) if args.tool_config else ToolConfig()
    mode = EvalMode(args.mode.upper())
    client = SurveyLLMEvalClient(config, mode)
    input_paths = iter_input_paths(Path(args.input_dir))
    print(input_paths)
    if not input_paths:
        raise SystemExit(f"No LaTeX inputs found under {args.input_dir}")

    output_root = Path(args.output_root)
    include_input_name = len(input_paths) > 1

    async def evaluate_with_report(path: Path) -> Path | None:
        output_file = output_file_for_path(args.output_file, path, include_input_name)
        try:
            output_path = await evaluate_one(path, client, output_root, output_file)
            print(f"Saved {mode.value} evaluation for {path}")
            return output_path
        except Exception as exc:
            print(f"Failed to evaluate {path}: {type(exc).__name__}: {exc}")
            return None

    await SessionManager.init()
    try:
        outputs = await asyncio.gather(*(evaluate_with_report(path) for path in input_paths))
        return [path for path in outputs if path is not None]
    finally:
        await SessionManager.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate survey papers with a direct LLM baseline.")
    parser.add_argument("--backend", choices=["llm", "claude-code"], default="llm")
    parser.add_argument("--input-dir", default=str(INPUT_DIR), help="LaTeX source directory, .tex file, or directory of sources.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Root directory for JSON outputs.")
    parser.add_argument("--output-file", default=None, help="JSON output path used by the claude-code backend.")
    parser.add_argument("--mode", choices=[mode.value for mode in EvalMode], default=EvalMode.PLAIN.value)
    parser.add_argument("--tool-config", default=None, help="Optional ToolConfig yaml path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.backend == "claude-code":
        run_claude_code_eval(args)
    else:
        asyncio.run(main_async(args))


if __name__ == "__main__":
    main()




