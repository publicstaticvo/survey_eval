from __future__ import annotations

import argparse
import asyncio
import json
import re, os
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from survey_eval.agent.main import discover_batch_papers, load_paper
from survey_eval.agent.tools.utility.latex_parser import LatexPaperParser
from survey_eval.agent.tools.utility.latex_parser.tex_parser import process_input_commands
from survey_eval.agent.tools.utility.paper_elements import Paper, Paragraph, Section, Sentence
from survey_eval.agent.tools.utility.utils import extract_json
from survey_eval.agent.tools.utility.llmclient import AsyncChat
from survey_eval.agent.tools.utility.request_utils import SessionManager
from survey_eval.agent.tools.utility.tool_config import ToolConfig
from survey_eval.baselines.prompts import CC_PROMPT, SYSTEM, USER_ARISE, USER_PLAIN, USER_TRUSTSURVEY


INPUT_DIR = Path("data/latex_surveys")
OUTPUT_ROOT = REPO_ROOT / "baselines" / "llm"
GRAPH_ENVIRONMENTS = {"figure", "figure*", "table", "table*", "tabular", "longtable"}

DEFAULT_PUBLICATION_DATE = "2026-06-30"
PDF_DATE_MANIFEST = REPO_ROOT / "agent" / "test_inputs" / "pdf_content_publication_dates.json"


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
        if isinstance(inputs, dict):
            survey_full_text = inputs["survey_full_text"]
            publication_date = inputs["publication_date"]
        else:
            survey_full_text = inputs
            publication_date = DEFAULT_PUBLICATION_DATE
        user_prompt = self._format_user_prompt(USER_PROMPTS[self.mode], survey_full_text, publication_date)
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ], {}

    def _format_user_prompt(self, template: str, inputs: str, publication_date: str) -> str:
        try:
            return template.format(SURVEY_FULL_TEXT=inputs, publication_date=publication_date)
        except (KeyError, ValueError):
            return template.replace("{SURVEY_FULL_TEXT}", inputs).replace("{publication_date}", publication_date)


def _clean_text(text: Any) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _paper_title(paper: Paper | dict[str, Any], fallback: str = "untitled") -> str:
    if isinstance(paper, dict):
        return _clean_text(paper.get("title")) or fallback
    return _clean_text(paper.title or paper.name) or fallback


def _slugify_title(title: str) -> str:
    slug = re.sub(r"\s+", "_", title.strip().lower())
    slug = re.sub(r"[^a-z0-9_\-]+", "", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "untitled"


def _caption_block(item: Sentence | dict[str, Any], counters: dict[str, int]) -> str:
    if isinstance(item, dict):
        env_name = item.get("environment_type", "")
        caption = _clean_text(item.get("caption") or item.get("text"))
    else:
        env_name = item.environment_type
        caption = _clean_text(item.caption or item.text)
    if not caption:
        return ""
    counter_key = "table" if env_name in {"table", "table*", "tabular", "longtable"} else "figure"
    counters[counter_key] += 1
    label = "Table" if counter_key == "table" else "Figure"
    return f"<{label} {counters[counter_key]}: {caption}>"


def _render_paragraph(paragraph: Paragraph | list[dict[str, Any]], counters: dict[str, int]) -> list[str]:
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
        elif isinstance(item, Sentence):
            if item.environment_type in GRAPH_ENVIRONMENTS:
                caption = _caption_block(item, counters)
                if caption:
                    if text_parts:
                        blocks.append(" ".join(text_parts).strip())
                        text_parts = []
                    blocks.append(caption)
                continue
            text = _clean_text(item.text)
        else:
            text = _clean_text(getattr(item, "text", ""))

        if text:
            text_parts.append(text)

    if text_parts:
        blocks.append(" ".join(text_parts).strip())
    return [block for block in blocks if block]


def _render_object_section(section: Section, depth: int, counters: dict[str, int]) -> list[str]:
    lines = [f"{'#' * depth} {_clean_text(section.name)}"]
    for paragraph in section.paragraphs:
        lines.extend(_render_paragraph(paragraph, counters))
    for child in section.children:
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
        raw_authors = []
        for author in authors:
            if isinstance(author, dict):
                name = author.get("name") or " ".join([*author.get("forenames", []), author.get("surname", "")])
                raw_authors.append(str(name))
            elif _clean_text(author):
                raw_authors.append(str(author))
    else:
        raw_authors = re.split(r"\s+and\s+", str(authors)) if str(authors).strip() else []
    formatted = []
    for author in raw_authors:
        author = _clean_text(author)
        if not author:
            continue
        if "," in author:
            family, given = [part.strip() for part in author.split(",", 1)]
            author = " ".join(part for part in (given, family) if part)
        formatted.append(author)
    return "; ".join(formatted)


def _format_reference(key: str, entry: Any) -> str:
    if not isinstance(entry, dict):
        return f"[{key}] {_clean_text(entry)}"
    info = _clean_text(entry.get("info"))
    authors = _format_authors(entry) or "Unknown authors"
    title = _reference_field(entry, "title")
    venue = _reference_field(entry, "journal", "booktitle", "venue", "publisher", "school")
    year = _reference_field(entry, "year", "date")
    if not title and info:
        return f"[{key}] {info}"
    parts = [authors]
    if title:
        parts.append(f'"{title}"')
    if venue:
        parts.append(venue)
    if year:
        parts.append(year)
    return f"[{key}] " + ". ".join(parts) + "."


def _iter_bibliography(bibliography: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(bibliography, dict):
        return ((str(key), entry) for key, entry in bibliography.items())
    if isinstance(bibliography, list):
        return ((str(key), "") for key in bibliography)
    return ()


def _citation_number_map(paper: Paper | dict[str, Any]) -> dict[str, int]:
    """Recover the numeric citation labels assigned while parsing the paper."""
    mapping: dict[str, int] = {}

    def add_citations(citations: Any) -> None:
        if not isinstance(citations, dict):
            return
        for marker, key in citations.items():
            try:
                number = int(marker)
            except (TypeError, ValueError):
                continue
            if key and str(key) not in mapping:
                mapping[str(key)] = number

    def visit_dict_section(section: Any) -> None:
        if not isinstance(section, dict):
            return
        for paragraph in section.get("paragraphs", []) or []:
            sentences = paragraph.get("sentences", []) if isinstance(paragraph, dict) else paragraph or []
            for sentence in sentences:
                if isinstance(sentence, dict):
                    add_citations(sentence.get("citations"))
        for child in section.get("sections", []) or []:
            visit_dict_section(child)

    if isinstance(paper, Paper):
        for sentence in paper.get_sentences():
            add_citations(sentence.citations)
    else:
        visit_dict_section(paper.get("abstract"))
        for section in (*(paper.get("sections", []) or []), *(paper.get("limitation", []) or []), *(paper.get("appendix", []) or [])):
            visit_dict_section(section)
    return mapping


def _reference_labels(paper: Paper | dict[str, Any], bibliography: Any) -> dict[str, int]:
    """Return numeric labels for bibliography keys, with stable fallback numbering."""
    labels = _citation_number_map(paper)
    used = set(labels.values())
    next_number = max(used, default=0) + 1
    for key, _ in _iter_bibliography(bibliography):
        key = str(key)
        if key not in labels:
            while next_number in used:
                next_number += 1
            labels[key] = next_number
            used.add(next_number)
            next_number += 1
    return labels
def render_paper_markdown(paper: Paper | dict[str, Any]) -> str:
    counters = {"figure": 0, "table": 0}
    lines = [f"# {_paper_title(paper)}"]

    abstract = paper.get("abstract") if isinstance(paper, dict) else paper.abstract
    if abstract:
        abstract_blocks: list[str] = []
        if isinstance(abstract, dict):
            for paragraph in abstract.get("paragraphs", []) or []:
                abstract_blocks.extend(_render_paragraph(paragraph, counters))
        elif isinstance(abstract, Section):
            for paragraph in abstract.paragraphs:
                abstract_blocks.extend(_render_paragraph(paragraph, counters))
            for child in abstract.children:
                abstract_blocks.extend(_render_object_section(child, 3, counters))
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
        for section in [*paper.children, *paper.limitation, *paper.appendix]:
            lines.extend(_render_object_section(section, 2, counters))
        bibliography = paper.references

    if bibliography:
        lines.append("## References")
        labels = _reference_labels(paper, bibliography)
        references = sorted(
            _iter_bibliography(bibliography),
            key=lambda item: labels.get(str(item[0]), float("inf")),
        )
        for key, entry in references:
            key = str(key)
            lines.append(_format_reference(str(labels.get(key, key)), entry))

    return "\n\n".join(line for line in lines if _clean_text(line))



def iter_input_paths(input_dir: Path) -> list[Path]:
    if input_dir.is_file():
        return [input_dir]
    children = sorted(input_dir.iterdir())
    tex_files = [path for path in children if path.is_file() and path.suffix.lower() == ".tex"]
    if tex_files:
        return [input_dir]
    json_files = [path for path in children if path.is_file() and path.suffix.lower() == ".json"]
    if json_files:
        return json_files
    return [batch_paper.source_path for batch_paper in discover_batch_papers(input_dir)]

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
    paper: Paper | dict[str, Any] | None,
    output_file: str | None,
) -> Path:
    if output_file:
        requested_path = Path(output_file)
        suffix = requested_path.suffix or ".json"
        filename = f"{requested_path.stem}_{mode.value.lower()}{suffix}"
        if requested_path.is_absolute():
            return requested_path.with_name(filename)
        return output_root / requested_path.parent / filename

    title_slug = _slugify_title(_paper_title(paper, input_path.stem)) if paper is not None else _slugify_title(input_path.stem)
    return output_root / title_slug / f"{mode.value.lower()}.json"


def _load_native_latex(path: Path) -> str:
    """Expand TeX inputs and append external bibliography sources for the native baseline."""
    parser = LatexPaperParser()
    source_path = path
    if source_path.is_dir():
        main_path = parser._find_main_tex(source_path)
        if not main_path:
            raise FileNotFoundError(f"No TeX file found in {source_path}")
        main_path = Path(main_path)
    else:
        main_path = source_path
    base_path = main_path.parent
    raw_content = parser._read_text_file(main_path)
    processed_content = process_input_commands(raw_content, str(base_path))
    bibliography_blocks = []
    for match in re.finditer(r"\\bibliography\s*\{([^{}]*)\}", raw_content):
        for name in match.group(1).split(","):
            stem = name.strip()
            if not stem:
                continue
            stem = stem.removesuffix(".bib").removesuffix(".bbl")
            for suffix in (".bbl", ".bib"):
                bibliography_path = base_path / f"{stem}{suffix}"
                if bibliography_path.is_file():
                    bibliography_blocks.append(
                        f"% Native baseline bibliography source: {bibliography_path.name}\n{parser._read_text_file(bibliography_path)}"
                    )
                    break
    if bibliography_blocks:
        processed_content = processed_content.rstrip() + "\n\n" + "\n\n".join(bibliography_blocks)
    return processed_content


def load_llm_input(path: Path, render_markdown: bool = False) -> tuple[str, Paper | dict[str, Any] | None]:
    """Return native TeX/JSON by default, or a parsed Markdown view when requested."""
    if path.suffix.lower() == ".json":
        paper = load_paper(path)
        if render_markdown:
            return render_paper_markdown(paper), paper
        return path.read_text(encoding="utf-8"), paper

    if not render_markdown:
        return _load_native_latex(path), None

    parser = LatexPaperParser()
    paper = parser.parse(path)
    if paper is None:
        raise ValueError(f"Could not parse TeX input for Markdown rendering: {path}")
    return render_paper_markdown(paper), paper

def _publication_date_for_path(path: Path) -> str:
    """Resolve the date used as the literature cutoff for one survey input."""
    resolved = path.resolve()
    pdf_root = (REPO_ROOT / "agent" / "test_inputs" / "pdf_content").resolve()
    crawled_root = (REPO_ROOT / "agent" / "test_inputs" / "crawled_surveys").resolve()
    if pdf_root in resolved.parents or resolved == pdf_root:
        if PDF_DATE_MANIFEST.is_file() and path.is_file():
            manifest = json.loads(PDF_DATE_MANIFEST.read_text(encoding="utf-8"))
            entry = manifest.get(path.name) or {}
            if entry.get("publication_date"):
                return entry["publication_date"]
        return DEFAULT_PUBLICATION_DATE
    if crawled_root in resolved.parents or resolved == crawled_root:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return str(data.get("publication_date") or DEFAULT_PUBLICATION_DATE)
    return DEFAULT_PUBLICATION_DATE


def _claude_code_requirements(mode: EvalMode, publication_date: str) -> str:
    """Extract only evaluation instructions; Claude Code reads the survey from the path."""
    prompt = USER_PROMPTS[mode]
    prompt = re.sub(r"<survey>.*?</survey>\s*", "", prompt, count=1, flags=re.DOTALL)
    prompt = prompt.strip().format(publication_date=publication_date, SURVEY_FULL_TEXT="")
    return prompt.replace("{{", "{").replace("}}", "}")

def build_claude_code_prompt(args: argparse.Namespace) -> str:
    mode = EvalMode(args.mode.upper())
    return CC_PROMPT.format(
        input_dir=str(Path(args.input_dir).resolve()),
        requirements=_claude_code_requirements(mode, _publication_date_for_path(Path(args.input_dir))),
        output_file=args.output_file,
    )


def _top_level_baseline_files() -> set[Path]:
    baselines_root = REPO_ROOT / "baselines"
    if not baselines_root.is_dir():
        return set()
    return {path.resolve() for path in baselines_root.iterdir() if path.is_file()}


def _remove_new_top_level_baseline_files(before: set[Path]) -> None:
    for path in sorted(_top_level_baseline_files() - before):
        path.unlink()

def run_claude_code_eval(args: argparse.Namespace) -> None:
    if not args.output_file:
        raise SystemExit("--output-file is required when --backend claude-code")
    output_path = resolve_output_path(Path(args.output_root), EvalMode(args.mode.upper()), Path(args.input_dir), None, args.output_file)
    if output_path.exists():
        raise SystemExit(f"Output file already exists: {output_path}")
    args.output_file = str(output_path)
    prompt = build_claude_code_prompt(args)
    baseline_files = _top_level_baseline_files()
    try:
        print(f"Start! evaluation with model {os.environ.get('ANTHROPIC_MODEL', None)}")
        subprocess.run(["claude", "-p", prompt, "--dangerously-skip-permissions"], check=True)
        print("Finish!")
    finally:
        _remove_new_top_level_baseline_files(baseline_files)


async def evaluate_one(path: Path, client: SurveyLLMEvalClient, output_root: Path, output_file: str | None, render_markdown: bool = False) -> Path:
    survey_full_text, paper = load_llm_input(path, render_markdown=render_markdown)
    survey_full_text = f"\n<survey>\n{survey_full_text}\n</survey>\n"
    output_path = resolve_output_path(output_root, client.mode, path, paper, output_file)
    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")
    print(f"Save to: {output_path}")
    print(f"Start! evaluation with model {client.llm.model}")
    result = await client.call(inputs={"survey_full_text": survey_full_text, "publication_date": _publication_date_for_path(path)})
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
        raise SystemExit(f"No JSON files or TeX inputs found under {args.input_dir}")

    output_root = Path(args.output_root)
    include_input_name = len(input_paths) > 1

    async def evaluate_with_report(path: Path) -> Path | None:
        output_file = output_file_for_path(args.output_file, path, include_input_name)
        try:
            output_path = await evaluate_one(path, client, output_root, output_file, args.render_paper_markdown)
            print(f"Saved {mode.value} evaluation for {path}")
            return output_path
        except Exception as exc:
            print(f"Failed to evaluate {path}: {type(exc).__name__}: {exc}")
            return None

    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))

    await SessionManager.init()
    try:
        async def limited_evaluate(path: Path) -> Path | None:
            async with semaphore:
                return await evaluate_with_report(path)

        outputs = await asyncio.gather(*(limited_evaluate(path) for path in input_paths))
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
    parser.add_argument("--max-concurrency", type=int, default=2, help="Maximum concurrent LLM evaluations for --backend llm.")
    parser.add_argument("--render-paper-markdown", action="store_true", help="Parse the source and send a Markdown rendering, including parsed references, instead of native TeX/JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.backend == "claude-code":
        run_claude_code_eval(args)
    else:
        asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

