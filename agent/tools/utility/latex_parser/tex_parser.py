"""
Latex Academic Paper Parser with Object-Oriented Structure
Implements Paper, Section, Paragraph, and Sentence classes
"""

import os
import re
import json
import glob
import io
import contextlib
import logging
from typing import List, Any, Union, Optional
from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexwalker import (
    LatexWalker,
    LatexEnvironmentNode,
    LatexMacroNode,
    LatexCharsNode,
    LatexCommentNode,
    LatexSpecialsNode,
)
try:
    from ..paper_elements import *
except ImportError:
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from paper_elements import *
from .constants import *
from .bib_parser import parse_bbl_file, parse_bib_file, detect_encoding


SECTION_MACROS = {'section', 'subsection', 'subsubsection'}
CITATION_MACROS = {
    'cite', 'citep', 'citet', 'citealt', 'citeyearpar',
    'citealp', 'citeauthor', 'citeyear', 'citetext',
    'parencite', 'textcite', 'autocite', 'footcite', 'citeyear*',
}

TEXTUAL_CITATION_MACROS = {
    'citet', 'citealt', 'citeauthor', 'textcite',
}


def process_input_commands(latex_content, base_path, current_path=None):
    r"""
    Process \input{filename} commands by replacing them with file contents
    
    Args:
        latex_content: Latex content string
        
    Returns:
        str: Latex content with all \input commands resolved
    """
    # Pattern to match \input{filename} or \input{filename.tex}
    # Handles optional spaces and both with/without .tex extension
    pattern = re.compile(r'(?<!%)\\(?:input|include)\s*\{([^}]+)\}', re.MULTILINE)
    
    def replace_input(match):
        filename = match.group(1).strip()
        
        # Add .tex extension if not present
        if filename.startswith("\"") and filename.endswith("\""):
            filename = filename[1:-1]

        if all(x not in filename for x in ['.tex', '.bbl']):
            filename += '.tex'

        if filename.startswith("./"): filename = filename[2:]

        if "/" in filename:
            filename = filename.split("/")
        elif "\\" in filename:
            filename = filename.split("\\")
        else:
            filename = [filename]
        
        current_path = getattr(replace_input, "current_path", None) or base_path
        candidate_paths = [os.path.join(base_path, *filename)]
        current_candidate = os.path.join(current_path, *filename)
        if current_candidate not in candidate_paths:
            candidate_paths.append(current_candidate)
        filepath = next((path for path in candidate_paths if os.path.exists(path)), candidate_paths[0])
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            return process_input_commands(file_content, base_path, os.path.dirname(filepath))
        
        except FileNotFoundError:
            print(f"Warning: Could not find file '{filepath}' for \\input command")
            return f"% File not found: {filename}"
        
        except UnicodeDecodeError:
            try:
                file_content, _ = detect_encoding(filepath)
                return process_input_commands(file_content, base_path, os.path.dirname(filepath))
            except Exception as e:
                print(f"Warning: Error reading file '{filepath}': {e}")
                return f"% Error reading file: {filename}"
        
        except Exception as e:
            print(f"Warning: Error reading file '{filepath}': {e}")
            return f"% Error reading file: {filename}"
    
    # Replace all \input commands
    replace_input.current_path = current_path or base_path
    processed_content = re.sub(pattern, replace_input, latex_content)

    # Remove \href commands that would cause bugs
    processed_content = re.sub(r"\\href\s*\{[^\}]*\}\s*\{([^\}]*)\}", r"\1", processed_content)
    processed_content = re.sub(r"\\href\s*\{([^\}]*)\}", "", processed_content)
    
    return processed_content


class LatexPaperParser:
    """Parser to convert Latex source directories, files, or strings into Paper objects."""

    TEX_SECTION_RE = re.compile(
        r"\\(?P<level>section|subsection|subsubsection)\s*\{(?P<name>(?:[^{}]|\\[{}])*)\}",
        re.DOTALL,
    )
    TEX_SECTION_COMMAND_RE = re.compile(
        r"\\(?P<level>section|subsection|subsubsection)\s*(?P<star>\*)?\s*(?:\[[^\]]*\])?\s*\{",
        re.DOTALL,
    )
    TEX_APPENDIX_RE = re.compile(r"\\appendix\b|\\begin\s*\{\s*appendices\s*\}", re.IGNORECASE)
    
    def __init__(self):
        self.converter = LatexNodes2Text(math_mode="verbatim")
        self.section_levels = {
            'section': 1,
            'subsection': 2,
            'subsubsection': 3,
            'paragraph': 4,
            'subparagraph': 5
        }
        self.base_path = "."
        self.section_label_map = {}
        self.figure_table_label_map = {}
        self.latex_content = ""
        self.walker = LatexWalker("")
        self.bib_files = []
        self.bibliography_entries = {}
        self.unresolved_citation_keys = []
        self.citation_number_map = {}

    def _read_text_file(self, path: os.PathLike | str) -> str:
        path = os.fspath(path)
        for encoding in ("utf-8", "latin-1"):
            try:
                with open(path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        content, _ = detect_encoding(path)
        return content

    def _find_main_tex(self, source_dir: os.PathLike | str) -> str | None:
        source_dir = os.fspath(source_dir)
        tex_files = [path for path in glob.glob(os.path.join(source_dir, "**", "*.tex"), recursive=True) if os.path.isfile(path)]
        if not tex_files:
            return None

        scored = []
        preferred_names = {"main.tex", "ms.tex", "paper.tex", "article.tex", "root.tex", "uq_survey.tex"}
        for path in tex_files:
            try:
                content = self._read_text_file(path)
            except Exception:
                continue
            score = 0
            if "\\begin{document}" in content:
                score += 100
            if os.path.basename(path).lower() in preferred_names:
                score += 30
            if "\\documentclass" in content:
                score += 20
            if "\\section" in content:
                score += min(content.count("\\section") * 5, 40)
            score += min(len(content) // 2000, 20)
            scored.append((score, path))
        if not scored:
            return tex_files[0]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _as_existing_path(self, source: str | os.PathLike) -> str | None:
        if not isinstance(source, str):
            path = os.fspath(source)
        elif "\n" in source or "\\begin" in source:
            return None
        else:
            path = source
        try:
            return path if os.path.exists(path) else None
        except OSError:
            return None

    def _prepare_source(self, source: str | os.PathLike, base_path: str | os.PathLike | None = None) -> None:
        source_path = self._as_existing_path(source)
        if source_path and os.path.isdir(source_path):
            main_tex = self._find_main_tex(source_path)
            if not main_tex:
                raise FileNotFoundError(f"No TeX file found in {source_path}")
            latex_content = self._read_text_file(main_tex)
            self.base_path = os.path.dirname(main_tex)
        elif source_path and os.path.isfile(source_path):
            latex_content = self._read_text_file(source_path)
            self.base_path = os.path.dirname(source_path)
        else:
            latex_content = str(source or "")
            self.base_path = os.fspath(base_path or ".")

        processed_content = process_input_commands(latex_content, self.base_path)
        self.section_label_map = self._build_section_label_map(processed_content)
        self.figure_table_label_map = self._build_figure_table_label_map(processed_content)
        self.latex_content = self._remove_reference_macros(
            self._replace_labeled_refs(
                processed_content,
                {**self.section_label_map, **self.figure_table_label_map},
            )
        )
        self.walker = LatexWalker(self.latex_content)
        self.bib_files = []
        self.bibliography_entries = {}
        self.unresolved_citation_keys = []
        self.citation_number_map = {}
        self._collect_bibliography_files(processed_content)

    def _safe_nodes(self, nodes):
        return nodes or []

    def _collect_bibliography_files(self, content: str) -> None:
        for match in re.finditer(r"\\bibliography\s*\{([^{}]*)\}", content):
            for bib_name in match.group(1).split(","):
                bib_name = bib_name.strip()
                if not bib_name:
                    continue
                if not bib_name.endswith(".bib"):
                    bib_name = f"{bib_name}.bib"
                bib_path = os.path.join(self.base_path, bib_name)
                if os.path.exists(bib_path) and bib_path not in self.bib_files:
                    self.bib_files.append(bib_path)

    def _load_bibliography_entries(self) -> None:
        try:
            self.bibliography_entries = parse_bbl_file(self.latex_content)
        except Exception:
            self.bibliography_entries = {}
        for f in glob.glob(os.path.join(self.base_path, "*.bbl")):
            try:
                bib = parse_bbl_file(f)
                self.bibliography_entries.update(bib)
            except Exception as exc:
                print(f"Warning: Could not parse bibliography file '{f}': {exc}")
        if not self.bib_files:
            self.bib_files = glob.glob(os.path.join(self.base_path, "*.bib"))
        for f in self.bib_files:
            if "anthology" in f:
                continue
            try:
                bib = parse_bib_file(f)
                self.bibliography_entries.update(bib)
            except Exception as exc:
                print(f"Warning: Could not parse bibliography file '{f}': {exc}")

    def _build_section_label_map(self, content: str) -> dict[str, str]:
        content = self._strip_latex_comments(content)
        appendix = self.TEX_APPENDIX_RE.search(content)
        if appendix:
            content = content[:appendix.start()]

        headings = []
        counters = {"section": 0, "subsection": 0, "subsubsection": 0}
        for match in self.TEX_SECTION_COMMAND_RE.finditer(content):
            if match.group("star"):
                continue
            _, title_end = self._read_balanced_brace_content(content, match.end() - 1)
            if title_end == match.end() - 1:
                continue
            level = match.group("level")
            if level == "section":
                counters["section"] += 1
                counters["subsection"] = 0
                counters["subsubsection"] = 0
                section_index = str(counters["section"])
            elif level == "subsection":
                if counters["section"] == 0:
                    continue
                counters["subsection"] += 1
                counters["subsubsection"] = 0
                section_index = f"{counters['section']}.{counters['subsection']}"
            else:
                if counters["section"] == 0 or counters["subsection"] == 0:
                    continue
                counters["subsubsection"] += 1
                section_index = f"{counters['section']}.{counters['subsection']}.{counters['subsubsection']}"
            headings.append({"index": section_index, "start": match.start(), "content_start": title_end})

        label_map = {}
        for idx, heading in enumerate(headings):
            end = headings[idx + 1]["start"] if idx + 1 < len(headings) else len(content)
            chunk = content[heading["start"]:end]
            for label in re.findall(r"\\label\s*\{([^{}]+)\}", chunk):
                label_map[label.strip()] = heading["index"]
        return label_map

    def _build_figure_table_label_map(self, content: str) -> dict[str, str]:
        content = self._strip_latex_comments(content)
        label_map = {}
        counters = {"figure": 0, "table": 0}
        pattern = re.compile(
            r"\\begin\s*\{\s*(?P<env>figure\*?|table\*?|longtable)\s*\}",
            flags=re.DOTALL,
        )
        for match in pattern.finditer(content):
            env = match.group("env").rstrip("*")
            kind = "table" if env in {"table", "longtable"} else "figure"
            counters[kind] += 1
            end_match = re.search(
                rf"\\end\s*\{{\s*{re.escape(match.group('env'))}\s*\}}",
                content[match.end():],
                flags=re.DOTALL,
            )
            end = match.end() + end_match.end() if end_match else len(content)
            chunk = content[match.start():end]
            ref_text = f"{kind.title()} {counters[kind]}"
            for label in re.findall(r"\\label\s*\{([^{}]+)\}", chunk):
                label_map[label.strip()] = ref_text
        return label_map

    def _replace_labeled_refs(self, content: str, label_map: dict[str, str]) -> str:
        if not label_map:
            return content

        def replace(match):
            macro = match.group("macro")
            labels = [label.strip() for label in match.group("labels").split(",") if label.strip()]
            values = [label_map[label] for label in labels if label in label_map]
            if not values:
                return match.group(0)
            text = ", ".join(values)
            if all(re.match(r"^\d", value) for value in values):
                return f"Section {text}"
            return text

        return re.sub(
            r"\\(?P<macro>ref|autoref|cref|Cref)\s*\{(?P<labels>[^{}]+)\}",
            replace,
            content,
        )

    def _replace_section_refs(self, content: str, label_map: dict[str, str]) -> str:
        section_refs = {key: value for key, value in label_map.items()}
        return self._replace_labeled_refs(content, section_refs)

    def _macro_name(self, node) -> str:
        return getattr(node, "macroname", "") or ""

    def _is_macro(self, node, names: set[str]) -> bool:
        return isinstance(node, LatexMacroNode) and self._macro_name(node) in names

    def _collect_nodes_until(self, nodes, start_idx: int, stop_macros: set[str]):
        content = []
        idx = start_idx
        while idx < len(nodes):
            next_node = nodes[idx]
            if self._is_macro(next_node, stop_macros):
                break
            content.append(next_node)
            idx += 1
        return content, idx

    def _extract_preamble_macro_text(self, macro_name: str) -> str | None:
        pattern = re.compile(
            rf"\\{macro_name}\s*(?:\[[^\]]*\])?\s*\{{(?P<body>(?:[^{{}}]|\{{[^{{}}]*\}})*)\}}",
            flags=re.DOTALL,
        )
        match = pattern.search(self.latex_content)
        if not match:
            return None
        return self.converter.latex_to_text(match.group("body")).strip()

    def _node_latex(self, node) -> str:
        pos = getattr(node, "pos", None)
        length = getattr(node, "len", None)
        if pos is None or length is None:
            return ""
        return self.latex_content[pos:pos + length]

    def _macro_argument_latex(self, node) -> str:
        raw = self._node_latex(node)
        if raw:
            match = re.search(r"\{(?P<body>[^{}]*)\}\s*$", raw, flags=re.DOTALL)
            if match:
                return match.group("body")
        if node.nodeargd and node.nodeargd.argnlist:
            for arg in reversed(node.nodeargd.argnlist):
                if arg is not None:
                    raw_arg = self._node_latex(arg)
                    if raw_arg.startswith("{") and raw_arg.endswith("}"):
                        return raw_arg[1:-1]
                    text = self.converter.nodelist_to_text([arg]).strip()
                    if text:
                        return text
        return ""

    def _extract_citations_from_nodes(self, nodes) -> list[str]:
        citations = []

        def walk(items):
            for item in self._safe_nodes(items):
                if isinstance(item, LatexMacroNode):
                    if item.macroname in CITATION_MACROS:
                        citations.extend(self._extract_citation_keys(item))
                    if item.nodeargd and item.nodeargd.argnlist:
                        for arg in item.nodeargd.argnlist:
                            if hasattr(arg, "nodelist"):
                                walk(arg.nodelist)
                elif isinstance(item, LatexEnvironmentNode):
                    walk(item.nodelist)

        walk(nodes)
        return list(dict.fromkeys(citations))

    def _paragraph_name_from_macro(self, node) -> ParagraphName:
        title = self._extract_title(node)
        citations = []
        if node.nodeargd and node.nodeargd.argnlist:
            for arg in node.nodeargd.argnlist:
                if hasattr(arg, "nodelist"):
                    citations.extend(self._extract_citations_from_nodes(arg.nodelist))
        citations = list(dict.fromkeys(citations))
        title = re.sub(r"<cit\.>", " ", title)
        title = re.sub(r"\s+", " ", title).strip()
        return ParagraphName(text=title, citations=citations)

    def _is_limitation_title(self, title: str) -> bool:
        normalized = re.sub(r"[^a-z]+", " ", title.lower()).strip()
        return normalized in {"limitation", "limitations"}

    def _split_special_sections(self, sections: list[Section]) -> tuple[list[Section], list[Section], list[Section]]:
        body, limitation, appendix = [], [], []
        in_appendix = False
        for section in sections:
            title = section.name or ""
            if re.search(r"\\appendix\b|\\begin\s*\{\s*appendices\s*\}", title, flags=re.IGNORECASE):
                in_appendix = True
                continue
            if in_appendix:
                appendix.append(section)
            elif self._is_limitation_title(title):
                limitation.append(section)
            else:
                body.append(section)
        return body, limitation, appendix

    def _get_latex_nodes_quiet(self, text: str | None = None):
        sink = io.StringIO()
        walker = self.walker if text is None else LatexWalker(text)
        loggers = [
            logging.getLogger("pylatexenc"),
            logging.getLogger("pylatexenc.macrospec._environmentbodyparser"),
            logging.getLogger("pylatexenc.latexnodes.parsers._delimited"),
        ]
        previous_levels = [logger.level for logger in loggers]
        for logger in loggers:
            logger.setLevel(logging.ERROR)
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                return walker.get_latex_nodes()
        finally:
            for logger, level in zip(loggers, previous_levels):
                logger.setLevel(level)
   
    def parse(self, source: str | os.PathLike, base_path: str | os.PathLike | None = None) -> Optional[Paper]:
        """
        Single pass parse Latex content into a Paper object
        
        Returns:
            Paper: Structured paper object
        """
        self._prepare_source(source, base_path=base_path)
        self._load_bibliography_entries()
        paper = Paper()
        
        has_document = False
        nodelist, _, _ = self._get_latex_nodes_quiet()
        for node in self._safe_nodes(nodelist):
            if isinstance(node, LatexMacroNode):
                if node.macroname == 'title' and node.nodeargd and node.nodeargd.argnlist:
                    for arg in node.nodeargd.argnlist:
                        if arg is not None:
                            paper.title = self.converter.nodelist_to_text([arg]).strip()
                            break
                
                elif node.macroname == 'author' and node.nodeargd and node.nodeargd.argnlist:
                    for arg in node.nodeargd.argnlist:
                        if arg is not None:
                            paper.author = self.converter.nodelist_to_text([arg]).strip()
                            break
            
            elif isinstance(node, LatexEnvironmentNode):
                if node.environmentname == 'abstract':
                    paper.abstract = Section(name="Abstract")
                    self._create_paragraphs_from_nodes(node.nodelist, paper.abstract)

                elif node.environmentname == 'document':
                    has_document = True
                    paper.children, paper.limitation, paper.appendix, title, author, abstract = self._parse_sections(node.nodelist)
                    if not paper.children:
                        paper.children = self._parse_sections_fallback(node.nodelist)
                    if not paper.children:
                        paper.children = self._parse_sections_regex_fallback()
                    paper.children, extra_limitation, extra_appendix = self._split_special_sections(paper.children)
                    paper.limitation.extend(extra_limitation)
                    paper.appendix.extend(extra_appendix)
                    if title is not None: paper.title = title
                    if author is not None: paper.author = author
                    if abstract is not None: paper.abstract = abstract

        if not paper.title:
            paper.title = self._extract_preamble_macro_text("title")
        if not paper.author:
            paper.author = self._extract_preamble_macro_text("author")

        if not has_document:
            paper.children = self._parse_sections_regex_fallback()
            paper.children, paper.limitation, paper.appendix = self._split_special_sections(paper.children)
            if not paper.children:
                return
        
        paper.all_citation_keys = self._extract_all_citation_keys()
        paper.references = self._filter_bibliography_entries(paper.all_citation_keys)
        self.unresolved_citation_keys = [
            key for key in paper.all_citation_keys if key not in self.bibliography_entries
        ]
        paper.unresolved_citation_keys = self.unresolved_citation_keys
        return paper
    
    def _filter_bibliography_entries(self, citation_keys: list[str]) -> dict:
        cited_keys = set(citation_keys)
        return {
            key: value
            for key, value in self.bibliography_entries.items()
            if key in cited_keys
        }

    def get_bibliography_entry(self, citation_key: str) -> Optional[str]:
        """
        Retrieve the bibliography content for a given citation key
        
        Args:
            citation_key: The citation key to look up
            
        Returns:
            str: Bibliography entry text, or None if not found
        """
        return self.bibliography_entries.get(citation_key)
    
    def _parse_sections(self, nodes) -> tuple[list[Section], list[Section], list[Section], str | None, str | None, Any]:
        """Parse nodes into Section objects"""
        title, author, sections, limitation, appendix = None, None, [], [], []
        abstract = None
        nodes = self._safe_nodes(nodes)
        i = 0
        in_appendix = False
        
        while i < len(nodes):
            node = nodes[i]
            if isinstance(node, LatexMacroNode):            
                if node.macroname == "appendix":
                    in_appendix = True
                    i += 1
                elif self._is_macro(node, {'section'}):
                    section_name = self._extract_title(node)
                    section = Section(name=section_name)
                    
                    # Collect content until next section
                    section_content, j = self._collect_nodes_until(nodes, i + 1, {'section', 'appendix'})
                    
                    # Parse section content
                    self._parse_section_content(section_content, section)
                    if in_appendix:
                        appendix.append(section)
                    elif self._is_limitation_title(section_name):
                        limitation.append(section)
                    else:
                        sections.append(section)
                    i = j

                else:
                    i += 1
                    if node.macroname == 'title' and node.nodeargd and node.nodeargd.argnlist:
                        for arg in node.nodeargd.argnlist:
                            if arg is not None:
                                title = self.converter.nodelist_to_text([arg]).strip()
                
                    elif node.macroname == 'author' and node.nodeargd and node.nodeargd.argnlist:
                        for arg in node.nodeargd.argnlist:
                            if arg is not None:
                                author = self.converter.nodelist_to_text([arg]).strip()

            else:
                i += 1
                if isinstance(node, LatexEnvironmentNode): 
                    if node.environmentname == "abstract":
                        abstract = Section(name="Abstract")
                        self._create_paragraphs_from_nodes(node.nodelist, abstract)
                    elif node.environmentname == "appendices":
                        appendix_sections, _, nested_appendix, title_back, author_back, abstract_back = self._parse_sections(
                            self._safe_nodes(node.nodelist)
                        )
                        appendix.extend([*appendix_sections, *nested_appendix])
                        if title_back is not None: title = title_back
                        if author_back is not None: author = author_back
                        if abstract_back is not None: abstract = abstract_back
                    else:  # if node.environmentname in SPACING_ENVIRONMENTS
                        sections_in_environment, limitation_back, appendix_back, title_back, author_back, abstract_back = self._parse_sections(
                            self._safe_nodes(node.nodelist)
                        )
                        if title_back is not None: title = title_back
                        if author_back is not None: author = author_back
                        if abstract_back is not None: abstract = abstract_back
                        sections.extend(sections_in_environment)
                        limitation.extend(limitation_back)
                        appendix.extend(appendix_back)
        
        return sections, limitation, appendix, title, author, abstract

    def _parse_sections_fallback(self, nodes) -> List[Section]:
        """Fallback parser for TeX sources that expose only lower-level headings."""
        nodes = self._safe_nodes(nodes)
        heading_names = {'section'}
        if not any(self._is_macro(node, heading_names) for node in nodes):
            heading_names = {'subsection'}
        if not any(self._is_macro(node, heading_names) for node in nodes):
            heading_names = {'subsubsection'}
        if not any(self._is_macro(node, heading_names) for node in nodes):
            return []

        sections = []
        i = 0
        while i < len(nodes):
            node = nodes[i]
            if self._is_macro(node, heading_names):
                section = Section(name=self._extract_title(node))
                section_content, j = self._collect_nodes_until(nodes, i + 1, heading_names)
                self._parse_section_content(section_content, section)
                sections.append(section)
                i = j
            elif isinstance(node, LatexEnvironmentNode):
                sections.extend(self._parse_sections_fallback(node.nodelist))
                i += 1
            else:
                i += 1
        if sections:
            print(f"Latex fallback parser recovered {len(sections)} sections.")
        return sections

    def _parse_sections_regex_fallback(self) -> List[Section]:
        """Last-resort heading parser that keeps citations and heading hierarchy."""
        content = self.latex_content
        doc_match = re.search(r"\\begin\s*\{document\}(.+?)\\end\s*\{document\}", content, flags=re.DOTALL)
        if doc_match:
            content = doc_match.group(1)
        content = self._strip_latex_comments(content)
        heading_pattern = re.compile(
            r"\\(?P<level>section|subsection|subsubsection)\s*\*?\s*(?:\[[^\]]*\])?\s*\{",
            flags=re.DOTALL,
        )
        headings = []
        in_appendix = False
        for match in heading_pattern.finditer(content):
            appendix_match = self.TEX_APPENDIX_RE.search(content[:match.start()])
            if appendix_match:
                in_appendix = True
            title, title_end = self._read_balanced_brace_content(content, match.end() - 1)
            if title is None:
                continue
            headings.append(
                {
                    "level": match.group("level"),
                    "title": self.converter.latex_to_text(title).strip() or "Untitled",
                    "start": match.start(),
                    "content_start": title_end,
                    "appendix": in_appendix,
                }
            )
        if not headings:
            return []

        roots = []
        stack = []
        level_rank = {"section": 1, "subsection": 2, "subsubsection": 3}

        for idx, heading in enumerate(headings):
            heading["end"] = headings[idx + 1]["start"] if idx + 1 < len(headings) else len(content)
            node = {
                "level": heading["level"],
                "title": heading["title"],
                "content_start": heading["content_start"],
                "end": heading["end"],
                "appendix": heading["appendix"],
                "children": [],
            }
            while stack and level_rank[stack[-1]["level"]] >= level_rank[node["level"]]:
                stack.pop()
            if stack:
                stack[-1]["children"].append(node)
            else:
                roots.append(node)
            stack.append(node)

        def assign_content(node):
            content_end = min((child["content_start"] for child in node["children"]), default=node["end"])
            node["content"] = content[node["content_start"]:content_end]
            for child in node["children"]:
                assign_content(child)

        for root in roots:
            assign_content(root)

        def build(node):
            if node["level"] == "section":
                section = Section(name=node["title"])
                self._append_raw_content_paragraphs(node["content"], section)
                for child in node["children"]:
                    section.add_child(build(child))
                return section
            if node["level"] == "subsection":
                subsection = Section(name=node["title"])
                self._append_raw_content_paragraphs(node["content"], subsection)
                for child in node["children"]:
                    subsection.add_child(build(child))
                return subsection
            subsubsection = Section(name=node["title"])
            self._append_raw_content_paragraphs(node["content"], subsubsection)
            return subsubsection

        sections = [build(root) for root in roots]
        print(f"Latex regex fallback recovered {len(sections)} sections.")
        return sections

    def _strip_latex_comments(self, content: str) -> str:
        lines = []
        for line in content.splitlines(keepends=True):
            line_end_match = re.search(r"(\r?\n)$", line)
            line_end = line_end_match.group(1) if line_end_match else ""
            line_body = line[:-len(line_end)] if line_end else line
            escaped = False
            cut = len(line_body)
            for idx, char in enumerate(line_body):
                if char == "\\":
                    escaped = not escaped
                    continue
                if char == "%" and not escaped:
                    cut = idx
                    break
                escaped = False
            lines.append(line_body[:cut] + line_end)
        return "".join(lines)

    def _read_balanced_brace_content(self, content: str, open_pos: int) -> tuple[str | None, int]:
        if open_pos >= len(content) or content[open_pos] != "{":
            return None, open_pos
        depth = 0
        idx = open_pos
        while idx < len(content):
            char = content[idx]
            if char == "\\":
                idx += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return content[open_pos + 1:idx], idx + 1
            idx += 1
        return None, open_pos

    def _append_raw_content_paragraphs(self, raw_content: str, parent) -> None:
        raw_content = self._remove_heading_commands(raw_content).strip()
        if not raw_content:
            return
        parts = [part.strip() for part in re.split(r"\n\s*\n+", raw_content) if part.strip()]
        for part in parts:
            paragraph = self._build_fallback_paragraph(part)
            if paragraph and paragraph.sentences:
                parent.add_paragraph(paragraph)

    def _remove_heading_commands(self, content: str) -> str:
        content = re.sub(
            r"\\(?:section|subsection|subsubsection)\s*\*?\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}",
            "",
            content,
            flags=re.DOTALL,
        )
        return self._remove_reference_commands(content)

    def _remove_reference_commands(self, content: str) -> str:
        content = self._remove_reference_macros(content)
        content = re.sub(r"\\begin\s*\{\s*thebibliography\s*\}.*?\\end\s*\{\s*thebibliography\s*\}", " ", content, flags=re.DOTALL)
        return content

    def _remove_reference_macros(self, content: str) -> str:
        return re.sub(r"\\(?:bibstyle|bibliographystyle|bibliography|nocite)\s*\{[^{}]*\}", " ", content)

    def _build_fallback_paragraph(self, raw_content: str) -> Paragraph | None:
        raw_content = self._remove_reference_commands(raw_content)
        raw_content = re.sub(r"\\label\s*\{[^{}]*\}", "", raw_content)
        citation_markers = []

        def _replace_cite(match):
            keys = [key.strip() for key in match.group(1).split(",") if key.strip()]
            marker = f"CITMARK{len(citation_markers)}"
            citation_markers.append({"marker": marker, "keys": keys})
            return f" {marker} "

        cite_pattern = re.compile(r"\\(?:cite\w*|cite)\s*(?:\[[^\]]*\]\s*)*\{([^{}]+)\}")
        raw_content = re.sub(cite_pattern, _replace_cite, raw_content)
        raw_content = re.sub(r"\\(?:auto)?ref\s*\{[^{}]*\}", " REFMARK ", raw_content)
        raw_content = re.sub(r"\\paragraph\s*\*?\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}", r" PARAGRAPHMARK{\1} ", raw_content)
        raw_content = self._clean_fallback_latex_text(raw_content)
        text = self._rough_latex_to_text(raw_content)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return None

        paragraph = Paragraph()
        for sentence_text in self._split_into_sentences(text):
            sentence_keys = []
            for marker_info in citation_markers:
                if marker_info["marker"] in sentence_text:
                    sentence_keys.extend(marker_info["keys"])
                    sentence_text = sentence_text.replace(marker_info["marker"], " ")
            sentence_text = sentence_text.replace("REFMARK", "<ref>")
            sentence_text = re.sub(r"\s+", " ", sentence_text).strip()
            sentence_text = re.sub(r"\s+([,.;:!?])", r"\1", sentence_text)
            sentence_keys = list(dict.fromkeys(sentence_keys))
            if sentence_text:
                paragraph.add_sentence(Sentence(text=sentence_text, citations=sentence_keys))
        return paragraph if paragraph.sentences else None

    def _clean_fallback_latex_text(self, raw_content: str) -> str:
        raw_content = re.sub(r"\{\s*\\color\s*\{[^{}]*\}", "", raw_content)
        raw_content = re.sub(r"\\color\s*\{[^{}]*\}", "", raw_content)
        raw_content = re.sub(r"\\(?:textit|emph|textbf|texttt|textsc)\s*\{", "{", raw_content)
        raw_content = raw_content.replace("~", " ")
        raw_content = re.sub(r"(?m)^\s*[{}]\s*$", "", raw_content)
        return raw_content

    def _rough_latex_to_text(self, raw_content: str) -> str:
        text = raw_content
        text = re.sub(r"\\begin\s*\{[^{}]*\}", " ", text)
        text = re.sub(r"\\end\s*\{[^{}]*\}", " ", text)
        text = re.sub(r"\\(?:section|subsection|subsubsection)\s*\*?\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}", r"\1\n", text)
        text = re.sub(r"\\(?:textit|emph|textbf|texttt|textsc)\s*\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
        text = text.replace("{", "").replace("}", "")
        text = re.sub(r"\$+", " ", text)
        return text

    def _parse_raw_latex_nodes(self, content: str):
        wrapped = f"\\begin{{document}}\n{content}\n\\end{{document}}"
        try:
            nodes, _, _ = self._get_latex_nodes_quiet(wrapped)
        except Exception:
            return []
        docs = [
            node for node in nodes or []
            if isinstance(node, LatexEnvironmentNode) and node.environmentname == "document"
        ]
        if docs and docs[0].nodelist:
            return list(docs[0].nodelist)
        try:
            nodes, _, _ = self._get_latex_nodes_quiet(content)
            return list(nodes or [])
        except Exception:
            return []
    
    def _parse_section_content(self, nodes, parent_section: Section):
        """Parse content of a section (subsections and paragraphs)"""
        nodes = self._safe_nodes(nodes)
        i = 0
        current_text_nodes = []
        
        while i < len(nodes):
            node = nodes[i]
            
            if self._is_macro(node, {'subsection'}):
                # Save accumulated text as paragraphs
                if current_text_nodes:
                    self._create_paragraphs_from_nodes(current_text_nodes, parent_section)
                    current_text_nodes = []
                
                # Parse subsection
                subsection_name = self._extract_title(node)
                subsection = Section(name=subsection_name)
                
                # Collect subsection content
                subsection_content, j = self._collect_nodes_until(nodes, i + 1, {'subsection', 'section'})
                
                # Parse subsection content
                self._parse_subsection_content(subsection_content, subsection)
                parent_section.add_child(subsection)
                i = j
            else:                
                if not (isinstance(node, LatexMacroNode) and node.macroname in DELETE_MACROS):
                    current_text_nodes.append(node)
                if isinstance(node, LatexMacroNode) and node.macroname == "bibliography":
                    if node.nodeargd and node.nodeargd.argnlist:
                        for arg in node.nodeargd.argnlist:
                            if arg is not None:
                                bib_file = self.converter.nodelist_to_text([arg]).strip()
                                if not bib_file.endswith(".bib"): bib_file = f"{bib_file}.bib"
                                bib_path = os.path.join(self.base_path, bib_file)
                                if os.path.exists(bib_path): self.bib_files.append(bib_path)
                                break
                i += 1
        
        # Add remaining text
        if current_text_nodes:
            self._create_paragraphs_from_nodes(current_text_nodes, parent_section)
    
    def _parse_subsection_content(self, nodes, parent_subsection: Section):
        """Parse content of a subsection (subsubsections and paragraphs)"""
        nodes = self._safe_nodes(nodes)
        i = 0
        current_text_nodes = []
        
        while i < len(nodes):
            node = nodes[i]
            
            if self._is_macro(node, {'subsubsection'}):
                # Save accumulated text as paragraphs
                if current_text_nodes:
                    self._create_paragraphs_from_nodes(current_text_nodes, parent_subsection)
                    current_text_nodes = []
                
                # Parse subsubsection
                subsubsection_name = self._extract_title(node)
                
                # Collect subsubsection content
                subsubsection_content, j = self._collect_nodes_until(nodes, i + 1, {'subsubsection', 'subsection'})
                
                # Parse subsubsection content (paragraphs)
                # self._parse_subsubsection_content(subsubsection_content, subsubsection)
                subsubsection = Section(name=subsubsection_name)
                self._create_paragraphs_from_nodes(subsubsection_content, subsubsection)
                parent_subsection.add_child(subsubsection)
                i = j
            else:
                if not (isinstance(node, LatexMacroNode) and node.macroname in DELETE_MACROS):
                    current_text_nodes.append(node)
                i += 1

        if current_text_nodes:
            self._create_paragraphs_from_nodes(current_text_nodes, parent_subsection)
    
    def _create_paragraphs_from_nodes(self, nodes, parent) -> List[Paragraph]:
        """Create paragraph objects from text nodes, splitting by \n\n"""
        nodes = self._safe_nodes(nodes)
        sentences = self._parse_content_with_environments(nodes)
        paragraphs = self._group_contents_into_paragraphs(sentences)   
        for paragraph in paragraphs:
            parent.add_paragraph(paragraph)     
        return paragraphs
    
    def _parse_content_with_environments(self, nodes) -> List[Sentence]:
        """
        Parse nodes into a list of Sentences
        
        Args:
            nodes: List of LaTeX nodes
            
        Returns:
            list: List of Sentence objects
        """
        content_items, accumulated_nodes = [], []
        nodes = self._safe_nodes(nodes)
        
        # First pass: identify and extract preserved environments
        for node in nodes:            
            if isinstance(node, LatexEnvironmentNode):
                # Check if this environment should be preserved
                # if node.environmentname in PRESERVED_ENVIRONMENTS:
                    # Process accumulated text nodes first

                if node.environmentname == "thebibliography": continue
                
                if accumulated_nodes:
                    sentences = self._parse_text_with_citations_and_breaks(accumulated_nodes)
                    content_items.extend(sentences)
                    accumulated_nodes = []
                
                # Add the environment as-is
                env_content = self._extract_raw_environment(node)
                env_citations = self._extract_citations_from_environment(node)
                env_caption = self._extract_environment_caption(node)
                latex_env = Sentence(
                    text=env_content,
                    citations=env_citations,
                    environment_type=node.environmentname,
                    caption=env_caption,
                )
                content_items.append(latex_env)
            else:
                if isinstance(node, LatexMacroNode) and node.macroname == 'paragraph':
                    if accumulated_nodes:
                        sentences = self._parse_text_with_citations_and_breaks(accumulated_nodes)
                        content_items.extend(sentences)
                        accumulated_nodes = []
                    content_items.append(self._paragraph_name_from_macro(node))
                elif not (isinstance(node, LatexMacroNode) and node.macroname in DELETE_MACROS):
                    accumulated_nodes.append(node)
        
        # Process any remaining accumulated nodes
        if accumulated_nodes:
            sentences = self._parse_text_with_citations_and_breaks(accumulated_nodes)
            content_items.extend(sentences)
        
        return content_items
    
    def _extract_citations_from_environment(self, env_node: LatexEnvironmentNode) -> List[str]:
        """
        Extract all citation keys from an environment node
        
        Args:
            env_node: LatexEnvironmentNode to extract citations from
            
        Returns:
            list: List of citation keys found in the environment
        """
        citations = set()
        
        def find_citations_recursive(nodes):
            if nodes is None:
                return
            
            for node in nodes:
                if isinstance(node, LatexMacroNode):
                    if node.macroname in CITATION_MACROS:
                        citations.update(self._extract_citation_keys(node))
                    
                    # Also check in macro arguments
                    if node.nodeargd and node.nodeargd.argnlist:
                        for arg in node.nodeargd.argnlist:
                            if hasattr(arg, 'nodelist'):
                                find_citations_recursive(arg.nodelist)
                
                elif isinstance(node, LatexEnvironmentNode):
                    find_citations_recursive(node.nodelist)
        
        # Search in the environment's nodelist
        find_citations_recursive(env_node.nodelist)
        
        # Also check environment arguments (for cases like \begin{lemma}[Title \cite{key}])
        if hasattr(env_node, 'nodeargd') and env_node.nodeargd:
            if hasattr(env_node.nodeargd, 'argnlist') and env_node.nodeargd.argnlist:
                for arg in env_node.nodeargd.argnlist:
                    if hasattr(arg, 'nodelist'):
                        find_citations_recursive(arg.nodelist)
        
        return sorted(list(citations))
    
    def _extract_raw_environment(self, env_node: LatexEnvironmentNode) -> str:
        r"""
        Extract the raw LaTeX content of an environment
        
        Args:
            env_node: LatexEnvironmentNode to extract
            
        Returns:
            str: Raw LaTeX string including \begin and \end
        """
        env_name = env_node.environmentname
        
        # Get the content
        content = self.converter.nodelist_to_text(self._safe_nodes(env_node.nodelist))
        
        # Reconstruct the environment
        result = f"\\begin{{{env_name}}}\n{content}\n\\end{{{env_name}}}"
        
        return result

    def _extract_environment_caption(self, env_node: LatexEnvironmentNode) -> str:
        if env_node.environmentname not in GRAPH_ENVIRONMENTS:
            return ""
        raw = self._node_latex(env_node)
        if not raw:
            return ""
        match = re.search(r"\\caption\s*(?:\[[^\]]*\])?\s*\{", raw, flags=re.DOTALL)
        if not match:
            return ""
        body, _ = self._read_balanced_brace_content(raw, match.end() - 1)
        if body is None:
            return ""
        return re.sub(r"\s+", " ", self.converter.latex_to_text(body)).strip()

    def _parse_text_with_citations_and_breaks(self, nodes) -> List[Union[Sentence, dict]]:
        r"""
        Parse text nodes into sentences with paragraph break detection
        Returns list of Sentence objects and paragraph break markers
        
        Args:
            nodes: List of LaTeX nodes
            
        Returns:
            list: Sentence objects with special 'paragraph_break' markers
        """
        segments = self._extract_text_segments_with_breaks(nodes)
        if not segments:
            return []

        chunks = []
        current_chunk = []
        for segment in segments:
            if segment.get('paragraph_break'):
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
            else:
                current_chunk.append(segment)
        if current_chunk:
            chunks.append(current_chunk)

        sentences = []
        for chunk in chunks:
            chunk_sentences = self._parse_sentence_segments(chunk)
            if not chunk_sentences:
                continue
            if sentences:
                sentences.append({'paragraph_break': True})
            sentences.extend(chunk_sentences)

        return sentences

    def _parse_sentence_segments(self, segments) -> List[Sentence]:
        """Parse one LaTeX paragraph worth of text/citation segments into sentences."""
        full_text = ""
        citation_markers = []
        segments[0]['text'] = segments[0]['text'].lstrip()

        for segment in segments:
            if segment['citations']:
                marker = f"CITMARK{len(citation_markers)}"
                citation_markers.append({
                    "marker": marker,
                    "keys": segment["citations"],
                    "text": segment.get("citation_text") or self._format_citation_text(
                        segment["citations"], segment.get("citation_macro")
                    ),
                })
                full_text += f" {marker} "
            else:
                full_text += segment['text']

        # Split into sentences
        sentence_texts = self._split_into_sentences(full_text)

        # Assign citations to sentences
        sentences = []
        for sentence_text in sentence_texts:
            sentence_citations = []
            for marker_info in citation_markers:
                if marker_info["marker"] in sentence_text:
                    sentence_citations.extend(marker_info["keys"])
                    sentence_text = sentence_text.replace(marker_info["marker"], f" {marker_info['text']} ")

            # Remove duplicates while preserving order
            unique_citations = []
            for cite in sentence_citations:
                if cite not in unique_citations:
                    unique_citations.append(cite)
            citation_map = self._citation_number_dict(unique_citations)

            sentence_text = re.sub(r"\s+", " ", sentence_text).strip()
            sentence_text = re.sub(r"\s+([,.;:!?])", r"\1", sentence_text)
            sentence_text = self._normalize_rendered_citation_punctuation(sentence_text)
            if not sentence_text and not citation_map:
                continue
            sentence = Sentence(text=sentence_text, citations=citation_map)
            sentences.append(sentence)

        return sentences

    def _split_text_by_latex_paragraphs(self, text: str):
        """Split plain LaTeX chars by blank-line paragraph breaks after removing comments."""
        text = self._strip_latex_comments(text)
        if not text:
            return []

        segments = []
        pending_break = False
        parts = re.split(r"((?:[ \t]*\r?\n){2,})", text)
        for part in parts:
            if not part:
                continue
            if re.fullmatch(r"(?:[ \t]*\r?\n){2,}", part):
                pending_break = True
                continue
            if pending_break and segments:
                segments.append({'text': '\n', 'citations': [], 'paragraph_break': True})
            pending_break = False
            segments.append({'text': part, 'citations': [], 'paragraph_break': False})

        if pending_break:
            segments.append({'text': '\n', 'citations': [], 'paragraph_break': True})

        return segments

    def _citation_number(self, citation_key: str) -> int:
        if citation_key not in self.citation_number_map:
            self.citation_number_map[citation_key] = len(self.citation_number_map) + 1
        return self.citation_number_map[citation_key]

    def _citation_number_dict(self, citation_keys: list[str]) -> dict[int, str]:
        return {self._citation_number(key): key for key in citation_keys}

    def _format_citation_text(self, citation_keys: list[str], macro_name: str | None = None) -> str:
        if not citation_keys:
            return "?"
        numbers = [self._citation_number(key) for key in citation_keys]
        return f"[{', '.join(str(number) for number in numbers)}]"

    def _normalize_rendered_citation_punctuation(self, text: str) -> str:
        text = re.sub(r"\.\.", ".", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text

    def _extract_text_segments_with_breaks(self, nodes):
        """
        Extract text segments from nodes, identifying citations and paragraph breaks
        
        Args:
            nodes: List of LaTeX nodes
            segments: List to append segments to (modified in place)
        """
        if nodes is None:
            return []
        
        segments = []
        for node in nodes:
            if isinstance(node, LatexSpecialsNode):
                segments.extend(self._split_text_by_latex_paragraphs(self._node_latex(node)))
            elif isinstance(node, LatexCharsNode):
                segments.extend(self._split_text_by_latex_paragraphs(node.chars))
            
            elif isinstance(node, LatexMacroNode):
                if node.macroname in CITATION_MACROS:
                    citations = self._extract_citation_keys(node)
                    citation_text = self._format_citation_text(citations, node.macroname)
                    segments.append({
                        'text': citation_text,
                        'citations': citations,
                        'citation_macro': node.macroname,
                        'citation_text': citation_text,
                        'paragraph_break': False,
                    })
                elif node.macroname == 'par':
                    # Explicit paragraph break command
                    segments.append({'text': '\n', 'citations': [], 'paragraph_break': True})
                elif node.macroname not in DELETE_MACROS:
                    try:
                        text = self.converter.nodelist_to_text([node])
                        segments.extend(self._split_text_by_latex_paragraphs(text))
                    except:
                        if node.nodeargd and node.nodeargd.argnlist:
                            for arg in node.nodeargd.argnlist:
                                if hasattr(arg, 'nodelist'):
                                    segments.extend(self._extract_text_segments_with_breaks(arg.nodelist))
            
            elif isinstance(node, LatexEnvironmentNode) and node.environmentname not in PRESERVED_ENVIRONMENTS:
                segments.extend(self._extract_text_segments_with_breaks(self._safe_nodes(node.nodelist)))
            elif isinstance(node, LatexCommentNode):
                segments.extend(self._split_text_by_latex_paragraphs(getattr(node, "comment_post_space", "")))
            elif hasattr(node, "nodelist"):
                segments.extend(self._extract_text_segments_with_breaks(self._safe_nodes(node.nodelist)))
            else:
                try:
                    text = self.converter.nodelist_to_text([node])
                    segments.extend(self._split_text_by_latex_paragraphs(text))
                except:
                    pass
        
        return segments
 
    def _group_contents_into_paragraphs(self, content_items) -> List[Paragraph]:
        """
        Group content items (sentences and environments) into paragraphs
        Split by paragraph break markers
        
        Args:
            content_items: List of Sentence objects and break markers
            
        Returns:
            list: List of Paragraph objects
        """
        if not content_items:
            return []
        
        paragraphs = []
        current_paragraph = Paragraph()
        
        for item in content_items:
            if isinstance(item, dict) and item.get('paragraph_break'):
                # Paragraph break marker - finish current paragraph and start new one
                if current_paragraph.sentences:
                    paragraphs.append(current_paragraph)
                    current_paragraph = Paragraph()
                
            elif isinstance(item, Sentence) and item.environment_type in GRAPH_ENVIRONMENTS:
                if current_paragraph.sentences:
                    paragraphs.append(current_paragraph)
                    current_paragraph = Paragraph()
                current_paragraph.add_sentence(item)
                paragraphs.append(current_paragraph)
                current_paragraph = Paragraph()

            elif isinstance(item, ParagraphName):
                if current_paragraph.sentences:
                    paragraphs.append(current_paragraph)
                    current_paragraph = Paragraph()
                current_paragraph.add_sentence(item)
                
            elif isinstance(item, Sentence):
                current_paragraph.add_sentence(item)
        
        # Add final paragraph if not empty
        if current_paragraph.sentences:
            paragraphs.append(current_paragraph)
        
        return paragraphs if paragraphs else []
    
    def _extract_citation_keys(self, node):
        r"""
        Extract citation keys from a citation macro node
        Handles citations with optional arguments like \cite[prenote][postnote]{key}
        
        Args:
            node: LatexMacroNode for a citation command
            
        Returns:
            list: List of citation keys
        """
        raw_keys = self._macro_argument_latex(node)
        if raw_keys:
            keys = [re.sub(r"\s+", "", key) for key in raw_keys.split(",")]
            keys = [key for key in keys if key and self._looks_like_citation_key(key)]
            if keys:
                return list(dict.fromkeys(keys))

        citations = []
        if node.nodeargd and node.nodeargd.argnlist:
            # Citation commands can have optional arguments before the key
            # Format: \cite[prenote][postnote]{keys}
            # We want to extract only the mandatory argument (the keys)
            
            # The last non-None argument is typically the citation keys
            for arg in reversed(node.nodeargd.argnlist):
                if arg is not None:
                    # Check if this looks like citation keys (not optional text)
                    # Optional arguments usually contain text like page numbers, sections, etc.
                    # Citation keys are usually simple alphanumeric identifiers                    
                    cite_text = self.converter.nodelist_to_text([arg]).strip()                    
                    # If this is the mandatory argument with keys, it should be last
                    # and contain comma-separated citation keys
                    # We take the last non-None argument as the citation keys
                    keys = [k.strip() for k in cite_text.split(',') if k.strip()]                    
                    # Check if these look like citation keys (not optional notes)
                    # Citation keys typically don't contain spaces or special punctuation
                    if keys and all(self._looks_like_citation_key(k) for k in keys):
                        citations.extend(keys)
                        break
                    # If it doesn't look like citation keys, it might be postnote
                    # Continue to next argument
        
        return citations
    
    def _looks_like_citation_key(self, text):
        """
        Check if text looks like a citation key vs. optional note text
        Citation keys are typically alphanumeric with underscores, hyphens, colons
        
        Args:
            text: String to check
            
        Returns:
            bool: True if it looks like a citation key
        """
        # Citation keys typically:
        # - Don't start with special characters like \, -, etc.
        # - Don't contain spaces (or very few)
        # - Are relatively short
        # - Contain mostly alphanumeric chars, underscores, hyphens, colons
        
        if not text:
            return False
        
        # If it starts with Latex commands or special symbols, it's likely a note
        if text.startswith("\\"):
            return False
        
        # If it has multiple spaces, it's likely descriptive text
        if text.count(' ') > 2:
            return False
        
        # Check character composition
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-:.')
        text_chars = set(text.replace(' ', ''))
        
        # If most characters are allowed citation key characters, it's likely a key
        if len(text_chars - allowed_chars) / max(len(text_chars), 1) < 0.3:
            return True
        
        return False
    
    def _split_into_sentences(self, text):
        """Split text into sentences"""
        text = text.strip()
        if not text:
            return []

        candidates = self._scan_sentence_candidates(text)
        result = []
        for sentence in candidates:
            sentence = sentence.strip()
            if sentence:
                if result and self._starts_with_lowercase_ascii(sentence):
                    result[-1] = f"{result[-1].rstrip()} {sentence}"
                else:
                    result.append(sentence)
        return self._merge_short_sentences_forward(result)

    def _sentence_word_count(self, sentence: str) -> int:
        return len(re.findall(r"[A-Za-z0-9]+", sentence or ""))

    def _merge_short_sentences_forward(self, sentences: list[str]) -> list[str]:
        merged = []
        index = 0
        while index < len(sentences):
            sentence = sentences[index]
            if index + 1 < len(sentences) and self._sentence_word_count(sentence) <= 4:
                merged.append(f"{sentence.rstrip()} {sentences[index + 1].lstrip()}".strip())
                index += 2
            else:
                merged.append(sentence)
                index += 1
        return merged

    def _scan_sentence_candidates(self, text: str) -> list[str]:
        abbreviations = {
            "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "vs",
            "Fig", "Figs", "Sec", "Secs", "Eq", "Eqs", "Ref", "Refs",
            "Tab", "Tabs", "No", "Vol", "Inc", "Ltd", "Co",
        }
        open_to_close = {"(": ")", "[": "]", "{": "}"}
        close_chars = set(open_to_close.values())
        stack = []
        sentences = []
        start = 0
        i = 0
        while i < len(text):
            char = text[i]
            if char in open_to_close:
                stack.append(open_to_close[char])
            elif char in close_chars and stack and char == stack[-1]:
                stack.pop()

            if char in ".!?" and not stack and self._is_sentence_boundary(text, i, abbreviations):
                end = i + 1
                while end < len(text) and text[end] in ".!?":
                    end += 1
                while end < len(text) and text[end] in "\"')]}":
                    end += 1
                sentences.append(text[start:end])
                start = end
                while start < len(text) and text[start].isspace():
                    start += 1
                i = start
                continue
            i += 1

        if start < len(text):
            sentences.append(text[start:])
        return sentences

    def _is_sentence_boundary(self, text: str, idx: int, abbreviations: set[str]) -> bool:
        char = text[idx]
        if char in "!?":
            return True
        if text[idx:idx + 3] == "...":
            return True
        if idx > 0 and idx + 1 < len(text) and text[idx - 1].isalpha() and text[idx + 1].isalpha():
            return False

        word_match = re.search(r"([A-Za-z]+)$", text[:idx])
        word = word_match.group(1) if word_match else ""
        if word in {"e", "i", "g"} and self._is_part_of_latin_abbreviation(text, idx):
            return False
        if word == "al" and re.search(r"\bet\s+al$", text[:idx]):
            return False
        if word == "etc":
            return idx + 1 >= len(text) or text[idx + 1].isspace()
        if word in abbreviations:
            return False

        return idx + 1 >= len(text) or text[idx + 1].isspace() or text[idx + 1] in "\"')]}"

    def _is_part_of_latin_abbreviation(self, text: str, idx: int) -> bool:
        window = text[max(0, idx - 3):idx + 3].lower()
        return "e.g." in window or "i.e." in window

    def _starts_with_lowercase_ascii(self, text: str) -> bool:
        stripped = text.lstrip()
        return bool(stripped) and "a" <= stripped[0] <= "z"
    
    def _extract_title(self, node):
        """Extract title from a section/subsection macro node"""
        if node.nodeargd and node.nodeargd.argnlist:
            for arg in reversed(node.nodeargd.argnlist):
                if arg is not None:
                    return self.converter.nodelist_to_text([arg]).strip()
        return "Untitled"
    
    def _extract_all_citation_keys(self):
        """Extract all unique citations"""
        citations = set()
        nodelist, _, _ = self._get_latex_nodes_quiet()
        
        def find_citations(nodes):
            if nodes is None:
                return
            
            for node in nodes:
                if isinstance(node, LatexMacroNode):
                    if node.macroname in CITATION_MACROS:
                        citations.update(self._extract_citation_keys(node))
                
                if isinstance(node, LatexEnvironmentNode):
                    if node.environmentname == 'thebibliography':
                        continue
                    find_citations(node.nodelist)
                elif isinstance(node, LatexMacroNode) and node.nodeargd:
                    if node.nodeargd.argnlist:
                        for arg in node.nodeargd.argnlist:
                            if hasattr(arg, 'nodelist'):
                                find_citations(arg.nodelist)
        
        find_citations(nodelist)
        return sorted(list(citations))
    
    def _clean_title(self, value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip()

    def _head_record(self, section_index: str, section_name: str) -> dict[str, str] | None:
        section_index = self._clean_title(section_index)
        section_name = self._clean_title(section_name)
        if not section_name: return None
        return {"section_index": section_index, "section_name": section_name}
    
    def get_titles(self, source: str | os.PathLike | None = None, base_path: str | os.PathLike | None = None):
        if source is not None:
            self._prepare_source(source, base_path=base_path)
        content = self.latex_content
        appendix = self.TEX_APPENDIX_RE.search(content)
        if appendix: content = content[:appendix.start()]

        counters = {"section": 0, "subsection": 0, "subsubsection": 0}
        records = []
        for match in self.TEX_SECTION_COMMAND_RE.finditer(content):
            if match.group("star"):
                continue
            name, title_end = self._read_balanced_brace_content(content, match.end() - 1)
            if name is None or title_end == match.end() - 1:
                continue
            level = match.group("level")
            if level == "section":
                counters["section"] += 1
                counters["subsection"] = 0
                counters["subsubsection"] = 0
                section_index = str(counters["section"])
            elif level == "subsection":
                if counters["section"] == 0:
                    continue
                counters["subsection"] += 1
                counters["subsubsection"] = 0
                section_index = f"{counters['section']}.{counters['subsection']}"
            else:
                if counters["section"] == 0 or counters["subsection"] == 0:
                    continue
                counters["subsubsection"] += 1
                section_index = f"{counters['section']}.{counters['subsection']}.{counters['subsubsection']}"

            name = self._clean_title(name.replace(r"\{", "{").replace(r"\}", "}"))
            record = self._head_record(section_index, name)
            if record: records.append(record)
        return records
