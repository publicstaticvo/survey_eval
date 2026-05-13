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
from typing import List, Any
from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexMacroNode, LatexCharsNode, LatexCommentNode
from .paper_elements import *
from .constants import *
from .bib_parser import parse_bbl_file, parse_bib_file, detect_encoding


SECTION_MACROS = {'section', 'subsection', 'subsubsection'}


def process_input_commands(latex_content, base_path):
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

        if "/" in filename:
            filename = filename.split("/")
        elif "\\" in filename:
            filename = filename.split("\\")
        else:
            filename = [filename]
        
        # Construct full path
        filepath = os.path.join(base_path, *filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Recursively process \input commands in the included file
            return process_input_commands(file_content, os.path.dirname(filepath))
        
        except FileNotFoundError:
            print(f"Warning: Could not find file '{filepath}' for \\input command")
            return f"% File not found: {filename}"
        
        except UnicodeDecodeError:
            try:
                file_content, _ = detect_encoding(filepath)
                return process_input_commands(file_content, os.path.dirname(filepath))
            except Exception as e:
                print(f"Warning: Error reading file '{filepath}': {e}")
                return f"% Error reading file: {filename}"
        
        except Exception as e:
            print(f"Warning: Error reading file '{filepath}': {e}")
            return f"% Error reading file: {filename}"
    
    # Replace all \input commands
    processed_content = re.sub(pattern, replace_input, latex_content)

    # Remove \href commands that would cause bugs
    processed_content = re.sub(r"\\href\s*\{[^\}]*\}\s*\{([^\}]*)\}", r"\1", processed_content)
    processed_content = re.sub(r"\\href\s*\{([^\}]*)\}", "", processed_content)
    
    return processed_content


class LatexPaperParser:
    """Parser to convert Latex documents into Paper objects"""

    TEX_SECTION_RE = re.compile(
        r"\\(?P<level>section|subsection|subsubsection)\s*\{(?P<name>(?:[^{}]|\\[{}])*)\}",
        re.DOTALL,
    )
    TEX_APPENDIX_RE = re.compile(r"\\appendix\b|\\begin\s*\{\s*appendices\s*\}", re.IGNORECASE)
    
    def __init__(self, latex_content: str, base_path='.'):
        self.base_path = base_path
        self.latex_content = process_input_commands(latex_content, base_path)
        self.walker = LatexWalker(self.latex_content)
        self.converter = LatexNodes2Text(math_mode="verbatim")
        self.section_levels = {
            'section': 1,
            'subsection': 2,
            'subsubsection': 3,
            'paragraph': 4,
            'subparagraph': 5
        }
        # Store bibliography entries
        self.bib_files = []
        self.bibliography_entries = {}

    def _safe_nodes(self, nodes):
        return nodes or []

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
   
    def parse(self) -> Optional[LatexPaper]:
        """
        Single pass parse Latex content into a Paper object
        
        Returns:
            Paper: Structured paper object
        """
        paper = LatexPaper()
        
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
                    paper.abstract = LatexSubSubSection(name="Abstract")
                    self._create_paragraphs_from_nodes(node.nodelist, paper.abstract)

                elif node.environmentname == 'document':
                    has_document = True
                    paper.sections, title, author, abstract = self._parse_sections(node.nodelist)
                    if not paper.sections:
                        paper.sections = self._parse_sections_fallback(node.nodelist)
                    if not paper.sections:
                        paper.sections = self._parse_sections_regex_fallback()
                    if title is not None: paper.title = title
                    if author is not None: paper.author = author
                    if abstract is not None: paper.abstract = abstract

        if not paper.title:
            paper.title = self._extract_preamble_macro_text("title")
        if not paper.author:
            paper.author = self._extract_preamble_macro_text("author")

        if not has_document:
            paper.sections = self._parse_sections_regex_fallback()
            if not paper.sections:
                return
        
        # parse citations
        self.bibliography_entries = parse_bbl_file(self.latex_content)
        for f in glob.glob(os.path.join(self.base_path, "*.bbl")):
            bib = parse_bbl_file(f)
            self.bibliography_entries.update(bib)
        if not self.bib_files: self.bib_files = glob.glob(os.path.join(self.base_path, "*.bib"))
        for f in self.bib_files:
            if "anthology" in f: continue
            bib = parse_bib_file(f)
            self.bibliography_entries.update(bib)
        paper.bibliography = self.bibliography_entries
        paper.all_citation_keys = self._extract_all_citation_keys()
        return paper
    
    def get_bibliography_entry(self, citation_key: str) -> Optional[str]:
        """
        Retrieve the bibliography content for a given citation key
        
        Args:
            citation_key: The citation key to look up
            
        Returns:
            str: Bibliography entry text, or None if not found
        """
        return self.bibliography_entries.get(citation_key)
    
    def _parse_sections(self, nodes) -> List[LatexSection]:
        """Parse nodes into Section objects"""
        title, author, sections = None, None, []
        abstract = None
        nodes = self._safe_nodes(nodes)
        i = 0
        
        while i < len(nodes):
            node = nodes[i]
            if isinstance(node, LatexMacroNode):            
                if self._is_macro(node, {'section'}):
                    section_name = self._extract_title(node)
                    section = LatexSection(name=section_name)
                    
                    # Collect content until next section
                    section_content, j = self._collect_nodes_until(nodes, i + 1, {'section'})
                    
                    # Parse section content
                    self._parse_section_content(section_content, section)
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
                        abstract = LatexSubSubSection(name="Abstract")
                        self._create_paragraphs_from_nodes(node.nodelist, abstract)
                    else:  # if node.environmentname in SPACING_ENVIRONMENTS
                        sections_in_environment, title_back, author_back, abstract_back = self._parse_sections(
                            self._safe_nodes(node.nodelist)
                        )
                        if title_back is not None: title = title_back
                        if author_back is not None: author = author_back
                        if abstract_back is not None: abstract = abstract_back
                        sections.extend(sections_in_environment)
        
        return sections, title, author, abstract

    def _parse_sections_fallback(self, nodes) -> List[LatexSection]:
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
                section = LatexSection(name=self._extract_title(node))
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

    def _parse_sections_regex_fallback(self) -> List[LatexSection]:
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
        for match in heading_pattern.finditer(content):
            title, title_end = self._read_balanced_brace_content(content, match.end() - 1)
            if title is None:
                continue
            headings.append(
                {
                    "level": match.group("level"),
                    "title": self.converter.latex_to_text(title).strip() or "Untitled",
                    "start": match.start(),
                    "content_start": title_end,
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
                section = LatexSection(name=node["title"])
                self._append_raw_content_paragraphs(node["content"], section)
                for child in node["children"]:
                    section.add_child(build(child))
                return section
            if node["level"] == "subsection":
                subsection = LatexSubSection(name=node["title"])
                self._append_raw_content_paragraphs(node["content"], subsection)
                for child in node["children"]:
                    subsection.add_child(build(child))
                return subsection
            subsubsection = LatexSubSubSection(name=node["title"])
            self._append_raw_content_paragraphs(node["content"], subsubsection)
            return subsubsection

        sections = [build(root) for root in roots]
        print(f"Latex regex fallback recovered {len(sections)} sections.")
        return sections

    def _strip_latex_comments(self, content: str) -> str:
        lines = []
        for line in content.splitlines():
            escaped = False
            cut = len(line)
            for idx, char in enumerate(line):
                if char == "\\":
                    escaped = not escaped
                    continue
                if char == "%" and not escaped:
                    cut = idx
                    break
                escaped = False
            lines.append(line[:cut])
        return "\n".join(lines)

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
                parent.add_child(paragraph)

    def _remove_heading_commands(self, content: str) -> str:
        return re.sub(
            r"\\(?:section|subsection|subsubsection)\s*\*?\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}",
            "",
            content,
            flags=re.DOTALL,
        )

    def _build_fallback_paragraph(self, raw_content: str) -> LatexParagraph | None:
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
        raw_content = self._clean_fallback_latex_text(raw_content)
        text = self.converter.latex_to_text(raw_content)
        if not text.strip() and raw_content.strip():
            text = self._rough_latex_to_text(raw_content)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return None

        paragraph = LatexParagraph()
        for sentence_text in self._split_into_sentences(text):
            sentence_keys = []
            for marker_info in citation_markers:
                if marker_info["marker"] in sentence_text:
                    sentence_keys.extend(marker_info["keys"])
                    sentence_text = sentence_text.replace(marker_info["marker"], "<cit.>")
            sentence_text = sentence_text.replace("REFMARK", "<ref>")
            sentence_text = re.sub(r"\s+", " ", sentence_text).strip()
            sentence_text = re.sub(r"\s+([,.;:!?])", r"\1", sentence_text)
            sentence_keys = list(dict.fromkeys(sentence_keys))
            if sentence_text:
                paragraph.add_sentence(LatexSentence(text=sentence_text, citations=sentence_keys))
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
    
    def _parse_section_content(self, nodes, parent_section: LatexSection):
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
                subsection = LatexSubSection(name=subsection_name)
                
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
    
    def _parse_subsection_content(self, nodes, parent_subsection: LatexSubSection):
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
                subsubsection = LatexSubSubSection(name=subsubsection_name)
                self._create_paragraphs_from_nodes(subsubsection_content, subsubsection)
                parent_subsection.add_child(subsubsection)
                i = j
            else:
                if not (isinstance(node, LatexMacroNode) and node.macroname == 'label'):
                    current_text_nodes.append(node)
                i += 1

        if current_text_nodes:
            self._create_paragraphs_from_nodes(current_text_nodes, parent_subsection)
    
    def _create_paragraphs_from_nodes(self, nodes, parent) -> List[LatexParagraph]:
        """Create paragraph objects from text nodes, splitting by \n\n"""
        nodes = self._safe_nodes(nodes)
        sentences = self._parse_content_with_environments(nodes)
        paragraphs = self._group_contents_into_paragraphs(sentences)   
        for paragraph in paragraphs:
            parent.add_child(paragraph)     
        return paragraphs
    
    def _parse_content_with_environments(self, nodes) -> List[Union[LatexSentence, LatexEnvironment]]:
        """
        Parse nodes into a list of Sentences and LatexEnvironment objects
        
        Args:
            nodes: List of LaTeX nodes
            
        Returns:
            list: List of Sentence and LatexEnvironment objects
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
                latex_env = LatexEnvironment(
                    environment_name=node.environmentname, 
                    text=env_content,
                    citations=env_citations
                )
                content_items.append(latex_env)
            else:
                if not (isinstance(node, LatexMacroNode) and node.macroname == 'label'):
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
                    if node.macroname in ['cite', 'citep', 'citet', 'citealt', 'citeyearpar',
                                          'citealp', 'citeauthor', 'citeyear', 'citetext']:
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

    def _parse_text_with_citations_and_breaks(self, nodes) -> List[Union[LatexSentence, dict]]:
        r"""
        Parse text nodes into sentences with paragraph break detection
        Returns list of Sentence objects and paragraph break markers
        
        Args:
            nodes: List of LaTeX nodes
            
        Returns:
            list: Sentence objects with special 'paragraph_break' markers
        """
        # Extract text segments with citation markers
        segments = self._extract_text_segments_with_breaks(nodes)
        if not segments: return []
        
        # Combine segments into full text while tracking citation positions and breaks
        full_text = ""
        citation_positions = []
        paragraph_break_positions = []
        segments[0]['text'] = segments[0]['text'].lstrip()
        
        for segment in segments:
            start_pos = len(full_text)
            full_text += segment['text']
            end_pos = len(full_text)
            
            if segment['citations']:
                citation_positions.append((start_pos, end_pos, segment['citations']))
            
            if segment.get('paragraph_break'):
                paragraph_break_positions.append(end_pos)
        
        # Split into sentences
        sentence_texts = self._split_into_sentences(full_text)        
        
        # Assign citations to sentences and detect paragraph breaks
        sentences = []
        char_pos = 0
        
        for sentence_text in sentence_texts:
            sentence_start = char_pos
            sentence_end = char_pos + len(sentence_text)
            
            # Find citations that overlap with this sentence
            sentence_citations = []
            for cite_start, cite_end, citations in citation_positions:
                if cite_start < sentence_end and cite_end > sentence_start:
                    sentence_citations.extend(citations)
            
            # Remove duplicates while preserving order
            unique_citations = []
            for cite in sentence_citations:
                if cite not in unique_citations:
                    unique_citations.append(cite)
            
            sentence = LatexSentence(text=sentence_text.strip(), citations=unique_citations)
            sentences.append(sentence)
            
            # Check if there's a paragraph break after this sentence
            for break_pos in paragraph_break_positions:
                if sentence_start < break_pos <= sentence_end:
                    # Mark paragraph break by inserting a marker
                    sentences.append({'paragraph_break': True})
                    break
            
            char_pos = sentence_end
        
        return sentences
    
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
            if isinstance(node, LatexCharsNode):
                # Check for paragraph breaks (multiple newlines)
                text = node.chars
                if '\n\n' in text or '\n\n\n' in text:
                    # Split by paragraph breaks
                    parts = re.split(r'\n\n+', text)
                    for i, part in enumerate(parts):
                        part = part.strip()
                        if part:
                            segments.append({'text': part, 'citations': [], 'paragraph_break': False})
                            if i < len(parts) - 1:  # Not the last part
                                segments.append({'text': '\n', 'citations': [], 'paragraph_break': True})
                else:
                    segments.append({'text': text, 'citations': [], 'paragraph_break': False})
            
            elif isinstance(node, LatexMacroNode):
                if node.macroname in ['cite', 'citep', 'citet', 'citealt', "citeyearpar",
                                      'citealp', 'citeauthor', 'citeyear', 'citetext']:
                    citations = self._extract_citation_keys(node)
                    citation_text = f' \\cite{{{",".join(citations)}}} '
                    segments.append({'text': citation_text, 'citations': citations, 'paragraph_break': False})
                elif node.macroname == 'par':
                    # Explicit paragraph break command
                    segments.append({'text': '\n', 'citations': [], 'paragraph_break': True})
                elif node.macroname not in DELETE_MACROS:
                    try:
                        text = self.converter.nodelist_to_text([node])
                        segments.append({'text': text, 'citations': [], 'paragraph_break': False})
                    except:
                        if node.nodeargd and node.nodeargd.argnlist:
                            for arg in node.nodeargd.argnlist:
                                if hasattr(arg, 'nodelist'):
                                    segments.extend(self._extract_text_segments_with_breaks(arg.nodelist))
            
            elif isinstance(node, LatexEnvironmentNode) and node.environmentname not in PRESERVED_ENVIRONMENTS:
                segments.extend(self._extract_text_segments_with_breaks(self._safe_nodes(node.nodelist)))
            else:
                try:
                    text = self.converter.nodelist_to_text([node])
                    segments.append({'text': text, 'citations': [], 'paragraph_break': False})
                except:
                    pass
        
        return segments
 
    def _group_contents_into_paragraphs(self, content_items) -> List[LatexParagraph]:
        """
        Group content items (sentences and environments) into paragraphs
        Split by paragraph break markers
        
        Args:
            content_items: List of Sentence, LatexEnvironment objects, and break markers
            
        Returns:
            list: List of Paragraph objects
        """
        if not content_items:
            return []
        
        paragraphs = []
        current_paragraph = LatexParagraph()
        
        for item in content_items:
            if isinstance(item, dict) and item.get('paragraph_break'):
                # Paragraph break marker - finish current paragraph and start new one
                if current_paragraph.sentences:
                    paragraphs.append(current_paragraph)
                    current_paragraph = LatexParagraph()
                
            elif isinstance(item, LatexEnvironment) and item.environment_name in GRAPH_ENVIRONMENTS:
                if current_paragraph.sentences:
                    paragraphs.append(current_paragraph)
                    current_paragraph = LatexParagraph()
                current_paragraph.add_sentence(item)
                paragraphs.append(current_paragraph)
                current_paragraph = LatexParagraph()
                
            elif isinstance(item, LatexSentence) or isinstance(item, LatexEnvironment):
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
        # - Don't start with special characters like \, §, etc.
        # - Don't contain spaces (or very few)
        # - Are relatively short
        # - Contain mostly alphanumeric chars, underscores, hyphens, colons
        
        if not text:
            return False
        
        # If it starts with Latex commands or special symbols, it's likely a note
        if text.startswith('\\') or text.startswith('§'):
            return False
        
        # If it has multiple spaces, it's likely descriptive text
        if text.count(' ') > 2:
            return False
        
        # If it's very long, it's likely a note
        if len(text) > 50:
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
        
        # Handle abbreviations
        abbreviations = [
            "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "vs", "etc",
            "Fig", "Figs", "Sec", "Secs", "Eq", "Eqs", "Ref", "Refs",
            "Tab", "Tabs", "No", "Vol", "Inc", "Ltd", "Co",
        ]
        for abbr in abbreviations:
            text = re.sub(rf'\b{re.escape(abbr)}\.\s', f'{abbr}<PERIOD> ', text)
        text = re.sub(r'\be\.g\.\s', r'e<PERIOD>g<PERIOD> ', text)
        text = re.sub(r'\bi\.e\.\s', r'i<PERIOD>e<PERIOD> ', text)
        text = re.sub(r'\bet al\.\s', r'et al<PERIOD> ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'([.!?]+(?:\s+|$)|\.\.\.(?:\s+|$))', text)
        
        # Recombine sentences with punctuation
        result = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i + 1].strip():
                sentence = sentences[i] + sentences[i + 1]
                i += 2
            else:
                sentence = sentences[i]
                i += 1
            
            sentence = sentence.replace('<PERIOD>', '.')
            if sentence:
                result.append(sentence)
        
        return result
    
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
                    if node.macroname in ['cite', 'citep', 'citet', 'citealt', 'citeyearpar',
                                          'citealp', 'citeauthor', 'citeyear', 'citetext']:
                        citations.update(self._extract_citation_keys(node))
                
                if isinstance(node, LatexEnvironmentNode):
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
    
    def get_titles(self):
        content = self.latex_content
        appendix = self.TEX_APPENDIX_RE.search(content)
        if appendix: content = content[:appendix.start()]

        counters = {"section": 0, "subsection": 0, "subsubsection": 0}
        records = []
        for match in self.TEX_SECTION_RE.finditer(content):
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

            name = self._clean_title(match.group("name").replace(r"\{", "{").replace(r"\}", "}"))
            record = self._head_record(section_index, name)
            if record: records.append(record)
        return records


def construct_citation_info(paper: LatexPaper, parser: LatexPaperParser) -> List[Dict[str, Any]]:

    def get_citation_info_in_paragraph(paragraph: LatexParagraph):
        sentences = []
        for i, sentence in enumerate(paragraph.sentences):
            if sentence.citations:
                citation_key_value = {}
                for citation in sentence.citations:
                    citation_value = parser.get_bibliography_entry(citation)
                    if citation_value: citation_key_value[citation] = citation_value
                    else: 
                        citation_key_value = {}
                        break
                if citation_key_value:
                    sentences.append({
                        "text": sentence.text, 
                        "citation": citation_key_value, 
                        "serial": " ".join(paragraph.get_next_sentence_until_citation(i, 3))
                    })
        return sentences
    
    all_citation_info = []
    # abstract
    if paper.abstract:
        for i, paragraph in enumerate(paper.abstract.children):
            citation_info = get_citation_info_in_paragraph(paragraph)
            for x in citation_info: x['section_id'] = f"0-{i + 1}"
            all_citation_info.extend(citation_info)

    for i, section in enumerate(paper.sections):
        subsection_start_idx = -1
        for j, subsection in enumerate(section.children):
            if isinstance(subsection, LatexParagraph):
                assert subsection_start_idx == -1
                citation_info = get_citation_info_in_paragraph(subsection)
                for x in citation_info: x['section_id'] = f"{i + 1}-{j + 1}"
                all_citation_info.extend(citation_info)
            else:
                if subsection_start_idx == -1: subsection_start_idx = j
                subsubsection_start_idx = -1
                for k, subsubsection in enumerate(subsection.children):
                    if isinstance(subsubsection, LatexParagraph):
                        assert subsubsection_start_idx == -1
                        citation_info = get_citation_info_in_paragraph(subsubsection)
                        chapter_str = f"{i + 1}.{j - subsection_start_idx + 1}-{k + 1}"
                        for x in citation_info: x['section_id'] = chapter_str
                        all_citation_info.extend(citation_info)
                    else:
                        if subsubsection_start_idx == -1: subsubsection_start_idx = k
                        for l, paragraph in enumerate(subsubsection.children):
                            citation_info = get_citation_info_in_paragraph(paragraph)
                            chapter_str = f"{i + 1}.{j - subsection_start_idx + 1}.{k - subsubsection_start_idx + 1}-{l + 1}"
                            for x in citation_info: x['section_id'] = chapter_str
                            all_citation_info.extend(citation_info)
    return all_citation_info
