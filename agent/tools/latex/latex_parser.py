"""
LaTeX paper parser to extract structure and content into Paper dataclass format.
Handles sections, subsections, references, \\input commands, and various environments.

This parser uses both regex for structure parsing and pylatexenc for advanced LaTeX handling.
"""

from __future__ import annotations
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum

try:
    from pylatexenc.latex2text import LatexNodes2Text
    from pylatexenc.latexnodes.nodes import LatexGroupNode, LatexMacroNode, LatexEnvironmentNode, LatexNode
    from pylatexenc.macrospec import MacroSpec
    from pylatexenc.latexwalker import LatexWalker
except ImportError:
    raise ImportError("Please install pylatexenc: pip install pylatexenc")


class EnvironmentType(Enum):
    """Types of LaTeX environments"""
    TEXT = "text"
    EQUATION = "equation"
    FIGURE = "figure"
    TABLE = "table"
    ALGORITHM = "algorithm"
    LISTING = "listing"
    VERBATIM = "verbatim"
    THEOREM = "theorem"
    PROOF = "proof"
    UNKNOWN = "unknown"


@dataclass
class Sentence:
    """Represents a single sentence with its citations"""
    text: str = ""  # without citations
    father: Optional[Paragraph] = None
    citations: List[str] = field(default_factory=list)
    environment_type: EnvironmentType = EnvironmentType.TEXT
    
    def __str__(self):
        cite_str = f" [{', '.join(self.citations)}]" if self.citations else ""
        return f"{self.text} {cite_str}"
    
    def get_skeleton(self) -> Dict[str, Union[str, List[str], str]]:
        return {
            "text": self.text, 
            "citations": self.citations,
            "environment_type": self.environment_type.value
        }


@dataclass
class Paragraph:
    """Represents a paragraph with multiple sentences"""
    father: Optional[Section] = None
    sentences: List[Sentence] = field(default_factory=list)
    environment_type: EnvironmentType = EnvironmentType.TEXT
    
    def add_sentence(self, sentence: Sentence):
        sentence.father = self
        self.sentences.append(sentence)
    
    def get_skeleton(self) -> List[Dict[str, Union[str, List[str]]]]:
        """Return all sentences in this paragraph"""
        return [s.get_skeleton() for s in self.sentences]


@dataclass
class Section:
    """Represents a section, subsection, subsubsection or abstract"""
    name: str = ""
    father: Optional[Union[Section, Paper]] = None
    paragraphs: List[Paragraph] = field(default_factory=list)
    children: List[Section] = field(default_factory=list)
    level: int = 0  # 0=section, 1=subsection, 2=subsubsection
    
    def add_paragraph(self, child: Paragraph):
        child.father = self
        self.paragraphs.append(child)
    
    def add_child(self, child: Section):
        child.father = self
        self.children.append(child)
        
    def get_skeleton(self, i) -> Dict[str, any]:
        return {
            "title": self.name,
            "section_id": i,
            "paragraphs": [paragraph.get_skeleton() for paragraph in self.paragraphs],
            "sections": [section.get_skeleton(f"{i}.{j + 1}") for j, section in enumerate(self.children)],
        }


@dataclass
class Paper(Section):
    """Represents the entire academic paper"""
    title: str = ""
    author: Optional[str] = None
    abstract: Optional[Section] = None  # Abstract should not have children
    references: dict = field(default_factory=dict)  # Maps citation keys to bibliography entries
    has_section_index: bool = True
    
    def get_skeleton(self) -> dict:
        return {
            "title": self.title,
            "author": self.author,
            "abstract": self.abstract.get_skeleton("") if self.abstract else "",
            "paragraphs": [paragraph.get_skeleton() for paragraph in self.paragraphs],
            "sections": [section.get_skeleton(i + 1) for i, section in enumerate(self.children)],
            "citations": self.references
        }


class LaTeXParser:
    """Parse LaTeX documents into Paper structure"""
    
    # Environments that should be treated as content
    CONTENT_ENVIRONMENTS = {
        'abstract', 'equation', 'equation*', 'align', 'align*', 
        'displaymath', 'gather', 'gather*', 'multline', 'multline*',
        'figure', 'figure*', 'table', 'table*', 'tabular',
        'algorithm', 'algorithmic', 'lstlisting', 'verbatim',
        'theorem', 'lemma', 'corollary', 'proof', 'definition',
        'example', 'remark', 'note', 'tcolorbox', 'AIbox'
    }
    
    # Macros that represent citations
    CITATION_MACROS = {'cite', 'citep', 'citet', 'citeauthor', 'citeyear'}
    
    def __init__(self, filepath: Union[str, Path]):
        """Initialize parser with a LaTeX file path"""
        self.filepath = Path(filepath)
        self.base_dir = self.filepath.parent
        self.processed_files: Set[Path] = set()
        self.converter = LatexNodes2Text()
        
    def parse(self) -> Paper:
        """Parse the LaTeX file and return Paper object"""
        paper = Paper()
        
        # Read and process the main file
        content = self._read_file(self.filepath)
        
        # Extract document class and preamble info
        self._extract_metadata(content, paper)
        
        # Parse the document structure
        self._parse_document(content, paper)
        
        return paper
    
    def _read_file(self, filepath: Path) -> str:
        """Read a LaTeX file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _extract_metadata(self, content: str, paper: Paper):
        """Extract title, author, and other metadata from preamble"""
        # Extract title
        title_match = re.search(r'\\title\s*(?:\[.*?\])?\s*{(.*?)}(?:\s|\\|$)', content, re.DOTALL)
        if title_match:
            paper.title = self._clean_latex(title_match.group(1)).strip()
        
        # Extract author
        author_match = re.search(r'\\author\s*{(.*?)}(?:\s|\\|$)', content, re.DOTALL)
        if author_match:
            paper.author = self._clean_latex(author_match.group(1)).strip()
    
    def _parse_document(self, content: str, paper: Paper):
        """Parse the document body into sections and paragraphs"""
        # Find begin{document} and end{document}
        doc_start = content.find(r'\begin{document}')
        doc_end = content.find(r'\end{document}')
        
        if doc_start == -1:
            doc_content = content
        else:
            doc_content = content[doc_start + len(r'\begin{document}'):doc_end if doc_end != -1 else len(content)]
        
        # Process \input commands
        doc_content = self._process_input_commands(doc_content)
        
        # Extract abstract
        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', doc_content, re.DOTALL)
        if abstract_match:
            abstract_section = Section(name="Abstract")
            abstract_text = abstract_match.group(1).strip()
            self._parse_text_into_section(abstract_text, abstract_section)
            paper.abstract = abstract_section
            # Remove abstract from doc_content to avoid processing it again
            doc_content = doc_content[:abstract_match.start()] + doc_content[abstract_match.end():]
        
        # Parse sections
        self._parse_sections(doc_content, paper)
        
        # Extract references
        self._extract_references(content, paper)
    
    def _process_input_commands(self, content: str) -> str:
        """Process \\input{filename} commands"""
        def replace_input(match):
            input_file = match.group(1).strip()
            # Try different extensions
            for ext in ['', '.tex']:
                filepath = self.base_dir / f"{input_file}{ext}".replace('\\', '/').replace('/', os.sep)
                if filepath.exists() and filepath not in self.processed_files:
                    self.processed_files.add(filepath)
                    return self._read_file(filepath)
            return match.group(0)  # Return original if file not found
        
        # Match \input{...} and \input{...}.tex patterns
        content = re.sub(r'\\input\s*{([^}]+)}', replace_input, content)
        content = re.sub(r'\\include\s*{([^}]+)}', replace_input, content)
        return content
    
    def _parse_sections(self, content: str, parent: Union[Paper, Section]):
        """Recursively parse sections and subsections"""
        # Find all section headers
        section_pattern = r'\\(section|subsection|subsubsection)\s*(?:\[.*?\])?\s*{([^}]+)}'
        
        sections = []
        for match in re.finditer(section_pattern, content):
            level_map = {'section': 0, 'subsection': 1, 'subsubsection': 2}
            level = level_map[match.group(1)]
            title = self._clean_latex(match.group(2))
            sections.append((match.start(), match.end(), level, title))
        
        if not sections:
            # No sections found, treat all content as paragraphs
            self._parse_text_into_section(content, parent)
            return
        
        # Process content before first section
        if sections[0][0] > 0:
            preamble_content = content[:sections[0][0]]
            self._parse_text_into_section(preamble_content, parent)
        
        # Process each section
        for i, (start, end, level, title) in enumerate(sections):
            # Find the content for this section (until next section of same/higher level)
            section_end = len(content)
            for j in range(i + 1, len(sections)):
                next_start, _, next_level, _ = sections[j]
                if next_level <= level:
                    section_end = next_start
                    break
            
            section_content = content[end:section_end]
            
            # Create section object
            section = Section(name=title, level=level)
            
            # Handle nested sections recursively
            self._parse_sections(section_content, section)
            
            # Add to parent
            parent.add_child(section)
    
    def _parse_text_into_section(self, content: str, section: Union[Paper, Section]):
        """Parse text content into paragraphs and sentences"""
        # Remove comments
        lines = content.split('\n')
        lines = [line[:line.find('%')] if '%' in line else line for line in lines]
        content = '\n'.join(lines)
        
        # Split into paragraphs (separated by blank lines or explicit paragraph breaks)
        paragraphs = re.split(r'\n\s*\n+', content.strip())
        
        for para_text in paragraphs:
            if not para_text.strip():
                continue
            
            # Check if this paragraph contains special environment
            env_type = self._detect_environment_type(para_text)
            
            if env_type == EnvironmentType.EQUATION or env_type == EnvironmentType.FIGURE or env_type == EnvironmentType.TABLE:
                # Handle special environments
                para = Paragraph(environment_type=env_type)
                clean_text = self._extract_environment_content(para_text)
                para.add_sentence(Sentence(text=clean_text, environment_type=env_type))
                section.add_paragraph(para)
            else:
                # Regular text paragraph
                para = Paragraph(environment_type=EnvironmentType.TEXT)
                # Split into sentences
                sentences = self._split_into_sentences(para_text)
                for sentence_text in sentences:
                    if sentence_text.strip():
                        # Extract citations
                        citations = self._extract_citations(sentence_text)
                        clean_text = self._clean_latex(sentence_text)
                        para.add_sentence(Sentence(text=clean_text, citations=citations))
                
                if para.sentences:
                    section.add_paragraph(para)
    
    def _detect_environment_type(self, text: str) -> EnvironmentType:
        """Detect the type of environment in the text"""
        text_lower = text.lower()
        
        if re.search(r'\\begin\{(equation|align|displaymath|gather|multline)', text):
            return EnvironmentType.EQUATION
        elif re.search(r'\\begin\{(figure|figure\*)', text):
            return EnvironmentType.FIGURE
        elif re.search(r'\\begin\{(table|table\*|tabular)', text):
            return EnvironmentType.TABLE
        elif re.search(r'\\begin\{(algorithm|algorithmic)', text):
            return EnvironmentType.ALGORITHM
        elif re.search(r'\\begin\{(lstlisting|verbatim)', text):
            return EnvironmentType.LISTING
        elif re.search(r'\\begin\{(theorem|lemma|corollary|proof|definition)', text):
            return EnvironmentType.THEOREM
        else:
            return EnvironmentType.TEXT
    
    def _extract_environment_content(self, text: str) -> str:
        """Extract content from LaTeX environments"""
        # Remove environment markers
        text = re.sub(r'\\begin\{[^}]+\}', '', text)
        text = re.sub(r'\\end\{[^}]+\}', '', text)
        # Remove common LaTeX commands
        text = re.sub(r'\\(includegraphics|caption|label|centering|vspace|hspace)\s*(?:\[[^\]]*\])?\s*{[^}]*}', '', text)
        return self._clean_latex(text).strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple heuristic: split on periods, question marks, exclamation marks
        # but preserve references and citations
        text = text.replace('et al.', 'et_al')  # Temporary placeholder
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore placeholders
        sentences = [s.replace('et_al', 'et al.') for s in sentences]
        
        return sentences
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation keys from text"""
        citations = []
        
        # Match \cite, \citep, \citet, etc.
        cite_pattern = r'\\(?:cite|citep|citet|citeauthor|citeyear)\s*(?:\[[^\]]*\])?\s*{([^}]+)}'
        for match in re.finditer(cite_pattern, text):
            keys = match.group(1).split(',')
            citations.extend([k.strip() for k in keys])
        
        return citations
    
    def _extract_text_from_latex_nodes(self, latex_str: str) -> str:
        """
        Extract clean text from LaTeX string using pylatexenc.
        Handles special environments and node types.
        """
        try:
            walker = LatexWalker(latex_str)
            nodes, _, _ = walker.get_latex_nodes(pos=0, stop_on_closing_brace=False)
            return self._nodes_to_text(nodes)
        except Exception as e:
            # Fallback to regex-based cleaning
            return self._clean_latex(latex_str)
    
    def _nodes_to_text(self, nodes: List[LatexNode]) -> str:
        """Convert latex nodes to plain text"""
        if not nodes:
            return ""
        
        text_parts = []
        for node in nodes:
            if isinstance(node, str):
                text_parts.append(node)
            elif hasattr(node, 'nodelist'):
                # Recursively process node lists
                text_parts.append(self._nodes_to_text(node.nodelist))
            elif isinstance(node, LatexGroupNode):
                # Extract content from group nodes
                text_parts.append(self._nodes_to_text(node.nodelist))
            elif isinstance(node, LatexMacroNode):
                # Skip macros by default, but extract their arguments
                if node.nodeargs:
                    for arg in node.nodeargs:
                        if hasattr(arg, 'nodelist'):
                            text_parts.append(self._nodes_to_text(arg.nodelist))
            elif isinstance(node, LatexEnvironmentNode):
                # Extract environment content
                if hasattr(node, 'nodelist') and node.nodelist:
                    text_parts.append(self._nodes_to_text(node.nodelist))
            elif hasattr(node, '__str__'):
                text_parts.append(str(node))
        
        return ' '.join(filter(None, text_parts))
    
    def _clean_latex(self, text: str) -> str:
        """Remove LaTeX commands and formatting from text"""
        # Remove citations with their content
        text = re.sub(r'\\(?:cite|citep|citet|citeauthor|citeyear)\s*(?:\[[^\]]*\])?\s*{[^}]*}', '', text)
        
        # Remove common formatting commands but keep their content
        text = re.sub(r'\\(textbf|textit|texttt|emph|underline|sout|text)\s*{([^}]*)}', r'\2', text)
        
        # Remove commands that take arguments
        text = re.sub(r'\\(ref|label|footnote|href|url|command)\s*(?:\[[^\]]*\])?\s*{[^}]*}', '', text)
        
        # Remove generic LaTeX commands
        text = re.sub(r'\\([a-zA-Z@]+)\s*(?:\[[^\]]*\])?\s*{[^}]*}', '', text)
        text = re.sub(r'\\([a-zA-Z@]+)\s*(?:\[[^\]]*\])?', '', text)
        
        # Remove special characters
        text = re.sub(r'\\%', '%', text)
        text = re.sub(r'\\&', '&', text)
        text = re.sub(r'\\#', '#', text)
        text = re.sub(r'\\\$', '$', text)
        text = re.sub(r'\\textasciitilde', '~', text)
        text = re.sub(r'\\textasciicircum', '^', text)
        
        # Remove inline math ($ ... $) but keep display math for now
        text = re.sub(r'\$[^$]*\$', '', text)
        
        # Remove line breaks
        text = re.sub(r'\\\\', ' ', text)
        text = re.sub(r'\n\n+', '\n', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_references(self, content: str, paper: Paper):
        """Extract bibliography entries from .bib file or thebibliography environment"""
        # First try to find \bibliography command
        bib_match = re.search(r'\\bibliography\s*{([^}]+)}', content)
        if bib_match:
            bib_files = bib_match.group(1).split(',')
            for bib_file in bib_files:
                bib_path = self.base_dir / f"{bib_file.strip()}.bib"
                if bib_path.exists():
                    self._parse_bibtex_file(bib_path, paper)
        
        # Also try to parse .bbl file (compiled bibliography)
        bbl_path = self.base_dir / "main.bbl"
        if bbl_path.exists():
            self._parse_bbl_file(bbl_path, paper)
        
        # Finally, try thebibliography environment
        thebib_match = re.search(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', content, re.DOTALL)
        if thebib_match:
            self._parse_thebibliography(thebib_match.group(0), paper)
    
    def _parse_bibtex_file(self, filepath: Path, paper: Paper):
        """Parse a BibTeX file"""
        content = self._read_file(filepath)
        
        # Simple BibTeX parser: extract @entry{key,...} patterns
        entry_pattern = r'@\w+\s*{\s*([^,]+),'
        for match in re.finditer(entry_pattern, content):
            key = match.group(1).strip()
            # Extract entry text until next @ or end
            start = match.start()
            end = content.find('@', match.end())
            if end == -1:
                end = len(content)
            entry_text = content[start:end]
            paper.references[key] = entry_text.strip()
    
    def _parse_bbl_file(self, filepath: Path, paper: Paper):
        """Parse a compiled .bbl file"""
        content = self._read_file(filepath)
        
        # Extract bibliography items
        item_pattern = r'\\bibitem\s*(?:\[[^\]]*\])?\s*{([^}]+)}(.*?)(?=\\bibitem|\Z)'
        for match in re.finditer(item_pattern, content, re.DOTALL):
            key = match.group(1).strip()
            text = match.group(2).strip()
            paper.references[key] = text
    
    def _parse_thebibliography(self, content: str, paper: Paper):
        """Parse thebibliography environment"""
        item_pattern = r'\\bibitem\s*(?:\[[^\]]*\])?\s*{([^}]+)}(.*?)(?=\\bibitem|\Z)'
        for match in re.finditer(item_pattern, content, re.DOTALL):
            key = match.group(1).strip()
            text = match.group(2).strip()
            paper.references[key] = text


def parse_paper(filepath: Union[str, Path]) -> Paper:
    """
    Convenience function to parse a LaTeX paper file.
    
    Args:
        filepath: Path to the main.tex file
        
    Returns:
        Paper object containing the parsed structure
    """
    parser = LaTeXParser(filepath)
    return parser.parse()
