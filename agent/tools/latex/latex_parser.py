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
    CITATION_MACROS = {'cite', 'citep', 'citet', 'citeauthor', 'citeyear', 'citeyearpar'}
    
    def __init__(self, filepath: Union[str, Path]):
        """Initialize parser with a LaTeX file path"""
        self.filepath = Path(filepath)
        self.base_dir = self.filepath.parent
        self.processed_files: Set[Path] = set()
        self.converter = LatexNodes2Text()
        self.command_definitions: Dict[str, str] = {}  # Store \newcommand definitions
        
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
        # Extract command definitions first
        self._extract_command_definitions(content)
        
        # Extract title
        title_match = re.search(r'\\title\s*(?:\[.*?\])?\s*{(.*?)}(?:\s|\\|$)', content, re.DOTALL)
        if title_match:
            paper.title = self._clean_latex(title_match.group(1), None).strip()
        
        # Extract author
        author_match = re.search(r'\\author\s*{(.*?)}(?:\s|\\|$)', content, re.DOTALL)
        if author_match:
            paper.author = self._clean_latex(author_match.group(1), None).strip()
    
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
        """Parse sections and subsections into a proper hierarchical tree"""
        section_pattern = r'\\(section|subsection|subsubsection)\s*(?:\[.*?\])?\s*{([^}]+)}'
        level_map = {'section': 0, 'subsection': 1, 'subsubsection': 2}

        raw_sections = []
        for match in re.finditer(section_pattern, content):
            raw_sections.append({
                'start': match.start(),
                'end': match.end(),
                'level': level_map[match.group(1)],
                'title': self._clean_latex(match.group(2), None),
                'node': Section(name=self._clean_latex(match.group(2), None), level=level_map[match.group(1)]),
                'parent': None,
                'section_end': len(content),
            })

        if not raw_sections:
            self._parse_text_into_section(content, parent)
            return

        # Compute section end boundaries
        for i, item in enumerate(raw_sections):
            for j in range(i + 1, len(raw_sections)):
                if raw_sections[j]['level'] <= item['level']:
                    item['section_end'] = raw_sections[j]['start']
                    break

        # Process content before the first top-level section
        if raw_sections[0]['start'] > 0:
            self._parse_text_into_section(content[:raw_sections[0]['start']], parent)

        # Build parent-child relationships using a level stack
        stack: List[Dict[str, Union[Paper, Section, int]]] = [{'node': parent, 'level': -1}]
        for item in raw_sections:
            while stack and item['level'] <= stack[-1]['level']:
                stack.pop()
            parent_node = stack[-1]['node']
            parent_node.add_child(item['node'])
            item['parent'] = stack[-1]
            stack.append({'node': item['node'], 'level': item['level'], 'item': item})

        # Parse direct text segments for each section
        for item in raw_sections:
            # Collect only immediate children of this section
            child_sections = [child for child in raw_sections if child.get('parent') and child['parent'].get('item') is item]
            child_sections.sort(key=lambda c: c['start'])

            segment_start = item['end']
            for child in child_sections:
                if child['start'] > segment_start:
                    self._parse_text_into_section(content[segment_start:child['start']], item['node'])
                segment_start = child['section_end']

            if segment_start < item['section_end']:
                self._parse_text_into_section(content[segment_start:item['section_end']], item['node'])
    
    def _parse_text_into_section(self, content: str, section: Union[Paper, Section]):
        """Parse text content into paragraphs and sentences"""
        # Remove comments
        lines = content.split('\n')
        lines = [line[:line.find('%')] if '%' in line else line for line in lines]
        content = '\n'.join(lines)
        
        # First, extract and separate out complex environments
        content, environments = self._extract_environments(content)
        
        # Split into paragraphs (separated by blank lines or explicit paragraph breaks)
        paragraphs = re.split(r'\n\s*\n+', content.strip())
        
        for para_text in paragraphs:
            if not para_text.strip():
                continue
            
            # Check if this paragraph contains special environment references
            env_type = self._detect_environment_type(para_text)
            
            if env_type in (EnvironmentType.EQUATION, EnvironmentType.FIGURE, EnvironmentType.TABLE):
                # Handle special environments
                para = Paragraph(environment_type=env_type)
                clean_text = self._extract_environment_content(para_text)
                if clean_text.strip():  # Only add if text is not empty
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
                        clean_text = self._clean_latex(sentence_text, citations)
                        
                        # Filter out empty sentences and sentences with only braces/punctuation
                        if self._is_valid_sentence(clean_text):
                            para.add_sentence(Sentence(text=clean_text, citations=citations))
                
                if para.sentences:
                    section.add_paragraph(para)
    
    def _extract_environments(self, content: str) -> tuple:
        """Extract complex environments from text, returning cleaned content and extracted environments"""
        # Extract figure, table, equation environments
        env_patterns = [
            (r'\\begin\{(?:figure|figure\*)\}.*?\\end\{(?:figure|figure\*)\}', 'figure'),
            (r'\\begin\{(?:table|table\*)\}.*?\\end\{(?:table|table\*)\}', 'table'),
            (r'\\begin\{(?:align|align\*|gather|gather\*|multline|multline\*|equation|equation\*)\}.*?\\end\{(?:align|align\*|gather|gather\*|multline|multline\*|equation|equation\*)\}', 'equation'),
        ]
        
        environments = []
        cleaned_content = content
        
        for pattern, env_type in env_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                # Extract caption if it's a figure/table
                env_text = match.group(0)
                caption_match = re.search(r'\\caption\s*\{([^}]*)\}', env_text)
                if caption_match:
                    environments.append({
                        'type': env_type,
                        'content': caption_match.group(1),
                        'full': env_text
                    })
                # Replace the environment with a placeholder
                cleaned_content = cleaned_content.replace(env_text, '')
        
        return cleaned_content, environments
    
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
        return self._clean_latex(text, None).strip()
    

    def _is_valid_sentence(self, text: str) -> bool:
        """Check if a sentence is valid (not empty, not just punctuation/braces)"""
        # Remove all whitespace and punctuation
        clean = re.sub(r'[\s\{\}\[\]\(\)\.,;:!?\-—‐–~^]', '', text)
        
        # Must have at least 2 characters of actual content (words)
        return len(clean) >= 2
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better handling of abbreviations"""
        # Protect common abbreviations with placeholders
        # Use word boundaries to avoid replacing parts of words
        replacements = [
            (r'\bet\s+al\.', 'et_al'),  # et al.
            (r'\be\.g\.', 'e_g'),  # e.g.
            (r'\bi\.e\.', 'i_e'),  # i.e.
            (r'\betc\.', 'etc_period'),  # etc.
            (r'\bvs\.', 'vs_period'),  # vs.
            (r'\bno\.', 'no_period'),  # no.
            (r'\bfig\.', 'fig_period'),  # fig.
            (r'\beq\.', 'eq_period'),  # eq.
            (r'\bDr\.', 'Dr_period'),  # Dr.
            (r'\bMr\.', 'Mr_period'),  # Mr.
            (r'\bMrs\.', 'Mrs_period'),  # Mrs.
            (r'\bProf\.', 'Prof_period'),  # Prof.
            (r'\bU\.S\.', 'US_period'),  # U.S.
        ]
        
        for pattern, placeholder in replacements:
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
        
        # Split on sentence boundaries (period followed by space and uppercase, or ? !)
        # More sophisticated: split on ./?/! followed by space and uppercase letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore abbreviations
        for pattern, placeholder in replacements:
            for i in range(len(sentences)):
                _, original = pattern, placeholder
                # Restore from placeholder
                if placeholder == 'et_al':
                    sentences[i] = sentences[i].replace('et_al', 'et al.')
                elif placeholder == 'e_g':
                    sentences[i] = sentences[i].replace('e_g', 'e.g.')
                elif placeholder == 'i_e':
                    sentences[i] = sentences[i].replace('i_e', 'i.e.')
                elif placeholder == 'etc_period':
                    sentences[i] = sentences[i].replace('etc_period', 'etc.')
                elif placeholder == 'vs_period':
                    sentences[i] = sentences[i].replace('vs_period', 'vs.')
                elif placeholder == 'no_period':
                    sentences[i] = sentences[i].replace('no_period', 'no.')
                elif placeholder == 'fig_period':
                    sentences[i] = sentences[i].replace('fig_period', 'fig.')
                elif placeholder == 'eq_period':
                    sentences[i] = sentences[i].replace('eq_period', 'eq.')
                elif placeholder == 'Dr_period':
                    sentences[i] = sentences[i].replace('Dr_period', 'Dr.')
                elif placeholder == 'Mr_period':
                    sentences[i] = sentences[i].replace('Mr_period', 'Mr.')
                elif placeholder == 'Mrs_period':
                    sentences[i] = sentences[i].replace('Mrs_period', 'Mrs.')
                elif placeholder == 'Prof_period':
                    sentences[i] = sentences[i].replace('Prof_period', 'Prof.')
                elif placeholder == 'US_period':
                    sentences[i] = sentences[i].replace('US_period', 'U.S.')
        
        return sentences
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation keys from text"""
        citations = []
        
        # Match \cite, \citep, \citet, etc.
        cite_pattern = r'\\(?:cite|citep|citet|citeauthor|citeyear)\s*(?:\[[^\]]*\])?\s*{([^}]+)}'
        for match in re.finditer(cite_pattern, text):
            keys = match.group(1).split(',')
            citations.extend([k.strip() for k in keys])
        
        # Also match standalone {key} that contain 4 digits (likely citation keys)
        brace_pattern = r'\{([^}]*\d{4}[^}]*)\}'
        for match in re.finditer(brace_pattern, text):
            key = match.group(1).strip()
            if key and key not in citations:  # Avoid duplicates
                citations.append(key)
        
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
            return self._clean_latex(latex_str, None)
    
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
    
    def _clean_latex(self, text: str, citations: List[str] = None) -> str:
        """Remove LaTeX commands and formatting from text using proper parsing"""
        try:
            # Use pylatexenc for proper LaTeX parsing
            walker = LatexWalker(text)
            nodes, _, _ = walker.get_latex_nodes(pos=0, stop_on_closing_brace=False)
            clean_text = self._nodes_to_clean_text(nodes)
        except Exception:
            # Fallback to a more careful regex-based approach
            clean_text = self._clean_latex_fallback(text, citations)
        
        # Remove any remaining citation braces based on extracted citations
        if citations:
            # Sort by length descending to remove nested ones first
            citations_sorted = sorted(set(citations), key=len, reverse=True)
            for cite in citations_sorted:
                clean_text = clean_text.replace(f'{{{cite}}}', '')
        
        # Remove any remaining citation-like braces
        clean_text = re.sub(r'\{[^}]*\d{4}[^}]*', '', clean_text)
        
        return clean_text
    
    def _clean_latex_fallback(self, text: str, citations: List[str] = None) -> str:
        """Fallback regex-based LaTeX cleaning with better handling of nested braces"""
        # First, substitute custom command definitions
        text = self._substitute_command_definitions(text)
        
        # Remove display math environments  
        text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
        
        # Remove inline math
        text = re.sub(r'\$[^$]*\$', '', text)
        
        # First pass: remove all citation commands with their complex arguments
        # This needs to handle nested braces properly
        text = self._remove_citation_commands(text)
        
        # Second pass: remove cross-reference commands
        text = self._remove_xref_commands(text)
        
        # Third pass: remove formatting commands, keeping content
        text = self._remove_formatting_commands(text)
        
        # Fourth pass: remove remaining LaTeX commands
        text = re.sub(r'\\(?:[a-zA-Z@]+)\s*(?:\[[^\]]*\])?', '', text)
        
        # Fifth pass: clean up special LaTeX characters and braces
        text = text.replace(r'\%', '%')
        text = text.replace(r'\&', '&')
        text = text.replace(r'\#', '#')
        text = text.replace(r'\$', '$')
        text = text.replace(r'\{', '{')
        text = text.replace(r'\}', '}')
        text = text.replace(r'\textasciitilde', '~')
        text = text.replace(r'\textasciicircum', '^')
        
        # Remove line breaks
        text = re.sub(r'\\\\', ' ', text)
        
        # Clean up remaining braces - remove empty braces and orphaned closing braces
        text = re.sub(r'\{\s*\}', '', text)  # Remove empty braces
        text = re.sub(r'(?<!\{)\s*\}\s*', '', text)  # Remove orphaned closing braces
        text = re.sub(r'\s*\{\s*', '{', text)  # Clean up braces with spaces
        text = re.sub(r'\s*\}\s*', '}', text)
        
        # Remove empty parentheses (with optional whitespace) - multiple passes for nested cases
        for _ in range(3):
            text = re.sub(r'\(\s*\)', '', text)
        
        # Clean up tildes not followed by content
        text = re.sub(r'~\s+', ' ', text)  # Tilde followed by whitespace
        text = re.sub(r'~', '', text)  # Remaining tildes
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\s\{\}]+$', '', text)  # Remove trailing whitespace/braces
        text = re.sub(r'^[\s\{\}]+', '', text)  # Remove leading whitespace/braces
        
        # Remove any remaining citation braces ( { ... } containing 4 digits )
        text = re.sub(r'\{[^}]*\d{4}[^}]*', '', text)
        
        return text.strip()
    
    def _substitute_command_definitions(self, text: str) -> str:
        """Substitute custom command definitions in the text"""
        for command, definition in self.command_definitions.items():
            # Clean the definition first (remove LaTeX commands from it)
            clean_definition = self._clean_latex_simple(definition)
            # Replace the command with its cleaned definition
            # Use word boundaries to avoid partial matches
            pattern = re.escape(command) + r'(?=\s|[^a-zA-Z@]|$)'
            text = re.sub(pattern, clean_definition, text)
        
        return text
    
    def _clean_latex_simple(self, text: str) -> str:
        """Simple LaTeX cleaning for command definitions"""
        # Remove \xspace and other simple commands
        text = re.sub(r'\\xspace', '', text)
        text = re.sub(r'\\(?:[a-zA-Z@]+)', '', text)
        return text.strip()
    
    def _remove_citation_commands(self, text: str) -> str:
        """Remove citation commands with proper brace handling"""
        # First, clean up leading tildes before citations (LaTeX non-breaking space)
        # Note: Order matters - try longer patterns first!
        text = re.sub(r'~\s*\\(?:citeyear|citeauthor|citep|citet|cite)(?=\s|\[|\{)', '', text)
        
        pattern = r'\\(?:citeyear|citeauthor|citep|citet|cite)(?=\s|\[|\{)\s*(?:\[[^\]]*\])?\s*'
        pos = 0
        result = []
        
        for match in re.finditer(pattern, text):
            result.append(text[pos:match.start()])
            pos = match.end()
            
            # Skip the argument
            if pos < len(text) and text[pos] == '{':
                brace_count = 0
                for i in range(pos, len(text)):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            pos = i + 1
                            break
        
        result.append(text[pos:])
        return ''.join(result)
    
    def _remove_xref_commands(self, text: str) -> str:
        """Remove cross-reference commands like \\cref, \\ref, etc."""
        # Order matters - try longer patterns first!
        pattern = r'\\(?:crefrange|crefs|cref|eqref|label|ref)(?=\s|\[|\{)\s*(?:\[[^\]]*\])?\s*'
        pos = 0
        result = []
        
        for match in re.finditer(pattern, text):
            result.append(text[pos:match.start()])
            pos = match.end()
            
            # Skip the argument
            if pos < len(text) and text[pos] == '{':
                brace_count = 0
                for i in range(pos, len(text)):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            pos = i + 1
                            break
        
        result.append(text[pos:])
        return ''.join(result)
    
    def _remove_formatting_commands(self, text: str) -> str:
        """Remove formatting commands but keep their content"""
        # Pattern to match formatting commands (longer patterns first)
        pattern = r'\\(?:textasciitilde|textasciicircum|textbf|textit|texttt|emph|underline|textmd|textrm|textsf|textup|textsl|text|small|tiny|large|Large|LARGE|huge|Huge)(?=\s|\{)\s*'
        pos = 0
        result = []
        
        for match in re.finditer(pattern, text):
            result.append(text[pos:match.start()])
            pos = match.end()
            
            # Extract the content inside braces
            if pos < len(text) and text[pos] == '{':
                brace_count = 0
                content_start = pos + 1
                for i in range(pos, len(text)):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            result.append(text[content_start:i])
                            pos = i + 1
                            break
        
        result.append(text[pos:])
        return ''.join(result)
    
    def _nodes_to_clean_text(self, nodes: List[LatexNode]) -> str:
        """Convert LaTeX nodes to clean text, properly handling structure"""
        if not nodes:
            return ""
        
        text_parts = []
        
        for node in nodes:
            if isinstance(node, str):
                text_parts.append(node)
            elif isinstance(node, LatexMacroNode):
                # For formatting macros, extract the argument
                if node.macro_name in ('textbf', 'textit', 'texttt', 'emph', 'underline', 
                                       'textmd', 'textrm', 'textsf', 'textup', 'textsl', 'text'):
                    if node.nodeargs:
                        for arg in node.nodeargs:
                            if hasattr(arg, 'nodelist'):
                                text_parts.append(self._nodes_to_clean_text(arg.nodelist))
                # Skip citation and reference macros
                elif node.macro_name in ('cite', 'citep', 'citet', 'citeauthor', 'citeyear', 
                                        'ref', 'label', 'footnote', 'href', 'url', 'link'):
                    pass
                # Skip other commands
                elif node.macro_name.startswith('includegraphics') or node.macro_name.startswith('vspace'):
                    pass
                # For other macros, try to extract content if available
                else:
                    if hasattr(node, 'nodeargs') and node.nodeargs:
                        for arg in node.nodeargs:
                            if hasattr(arg, 'nodelist'):
                                text_parts.append(self._nodes_to_clean_text(arg.nodelist))
            elif isinstance(node, LatexGroupNode):
                # Extract content from group nodes
                if hasattr(node, 'nodelist'):
                    text_parts.append(self._nodes_to_clean_text(node.nodelist))
            elif isinstance(node, LatexEnvironmentNode):
                # Skip certain environments
                if node.environmentname not in ('equation', 'equation*', 'align', 'align*', 
                                               'displaymath', 'gather', 'multline'):
                    if hasattr(node, 'nodelist') and node.nodelist:
                        text_parts.append(self._nodes_to_clean_text(node.nodelist))
            elif hasattr(node, 'nodelist'):
                # Recursively process nodes with nodelists
                text_parts.append(self._nodes_to_clean_text(node.nodelist))
        
        # Join and clean up
        result = ' '.join(filter(None, text_parts))
        # Clean up double spaces
        result = re.sub(r'\s+', ' ', result)
        return result.strip()
    
    def _extract_command_definitions(self, content: str):
        """Extract \newcommand definitions from the LaTeX content"""
        # Match \newcommand{\command}{definition} or \newcommand{\command}[args]{definition}
        newcommand_pattern = r'\\newcommand\s*\{(\\[a-zA-Z@]+)\}(?:\[[^\]]*\])?\s*\{([^}]*)\}'
        
        for match in re.finditer(newcommand_pattern, content):
            command = match.group(1)
            definition = match.group(2)
            # Store the command without the backslash as key
            self.command_definitions[command] = definition
        
        # Also handle \def commands
        def_pattern = r'\\def\s*(\\[a-zA-Z@]+)\s*\{([^}]*)\}'
        for match in re.finditer(def_pattern, content):
            command = match.group(1)
            definition = match.group(2)
            self.command_definitions[command] = definition
    
    def _extract_references(self, content: str, paper: Paper):
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
