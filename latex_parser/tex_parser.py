"""
Latex Academic Paper Parser with Object-Oriented Structure
Implements Paper, Section, Paragraph, and Sentence classes
"""

import os
import re
import json
import glob
from typing import List
from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexMacroNode, LatexCharsNode
from paper_elements import *
from bib_parser import parse_bbl_file, parse_bib_file


class LatexPaperParser:
    """Parser to convert Latex documents into Paper objects"""
    
    def __init__(self, latex_content: str, base_path='.'):
        self.base_path = base_path
        self.latex_content = self._process_input_commands(latex_content)
        self.walker = LatexWalker(self.latex_content)
        # keep_inline_math=True, keep_display_math=True,
        self.converter = LatexNodes2Text(math_mode="verbatim")
        self.section_levels = {
            'section': 1,
            'subsection': 2,
            'subsubsection': 3,
            'paragraph': 4,
            'subparagraph': 5
        }
        # Environments that should be kept as-is (not split into sentences)
        self.preserved_environments = {
            'theorem', 'lemma', 'corollary', 'proposition', 'definition',
            'remark', 'note', 'example', 'proof',
            'tikzpicture', 'figure', 'table', 'algorithm',
            'equation', 'equation*', 'align', 'align*', 
            'gather', 'gather*', 'multline', 'multline*',
            'eqnarray', 'eqnarray*', 'displaymath',
            'tabular', 'verbatim', 'lstlisting',
        }        
        self.graph_environments = {'tikzpicture', 'figure', 'table', 'tabular'}
        self.spacing_environments = {'doublespace', 'singlespace'}        
        # Store bibliography entries
        self.bibliography_entries = {}

    def _process_input_commands(self, latex_content):
        """
        Process \input{filename} commands by replacing them with file contents
        
        Args:
            latex_content: Latex content string
            
        Returns:
            str: Latex content with all \input commands resolved
        """
        # Pattern to match \input{filename} or \input{filename.tex}
        # Handles optional spaces and both with/without .tex extension
        pattern = re.compile(r'\\input\s*\{([^}]+)\}')
        
        def replace_input(match):
            filename = match.group(1).strip()
            
            # Add .tex extension if not present
            if "." not in filename:
                filename += '.tex'

            if "/" in filename:
                filename = filename.split("/")
            elif "\\" in filename:
                filename = filename.split("\\")
            else:
                filename = [filename]
            
            # Construct full path
            filepath = os.path.join(self.base_path, *filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Recursively process \input commands in the included file
                return self._process_input_commands(file_content)
            
            except FileNotFoundError:
                print(f"Warning: Could not find file '{filepath}' for \\input command")
                return f"% File not found: {filename}"
            
            except Exception as e:
                print(f"Warning: Error reading file '{filepath}': {e}")
                return f"% Error reading file: {filename}"
        
        # Replace all \input commands
        processed_content = re.sub(pattern, replace_input, latex_content)
        
        return processed_content
   
    def parse(self) -> Optional[LatexPaper]:
        """
        Single pass parse Latex content into a Paper object
        
        Returns:
            Paper: Structured paper object
        """
        paper = LatexPaper()
        
        has_document = False
        nodelist, _, _ = self.walker.get_latex_nodes()
        for node in nodelist:
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
                    if title is not None: paper.title = title
                    if author is not None: paper.author = author
                    if abstract is not None: paper.abstract = abstract

        if not has_document: return
        
        # parse citations
        self.bibliography_entries = parse_bbl_file(self.latex_content)
        for f in glob.glob(os.path.join(self.base_path, "*.bbl")):
            bib = parse_bbl_file(f)
            self.bibliography_entries.update(bib)
        for f in glob.glob(os.path.join(self.base_path, "*.bib")):
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
        i = 0
        
        while i < len(nodes):
            node = nodes[i]
            if isinstance(node, LatexMacroNode):            
                if node.macroname == 'section':
                    section_name = self._extract_title(node)
                    section = LatexSection(name=section_name)
                    
                    # Collect content until next section
                    j = i + 1
                    section_content = []
                    while j < len(nodes):
                        next_node = nodes[j]
                        if isinstance(next_node, LatexMacroNode) and next_node.macroname == 'section':
                            break
                        section_content.append(next_node)
                        j += 1
                    
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
                    elif node.environmentname in self.spacing_environments:
                        output = self._parse_sections(node.nodelist)
                        sections_in_environment, title_back, author_back, abstract_back = output
                        if title is not None: title = title_back
                        if author is not None: author = author_back
                        if abstract is not None: abstract = abstract_back
                        sections.extend(sections_in_environment)
        
        return sections, title, author, abstract
    
    def _parse_section_content(self, nodes, parent_section: LatexSection):
        """Parse content of a section (subsections and paragraphs)"""
        i = 0
        current_text_nodes = []
        
        while i < len(nodes):
            node = nodes[i]
            
            if isinstance(node, LatexMacroNode) and node.macroname == 'subsection':
                # Save accumulated text as paragraphs
                if current_text_nodes:
                    self._create_paragraphs_from_nodes(current_text_nodes, parent_section)
                    current_text_nodes = []
                
                # Parse subsection
                subsection_name = self._extract_title(node)
                subsection = LatexSubSection(name=subsection_name)
                
                # Collect subsection content
                j = i + 1
                subsection_content = []
                while j < len(nodes):
                    next_node = nodes[j]
                    if isinstance(next_node, LatexMacroNode) and next_node.macroname in ['subsection', 'section']:
                        break
                    subsection_content.append(next_node)
                    j += 1
                
                # Parse subsection content
                self._parse_subsection_content(subsection_content, subsection)
                parent_section.add_child(subsection)
                i = j
            else:
                current_text_nodes.append(node)
                i += 1
        
        # Add remaining text
        if current_text_nodes:
            self._create_paragraphs_from_nodes(current_text_nodes, parent_section)
    
    def _parse_subsection_content(self, nodes, parent_subsection: LatexSubSection):
        """Parse content of a subsection (subsubsections and paragraphs)"""
        i = 0
        current_text_nodes = []
        
        while i < len(nodes):
            node = nodes[i]
            
            if isinstance(node, LatexMacroNode) and node.macroname == 'subsubsection':
                # Save accumulated text as paragraphs
                if current_text_nodes:
                    self._create_paragraphs_from_nodes(current_text_nodes, parent_subsection)
                    current_text_nodes = []
                
                # Parse subsubsection
                subsubsection_name = self._extract_title(node)
                
                # Collect subsubsection content
                j = i + 1
                subsubsection_content = []
                while j < len(nodes):
                    next_node = nodes[j]
                    if isinstance(next_node, LatexMacroNode) and next_node.macroname in ['subsubsection', 'subsection']:
                        break
                    subsubsection_content.append(next_node)
                    j += 1
                
                # Parse subsubsection content (paragraphs)
                # self._parse_subsubsection_content(subsubsection_content, subsubsection)
                subsubsection = LatexSubSubSection(name=subsubsection_name)
                self._create_paragraphs_from_nodes(subsubsection_content, subsubsection)
                parent_subsection.add_child(subsubsection)
                i = j
            else:
                current_text_nodes.append(node)
                i += 1
    
    def _create_paragraphs_from_nodes(self, nodes, parent) -> List[LatexParagraph]:
        """Create paragraph objects from text nodes, splitting by \n\n"""
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
        
        # First pass: identify and extract preserved environments
        for node in nodes:            
            if isinstance(node, LatexEnvironmentNode):
                # Check if this environment should be preserved
                # if node.environmentname in self.preserved_environments:
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
        """
        Extract the raw LaTeX content of an environment
        
        Args:
            env_node: LatexEnvironmentNode to extract
            
        Returns:
            str: Raw LaTeX string including \begin and \end
        """
        env_name = env_node.environmentname
        
        # Get the content
        content = self.converter.nodelist_to_text(env_node.nodelist)
        
        # Reconstruct the environment
        result = f"\\begin{{{env_name}}}\n{content}\n\\end{{{env_name}}}"
        
        return result

    def _parse_text_with_citations_and_breaks(self, nodes) -> List[Union[LatexSentence, dict]]:
        """
        Parse text nodes into sentences with paragraph break detection
        Returns list of Sentence objects and paragraph break markers
        
        Args:
            nodes: List of LaTeX nodes
            
        Returns:
            list: Sentence objects with special 'paragraph_break' markers
        """
        # Extract text segments with citation markers
        segments = self._extract_text_segments_with_breaks(nodes)
        
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
                text = node.chars.strip()
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
                    segments.append({'text': '', 'citations': citations, 'paragraph_break': False})
                elif node.macroname == 'par':
                    # Explicit paragraph break command
                    segments.append({'text': '\n', 'citations': [], 'paragraph_break': True})
                else:
                    try:
                        text = self.converter.nodelist_to_text([node])
                        segments.append({'text': text, 'citations': [], 'paragraph_break': False})
                    except:
                        if node.nodeargd and node.nodeargd.argnlist:
                            for arg in node.nodeargd.argnlist:
                                if hasattr(arg, 'nodelist'):
                                    segments.extend(self._extract_text_segments_with_breaks(arg.nodelist))
            
            elif isinstance(node, LatexEnvironmentNode) and node.environmentname not in self.preserved_environments:
                # Skip preserved environments - they should not be split
                segments.extend(self._extract_text_segments_with_breaks(arg.nodelist))
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
                
            elif isinstance(item, LatexEnvironment) and item.environment_name in self.graph_environments:
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
        """
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
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.\s', r'\1<PERIOD>', text)
        
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
            
            sentence = sentence.replace('<PERIOD>', '. ')
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
        nodelist, _, _ = self.walker.get_latex_nodes()
        
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
