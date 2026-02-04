import re
import xml.etree.ElementTree as ET
from typing import List, Optional

from ..paper_elements import Paper, Section, Paragraph, Sentence


class PaperParser:
    NS = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    # Keywords that often indicate non-section headings
    NON_SECTION_KEYWORDS = [
        'figure', 'fig', 'table', 'theorem', 'lemma', 'proposition', 
        'corollary', 'definition', 'remark', 'example', 'proof',
        'algorithm', 'equation', 'appendix'
    ]
            
    def parse(self, xml_content: str, mode: str = "casual") -> Paper:
        root = ET.fromstring(xml_content)        
        # Create the root Paper object
        paper = Paper(name="root", father=None)        
        # Extract metadata
        paper.title = self._extract_title(root)
        paper.author = self._extract_authors_string(root)        
        # Extract references/bibliography
        self._extract_references(root)       
        # Extract abstract
        paper.abstract = self._extract_abstract(root, paper)
        # Extract body sections
        self._extract_body_sections(root, paper, mode)
        paper.references = {x['key']: x['title'] for x in self._citation_map.values() if 'key' in x}  
        return paper
    
    def _extract_title(self, root: ET.Element) -> str:
        """Extract paper title."""
        title_elem = root.find('.//tei:titleStmt/tei:title', self.NS)
        return title_elem.text if title_elem is not None else ""
    
    def _extract_authors_string(self, root: ET.Element) -> str:
        """Extract authors as a formatted string."""
        authors = []
        
        for author in root.findall('.//tei:sourceDesc//tei:author', self.NS):
            persname = author.find('.//tei:persName', self.NS)
            if persname is not None:
                forename = persname.find('.//tei:forename[@type="first"]', self.NS)
                surname = persname.find('.//tei:surname', self.NS)
                
                first = forename.text if forename is not None else ""
                last = surname.text if surname is not None else ""
                full_name = f"{first} {last}".strip()
                if full_name:
                    authors.append(full_name)
        
        return ", ".join(authors)
    
    def _extract_abstract(self, root: ET.Element, paper: Paper) -> Optional[Section]:
        """Extract abstract as a Section."""
        abstract_elem = root.find('.//tei:profileDesc/tei:abstract', self.NS)
        if abstract_elem is None:
            return None
        
        abstract_section = Section(name="Abstract", father=paper)
        
        # Extract paragraphs from abstract
        for div in abstract_elem.findall('.//tei:div', self.NS):
            for p_elem in div.findall('.//tei:p', self.NS):
                paragraph = Paragraph(father=abstract_section)
                self._parse_paragraph_element(p_elem, paragraph)
                if paragraph.sentences:
                    abstract_section.add_paragraph(paragraph)
        
        # If no divs, try direct paragraphs
        if not abstract_section.paragraphs:
            for p_elem in abstract_elem.findall('.//tei:p', self.NS):
                paragraph = Paragraph(father=abstract_section)
                self._parse_paragraph_element(p_elem, paragraph)
                if paragraph.sentences:
                    abstract_section.add_paragraph(paragraph)
        
        return abstract_section if abstract_section.paragraphs else None
   
    def _extract_body_sections(self, root: ET.Element, paper: Paper, mode: str):
        """Extract body sections with hierarchical structure."""
        body = root.find('.//tei:text/tei:body', self.NS)
        if body is None: return
        self._current_section_hierarchy = [paper]
        for div in body.findall('./tei:div', self.NS):
            self._parse_div_element(div, mode)

    def _parse_div_element(self, div_element: ET.Element, mode: str):
        """
        mode = "strict": 严格鉴别标题从属关系，用于解析待测综述和anchor survey
        mode = "casual": 仅解析出标题，不分级，用于解析引文
        """
        for child in div_element:
            if child.tag == f"{{{self.NS['tei']}}}head":
                n_attr = child.get('n', "")
                text = ' '.join(child.itertext()).strip()
                if mode == "strict":
                    while n_attr.count(".") + 1 < len(self._current_section_hierarchy):
                        # 回退section层级以找到
                        self._current_section_hierarchy.pop()
                section = Section(name=text, father=self._current_section_hierarchy[-1])
                self._current_section_hierarchy[-1].add_child(section)
                if mode == "strict": self._current_section_hierarchy.append(section)

            elif child.tag == f"{{{self.NS['tei']}}}p":
                paragraph = Paragraph(father=self._current_section_hierarchy[-1])
                self._parse_paragraph_element(child, paragraph)
                self._current_section_hierarchy[-1].add_paragraph(paragraph)
    
    def _parse_paragraph_element(self, p_element: ET.Element, paragraph: Paragraph):
        """
        Parse a paragraph element into Sentence objects.
        
        Args:
            p_elem: The paragraph XML element
            paragraph: Paragraph object to populate
            references: Dictionary of references for citation mapping
        """
        # Get all text and references in order
        text_parts = []
        current_text = []
        
        def process_elem(element: ET.Element, depth=0):
            # Add text before element
            if element.text:
                current_text.append(element.text)
            
            # Handle citation references
            if element.tag == f"{{{self.NS['tei']}}}ref" and element.get('type') == 'bibr':
                citation_id = element.get('target', '').replace('#', '')  # b1, b2, ..., b500
                key = re.sub(r'[^0-9]', '', ''.join(element.itertext()).strip())  # "Yu et al." or "[1]"
                
                # Add citation marker
                if current_text:
                    text_parts.append(('text', ''.join(current_text)))
                    current_text.clear()

                if citation_id in self._citation_map:
                    self._citation_map[citation_id]['key'] = key
                    text_parts.append(('citation', self._citation_map[citation_id]))
                
            else:
                # Recursively process child elements
                for child in element:
                    process_elem(child, depth + 1)
            
            # Add text after element
            if element.tail and depth > 0:
                current_text.append(element.tail)
        
        # Process the paragraph element
        process_elem(p_element)
        
        # Add any remaining text
        if current_text:
            text_parts.append(('text', ''.join(current_text)))
        
        # Split into sentences and assign citations
        self._create_sentences(text_parts, paragraph)
    
    def _create_sentences(self, text_parts: List[tuple[str, str]], paragraph: Paragraph):
        """
        Create Sentence objects from text parts and citations.
        
        Args:
            text_parts: List of tuples (type, content) where type is 'text' or 'citation'
            paragraph: Paragraph object to add sentences to
        """
        current_sentence = ""
        current_citations = []
        
        for part_type, content in text_parts:
            if part_type == 'text':
                # Split text into sentences (simple approach)
                sentences = re.split(r'(?<=[.!?])\s+', content)
                
                for i, sentence in enumerate(sentences):                    
                    if not (sentence := sentence.strip()): continue
                    
                    if i == 0:
                        # Continuation of current sentence
                        current_sentence += sentence
                    else:
                        # Create sentence for the previous one
                        if current_sentence:
                            sentence_obj = Sentence(current_sentence.strip(), paragraph, current_citations.copy())
                            paragraph.add_sentence(sentence_obj)
                        
                        # Start new sentence
                        current_sentence = sentence
                        current_citations = []
                
            elif part_type == 'citation':
                current_citations.append(content)
        
        # Add the last sentence
        if current_sentence := current_sentence.strip():
            sentence_obj = Sentence(current_sentence, paragraph, current_citations)
            paragraph.add_sentence(sentence_obj)
    
    def _extract_references(self, root: ET.Element) -> dict:
        """Extract bibliography/reference list as a dictionary."""        
        self._citation_map = {}
        back = root.find('.//tei:text/tei:back', self.NS)
        if back is None: return
        
        for biblstruct in back.findall('.//tei:listBibl/tei:biblStruct', self.NS):   
            if ref_id := biblstruct.get('{http://www.w3.org/XML/1998/namespace}id', ''):
                # Extract title
                title_element = biblstruct.find('.//tei:analytic/tei:title', self.NS)
                if title_element is None:
                    title_element = biblstruct.find('.//tei:monogr/tei:title', self.NS)
                title = ''.join(title_element.itertext()).strip() if title_element is not None else ""
                self._citation_map[ref_id] = {"title": title}

    def get_titles(self, xml_content: str) -> List[str]:
        root = ET.fromstring(xml_content)
        body = root.find('.//tei:text/tei:body', self.NS)
        if body is None:
            return
        
        heads = []
        for div in body.findall('./tei:div', self.NS):
            for child in div:
                if child.tag == f"{{{self.NS['tei']}}}head":
                    heads.append(' '.join(child.itertext()).lower())
        
        return heads
