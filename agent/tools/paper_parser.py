import re
import time
import random
import logging
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from paper_elements import Paper, Section, Paragraph, Sentence


@dataclass
class HTMLSection:
    numbers: List[int]
    level: int
    title: str
    element: Optional[ET.Element]
    paragraphs: List[str]


class GROBIDParser:
    """
    A class to parse academic papers using GROBID service.
    Returns a Paper object with hierarchical structure.
    """
    NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

    # Regex patterns for section numbering
    SECTION_PATTERN = re.compile(r'^((?:\d+\.)*\d+)\.?(?:\s+(.*?))?$')
    SECTION_NUMBER_PATTERN = re.compile(r'^((?:\d+\.)*\d+)\.\s+')
    
    # Keywords that often indicate non-section headings
    NON_SECTION_KEYWORDS = [
        'figure', 'fig', 'table', 'theorem', 'lemma', 'proposition', 
        'corollary', 'definition', 'remark', 'example', 'proof',
        'algorithm', 'equation', 'appendix'
    ]
    
    def __init__(self, grobid_url: str = "http://localhost:8070"):
        """
        Initialize the GROBID parser.
        
        Args:
            grobid_url: URL of the GROBID service (default: localhost:8070)
        """
        self.grobid_url = grobid_url        
        self.current_section_hierarchy = []
        self.citation_map = {}
    
    def process_pdf_to_xml(self, pdf_path: str) -> str:
        """
        Send PDF to GROBID and get TEI XML response.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            TEI XML string
        """
        url = f"{self.grobid_url}/api/processFulltextDocument"
        
        with open(pdf_path, 'rb') as f:
            files = {'input': f}
            while True:
                try:
                    time.sleep(2 * random.random())
                    response = requests.post(url, files=files)
                    response.raise_for_status()
                    return response.text
                except requests.exceptions.HTTPError as e:
                    if response.status_code != 503:
                        logging.error(f"{pdf_path} {e}")
                        return ""
                except Exception as e:
                    logging.error(f"{pdf_path} {e}")
                    return ""          
            
    def parse_xml(self, xml_content: str) -> Paper:
        """
        Parse GROBID TEI XML output into a Paper object.
        
        Args:
            xml_content: TEI XML string from GROBID
            
        Returns:
            Paper object containing the structured paper data
        """
        root = ET.fromstring(xml_content)
        
        # Create the root Paper object
        paper = Paper(name="root", father=None)
        
        # Extract metadata
        paper.title = self._extract_title(root)
        paper.author = self._extract_authors_string(root)
        
        # Extract references/bibliography
        paper.references = self._extract_references(root)
        
        # Extract abstract
        paper.abstract = self._extract_abstract(root, paper)

        # Extract references first
        self._extract_references(root, paper)
        
        # Extract body sections
        self._extract_body_sections(root, paper)

        paper.references = self.citation_map        
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
                self._parse_paragraph_element(p_elem, paragraph, paper.references)
                if paragraph.sentences:
                    abstract_section.add_paragraph(paragraph)
        
        # If no divs, try direct paragraphs
        if not abstract_section.paragraphs:
            for p_elem in abstract_elem.findall('.//tei:p', self.NS):
                paragraph = Paragraph(father=abstract_section)
                self._parse_paragraph_element(p_elem, paragraph, paper.references)
                if paragraph.sentences:
                    abstract_section.add_paragraph(paragraph)
        
        return abstract_section if abstract_section.paragraphs else None
   
    def _extract_body_sections(self, root: ET.Element, paper: Paper):
        """Extract body sections with hierarchical structure."""
        body = root.find('.//tei:text/tei:body', self.NS)
        if body is None:
            return
        
        # GROBID返回的XML文件往往不会按照你所期望的那样分好head，那最好的方法就是先提取整段文本再分割。
        pseudo_sections = []
        for div in body.findall('./tei:div', self.NS):
            sections = self._parse_div_element(div, paper.references)
            pseudo_sections.extend(sections)
    
    def _extract_text_from_element(self, element: ET.Element) -> str:
        """从元素中提取文本内容"""
        return ' '.join(element.itertext()).strip()

    def _parse_div_element(self, div_element: ET.Element, father: Section) -> List[HTMLSection]:
        """
        Parse a div element that may contain multiple sections within paragraphs.
        
        Args:
            div_elem: The div XML element
            parent: Parent Section object
            references: Dictionary of references for citation mapping
            
        Returns:
            List of Section objects
        """
        sections = []       
        # 和PDF parser中不同，处理明白段落和引用就行。
        current_head = None
        
        for child in div_element:
            if child.tag.endswith("head"):
                n_attr = child.get('n')
                text = self._extract_text_from_element(child)
                current_head = f"{n_attr} {text}" if n_attr else text
            
            else:
                paragraph = Paragraph(father=father)
                if current_head:
                    paragraph.add_sentence(Sentence(current_head, paragraph, []))
                self._parse_paragraph_element(child, paragraph)

        return sections
    
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
                ref_text = ''.join(element.itertext()).strip()  # "Yu et al." or "[1]"
                
                # Add citation marker
                if current_text:
                    text_parts.append(('text', ''.join(current_text)))
                    current_text.clear()

                if citation_id in self.citation_map:
                    self.citation_map[citation_id]['ref_text'] = ref_text
                    text_parts.append(('citation', self.citation_map[citation_id]))                
                
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
    
    def _extract_references(self, root: ET.Element, paper: Paper) -> dict:
        """Extract bibliography/reference list as a dictionary."""
        
        back = root.find('.//tei:text/tei:back', self.NS)
        if back is None: return
        
        for biblstruct in back.findall('.//tei:listBibl/tei:biblStruct', self.NS):
            ref_id = biblstruct.get('{http://www.w3.org/XML/1998/namespace}id', '')
            
            # Extract title
            title_element = biblstruct.find('.//tei:analytic/tei:title', self.NS)
            if title_element is None:
                title_element = biblstruct.find('.//tei:monogr/tei:title', self.NS)
            title = ''.join(title_element.itertext()).strip() if title_element is not None else ""
            
            # Extract authors
            ref_authors = []
            author_element = biblstruct.findall('.//tei:analytic/tei:author', self.NS)
            if author_element is None:
                author_element = biblstruct.findall('.//tei:monogr/tei:author', self.NS)
            for author in author_element:
                persname = author.find('.//tei:persName', self.NS)
                if persname is not None:
                    forename = persname.find('.//tei:forename', self.NS)
                    surname = persname.find('.//tei:surname', self.NS)
                    name = f"{forename.text if forename is not None else ''} {surname.text if surname is not None else ''}".strip()
                    if name: ref_authors.append(name)
            
            # Extract publication info
            monogr = biblstruct.find('.//tei:monogr', self.NS)
            journal = ""
            year = ""
            
            if monogr is not None:
                journal_element = monogr.find('.//tei:title', self.NS)
                if journal_element is not None: journal = ''.join(journal_element.itertext()).strip()
                
                date_element = monogr.find('.//tei:imprint/tei:date', self.NS)
                if date_element is not None: year = date_element.get('when', '')
            
            if ref_id:
                self.citation_map[ref_id] = {
                    'title': title,
                    'authors': ref_authors,
                    'journal': journal,
                    'year': year
                }
    
    def parse_pdf(self, pdf_path: str) -> Paper:
        """
        Complete pipeline: process PDF and parse results into Paper object.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Paper object with hierarchical structure
        """
        xml_content = self.process_pdf_to_xml(pdf_path)
        return self.parse_xml(xml_content)


# Example usage
if __name__ == "__main__":
    # Initialize parser (make sure GROBID is running on localhost:8070)
    parser = GROBIDParser()
    # with open("/data/tsyu/1710.03675.xml", encoding='utf-8') as f:
    #     paper = f.read()
    # paper = parser.parse_xml(paper)
    paper = parser.parse_pdf("/data/tsyu/survey_eval/crawled_papers/pdf/2306.16261.pdf")
    print(len(paper.children))
    print("=" * 50 + "Skeletion" + "=" * 50)
    x = paper.get_skeleton("all")
    # filename = "../crawled_papers/pdf/1710.03675.pdf"
