import requests
from typing import Dict, List, Optional
from pathlib import Path
import xml.etree.ElementTree as ET
import re

from paper_elements import Paper, Section, Paragraph, Sentence


class GROBIDParser:
    """
    A class to parse academic papers using GROBID service.
    Returns a Paper object with hierarchical structure.
    """
    
    def __init__(self, grobid_url: str = "http://localhost:8070"):
        """
        Initialize the GROBID parser.
        
        Args:
            grobid_url: URL of the GROBID service (default: localhost:8070)
        """
        self.grobid_url = grobid_url
        self.ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    def process_pdf(self, pdf_path: str) -> str:
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
            response = requests.post(url, files=files)
            response.raise_for_status()
            
        return response.text
    
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
        
        # Extract body sections
        self._extract_body_sections(root, paper)
        
        return paper
    
    def _extract_title(self, root: ET.Element) -> str:
        """Extract paper title."""
        title_elem = root.find('.//tei:titleStmt/tei:title', self.ns)
        return title_elem.text if title_elem is not None else ""
    
    def _extract_authors_string(self, root: ET.Element) -> str:
        """Extract authors as a formatted string."""
        authors = []
        
        for author in root.findall('.//tei:sourceDesc//tei:author', self.ns):
            persname = author.find('.//tei:persName', self.ns)
            if persname is not None:
                forename = persname.find('.//tei:forename[@type="first"]', self.ns)
                surname = persname.find('.//tei:surname', self.ns)
                
                first = forename.text if forename is not None else ""
                last = surname.text if surname is not None else ""
                full_name = f"{first} {last}".strip()
                if full_name:
                    authors.append(full_name)
        
        return ", ".join(authors)
    
    def _extract_abstract(self, root: ET.Element, paper: Paper) -> Optional[Section]:
        """Extract abstract as a Section."""
        abstract_elem = root.find('.//tei:profileDesc/tei:abstract', self.ns)
        if abstract_elem is None:
            return None
        
        abstract_section = Section(name="Abstract", father=paper)
        
        # Extract paragraphs from abstract
        for div in abstract_elem.findall('.//tei:div', self.ns):
            for p_elem in div.findall('.//tei:p', self.ns):
                paragraph = Paragraph(father=abstract_section)
                self._parse_paragraph_element(p_elem, paragraph, paper.references)
                if paragraph.sentences:
                    abstract_section.add_paragraph(paragraph)
        
        # If no divs, try direct paragraphs
        if not abstract_section.paragraphs:
            for p_elem in abstract_elem.findall('.//tei:p', self.ns):
                paragraph = Paragraph(father=abstract_section)
                self._parse_paragraph_element(p_elem, paragraph, paper.references)
                if paragraph.sentences:
                    abstract_section.add_paragraph(paragraph)
        
        return abstract_section if abstract_section.paragraphs else None
    
    def _extract_body_sections(self, root: ET.Element, paper: Paper):
        """Extract body sections with hierarchical structure."""
        body = root.find('.//tei:text/tei:body', self.ns)
        if body is None:
            return
        
        # Process top-level divs
        for div in body.findall('./tei:div', self.ns):
            section = self._parse_section(div, paper, paper.references)
            if section:
                paper.add_child(section)
    
    def _parse_section(self, div_elem: ET.Element, parent: Section, references: dict) -> Optional[Section]:
        """
        Recursively parse a section (div) element.
        
        Args:
            div_elem: The div XML element
            parent: Parent Section object
            references: Dictionary of references for citation mapping
            
        Returns:
            Section object or None
        """
        # Extract heading
        head = div_elem.find('./tei:head', self.ns)
        heading = ''.join(head.itertext()).strip() if head is not None else "Untitled Section"
        
        section = Section(name=heading, father=parent)
        
        # Process direct paragraphs (not in subsections)
        for p_elem in div_elem.findall('./tei:p', self.ns):
            paragraph = Paragraph(father=section)
            self._parse_paragraph_element(p_elem, paragraph, references)
            if paragraph.sentences:
                section.add_paragraph(paragraph)
        
        # Process subsections recursively
        for subdiv in div_elem.findall('./tei:div', self.ns):
            subsection = self._parse_section(subdiv, section, references)
            if subsection:
                section.add_child(subsection)
        
        return section if (section.paragraphs or section.children) else None
    
    def _parse_paragraph_element(self, p_elem: ET.Element, paragraph: Paragraph, references: dict):
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
        
        def process_elem(elem, depth=0):
            # Add text before element
            if elem.text:
                current_text.append(elem.text)
            
            # Handle citation references
            if elem.tag == f"{{{self.ns['tei']}}}ref" and elem.get('type') == 'bibr':
                citation_id = elem.get('target', '').replace('#', '')
                ref_text = ''.join(elem.itertext()).strip()
                
                # Add citation marker
                if current_text:
                    text_parts.append(('text', ''.join(current_text)))
                    current_text.clear()
                
                text_parts.append(('citation', citation_id))
            else:
                # Recursively process child elements
                for child in elem:
                    process_elem(child, depth + 1)
            
            # Add text after element
            if elem.tail and depth > 0:
                current_text.append(elem.tail)
        
        # Process the paragraph element
        process_elem(p_elem)
        
        # Add any remaining text
        if current_text:
            text_parts.append(('text', ''.join(current_text)))
        
        # Split into sentences and assign citations
        self._create_sentences(text_parts, paragraph)
    
    def _create_sentences(self, text_parts: List[tuple], paragraph: Paragraph):
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
                
                for i, sent in enumerate(sentences):
                    sent = sent.strip()
                    if not sent:
                        continue
                    
                    if i == 0:
                        # Continuation of current sentence
                        current_sentence += sent
                    else:
                        # Create sentence for the previous one
                        if current_sentence:
                            sentence_obj = Sentence(
                                text=current_sentence.strip(),
                                father=paragraph,
                                citations=current_citations.copy()
                            )
                            paragraph.add_sentence(sentence_obj)
                        
                        # Start new sentence
                        current_sentence = sent
                        current_citations = []
                
            elif part_type == 'citation':
                current_citations.append(content)
        
        # Add the last sentence
        if current_sentence.strip():
            sentence_obj = Sentence(
                text=current_sentence.strip(),
                father=paragraph,
                citations=current_citations
            )
            paragraph.add_sentence(sentence_obj)
    
    def _extract_references(self, root: ET.Element) -> dict:
        """Extract bibliography/reference list as a dictionary."""
        references = {}
        
        back = root.find('.//tei:text/tei:back', self.ns)
        if back is None:
            return references
        
        for biblstruct in back.findall('.//tei:listBibl/tei:biblStruct', self.ns):
            ref_id = biblstruct.get('{http://www.w3.org/XML/1998/namespace}id', '')
            
            # Extract title
            title_elem = biblstruct.find('.//tei:analytic/tei:title', self.ns)
            if title_elem is None:
                title_elem = biblstruct.find('.//tei:monogr/tei:title', self.ns)
            title = ''.join(title_elem.itertext()).strip() if title_elem is not None else ""
            
            # Extract authors
            ref_authors = []
            for author in biblstruct.findall('.//tei:analytic/tei:author', self.ns):
                persname = author.find('.//tei:persName', self.ns)
                if persname is not None:
                    forename = persname.find('.//tei:forename', self.ns)
                    surname = persname.find('.//tei:surname', self.ns)
                    name = f"{forename.text if forename is not None else ''} {surname.text if surname is not None else ''}".strip()
                    if name:
                        ref_authors.append(name)
            
            # Extract publication info
            monogr = biblstruct.find('.//tei:monogr', self.ns)
            journal = ""
            year = ""
            
            if monogr is not None:
                journal_elem = monogr.find('.//tei:title', self.ns)
                journal = ''.join(journal_elem.itertext()).strip() if journal_elem is not None else ""
                
                date_elem = monogr.find('.//tei:imprint/tei:date', self.ns)
                year = date_elem.get('when', '') if date_elem is not None else ""
            
            if ref_id:
                references[ref_id] = {
                    'title': title,
                    'authors': ref_authors,
                    'journal': journal,
                    'year': year
                }
        
        return references
    
    def parse_pdf(self, pdf_path: str) -> Paper:
        """
        Complete pipeline: process PDF and parse results into Paper object.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Paper object with hierarchical structure
        """
        xml_content = self.process_pdf(pdf_path)
        return self.parse_xml(xml_content)


# Example usage
if __name__ == "__main__":
    # Initialize parser (make sure GROBID is running on localhost:8070)
    parser = GROBIDParser(grobid_url="http://localhost:8070")
    
    # Parse a PDF file
    pdf_file = "path/to/your/paper.pdf"
    
    try:
        paper = parser.parse_pdf(pdf_file)
        
        print(f"Title: {paper.title}\n")
        print(f"Authors: {paper.author}\n")
        
        if paper.abstract:
            print(f"Abstract sections: {len(paper.abstract.paragraphs)}")
            print(f"Abstract sentences: {len(paper.abstract.get_sentences())}\n")
        
        print(f"Number of main sections: {len(paper.children)}")
        print(f"Total sentences in paper: {len(paper.get_sentences())}")
        print(f"Number of references: {len(paper.references)}\n")
        
        # Print section structure
        print("Paper Structure:")
        for i, section in enumerate(paper.children, 1):
            print(f"{i}. {section.name}")
            print(f"   Paragraphs: {len(section.paragraphs)}, Subsections: {len(section.children)}")
            for j, subsection in enumerate(section.children, 1):
                print(f"   {i}.{j}. {subsection.name}")
        
        # Print some sample sentences with citations
        print("\nSample sentences with citations:")
        all_sentences = paper.get_sentences()
        sentences_with_citations = [s for s in all_sentences if s.citations][:5]
        for sent in sentences_with_citations:
            print(f"- {sent}")
        
        # Print some references
        print("\nSample References:")
        for i, (ref_id, ref_data) in enumerate(list(paper.references.items())[:3], 1):
            print(f"{i}. [{ref_id}] {ref_data['title']}")
            print(f"   Authors: {', '.join(ref_data['authors'])}")
            print(f"   {ref_data['journal']}, {ref_data['year']}\n")
            
    except Exception as e:
        print(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()