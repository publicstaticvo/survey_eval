"""
TODO:实现提取引用
"""

import io
import re
import time
import random
import logging
import requests
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool
from typing import Dict, List, Optional, Tuple
from paper_elements import Paper, Section, Paragraph, Sentence


@dataclass
class HTMLSection:
    numbers: List[int]
    level: int
    title: str
    element: Optional[ET.Element]
    paragraphs: List[str]


class PaperProcessInput(BaseModel):
    claim: str = Field(description="The text of the claim to be verified.")
    cited_paper: str = Field(description="The single, specific paper ID to verify the claim against.")


class GROBIDParser(BaseTool):
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
    
    def __init__(self, grobid_url: str = "http://localhost:8070", **kwargs):
        """
        Initialize the GROBID parser.
        
        Args:
            grobid_url: URL of the GROBID service (default: localhost:8070)
        """
        super().__init__(**kwargs)
        self.grobid_url = grobid_url        
        self.current_section_hierarchy = []
    
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
        
        # Extract body sections
        self._extract_body_sections(root, paper)
        
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
            sections = self._parse_div_with_complex_paragraphs(div, paper.references)
            pseudo_sections.extend(sections)
        
        # 将section按照层级转化为Section类，需要分割句子。
        try:
            # 没有标题的情形
            last_chapter = [1]
            current_level = 1  # 当前节点的层级
            father_section = paper  # 当前节点的father
            if pseudo_sections[0].level == -1: paper.has_section_index = False
            for section in pseudo_sections:
                # 需要回溯
                if paper.has_section_index:
                    while current_level > section.level or (current_level >= 2 and section.numbers[current_level - 2] != last_chapter[current_level - 2]):
                        father_section = father_section.father
                        current_level -= 1
                    # 一次进两级，说明丢失了其中的一级
                    while current_level < section.level:
                        new_section = Section(name="", father=father_section)
                        father_section.add_child(new_section)
                        father_section = new_section
                        current_level += 1

                    last_chapter = section.numbers

                # 构建新的Section
                new_section = Section(section.title, father_section)
                # 将section.paragraph转化成Paragraph。先实现简单的句子分割，等以后再实现提取引用。
                for paragraph_text in section.paragraphs:
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph_text)
                    paragraph = Paragraph(father=new_section)
                    for sentence in sentences:
                        sentence = Sentence(sentence, paragraph)
                        paragraph.add_sentence(sentence)
                    new_section.add_paragraph(paragraph)
                father_section.add_child(new_section)

                if paper.has_section_index:
                    # 默认将下一个Section设为当前Section的subsection，即默认进一级
                    current_level = section.level + 1
                    father_section = new_section
        except Exception as e:
            logging.warning(f" get {e}, fallback to parse paragraphs")
            self.fallback_parse_paragraphs(paper, body, paper.references)
    
    def _extract_text_from_element(self, element: ET.Element) -> str:
        """从元素中提取文本内容"""
        return ' '.join(element.itertext()).strip()

    def _is_section_title(self, element: ET.Element) -> bool:
        """
        判断head元素是否是章节标题
        """
        text = self._extract_text_from_element(element)
        n_attr = element.get('n')
        if n_attr and n_attr.strip(): text = f"{n_attr.strip()} {text}"

        text_lower = text.lower().strip()
        if any(text_lower.startswith(x) for x in self.NON_SECTION_KEYWORDS): 
            return None, None
        
        # 检查是否包含典型的章节模式
        if self.SECTION_NUMBER_PATTERN.match(text):
            return True
        
        return False

    def _parse_section_title_from_p(self, text: str, element_type: str, n_attr: Optional[str] = None) -> Tuple[List[int], str]:
        """
        判断p元素的内容是否含有标题
        """
        if element_type == "head" and n_attr and n_attr.strip():
            text = f"{n_attr.strip()} {text}"
        text_lower = text.lower().strip()
        if any(text_lower.startswith(x) for x in self.NON_SECTION_KEYWORDS): 
            return None, None
        
        numbers, title = self._parse_section_number(text, element_type)
        if numbers and (element_type == "head" or self._compare_section_levels(numbers) != 2):
            return numbers, title
        
        return None, None
    
    def _parse_section_number(self, text: str, element_type: str, n_attr: Optional[str] = None) -> Tuple[List[int], str]:
        """
        解析章节号和标题文本
        参数:
            text: 标题文本
            n_attr: head元素的n属性值
        返回: (章节号列表, 标题文本)
        """
        # 优先使用n属性中的章节号
        text = text.strip()
        if element_type == "head" and n_attr and n_attr.strip():
            # 清理n属性中的章节号
            n_attr_clean = n_attr.strip().rstrip('.')
            match = self.SECTION_PATTERN.match(n_attr_clean)
            if match:
                numbers = match.group(1).split('.')
                # 如果n属性中有章节号，但文本中没有，使用文本作为标题
                title = text if text else (match.group(2).strip() if match.group(2) else "")
                return [int(num) for num in numbers], title
        
        # 如果没有n属性或n属性中没有有效的章节号，从文本中解析
        if text:
            if element_type == "head":
                section_id = self.SECTION_PATTERN.findall(text)
                if section_id:
                    section_id = section_id[0]
                    numbers = section_id[0].split('.')
                    title = section_id[1].strip() if len(section_id) >= 2 else ""
                    return [int(num) for num in numbers], title
            else:
                section_id = self.SECTION_NUMBER_PATTERN.match(text)
                if section_id:
                    numbers = section_id.group(1)
                    text = text.replace(numbers, "", 1).lstrip()
                    if text[0] == ".": text = text[1:].lstrip()
                    text_split = re.split(r'(?<=[.!?])\s+', text, 1)
                    return [int(num) for num in numbers.split('.')], text_split
        
        return None, text
    
    def _compare_section_levels(self, new_numbers: List[int]) -> int:
        """
        比较新章节号与当前层级的关系
        返回: 
          -1: 新章节是更高级别
          0: 同级
          1: 子级
        """
        if not self.current_section_hierarchy:
            return 1  # 第一个章节
        
        current = self.current_section_hierarchy[-1].numbers
        
        # 检查是否是直接子级
        if len(new_numbers) == len(current) + 1 and new_numbers[:-1] == current and new_numbers[-1] in [0, 1]:
            return 1
        
        # 检查同级
        if len(new_numbers) == len(current):
            if new_numbers[:-1] == current[:-1] and new_numbers[-1] == current[-1] + 1:
                return 0
        
        # 检查更高级别
        for i, j in zip(current, new_numbers):
            if j == i + 1: return -1
            elif i != j: break
        
        # 章节序号错误请检查
        return 2
    
    def _update_section_hierarchy(self, numbers: List[int], title: str, element: ET.Element, text: str = "") -> HTMLSection:
        """
        更新章节层级并返回当前章节信息
        """
        if not numbers:
            # 没有明确章节号，作为当前章节的子级处理
            if self.current_section_hierarchy:
                parent_numbers = self.current_section_hierarchy[-1].numbers
                numbers = parent_numbers + [1]  # 默认作为第一个子节
            else:
                numbers = [1]  # 第一个章节
        
        relation = self._compare_section_levels(numbers)
        
        if relation == -1:  # 更高级别
            # 回溯到合适层级
            while self.current_section_hierarchy:
                current = self.current_section_hierarchy[-1].numbers
                if len(numbers) <= len(current) and all(a == b for a, b in zip(numbers, current[:len(numbers)])):
                    break
                self.current_section_hierarchy.pop()
        
        elif relation == 2:
            # 寻找最近的共同祖先
            new_hierarchy = []
            for section in self.current_section_hierarchy:
                if (len(section.numbers) <= len(numbers) and 
                    section.numbers == numbers[:len(section.numbers)]):
                    new_hierarchy.append(section)
                else:
                    break
            self.current_section_hierarchy = new_hierarchy
        
        # 创建新章节
        section_data = HTMLSection(numbers, len(numbers), title, element, [text] if text else [])        
        self.current_section_hierarchy.append(section_data)
        return section_data

    def _parse_div_with_complex_paragraphs(self, div_element: ET.Element, references: dict) -> List[HTMLSection]:
        """
        Parse a div element that may contain multiple sections within paragraphs.
        Handles cases where section titles appear in <p> or <head> elements.
        
        Args:
            div_elem: The div XML element
            parent: Parent Section object
            references: Dictionary of references for citation mapping
            
        Returns:
            List of Section objects
        """
        sections = []       
        # 文本标题有几种表示形式：
        # 1. <p>1. Introduction. 正文
        # 2. <head n='2.1'>标题</head>
        # 3. <head>3.2.</head><p>标题。正文
        # 一个<div>有可能有多个section；带<head>的不一定是section。
        current_section = None
        has_section_index = (not self.current_section_hierarchy or self.current_section_hierarchy[-1].level >= 0)
        
        for child in div_element:
            if child.tag.endswith("head"):
                text = self._extract_text_from_element(child)
                n_attr = child.get('n')
                numbers, title = self._parse_section_title_from_p(text, 'head', n_attr)
                if has_section_index and numbers:
                    current_section = self._update_section_hierarchy(numbers, title, child)
                    sections.append(current_section)
                else:
                    # 检查是否为全文都没有段落标号的情况。
                    # all(not text.lower().startswith(x) for x in self.NON_SECTION_KEYWORDS)
                    if not self.current_section_hierarchy or not has_section_index:
                        # 此时所有section视为同级。
                        has_section_index = False
                        current_section = HTMLSection([], -1, text, child, [])
                        self.current_section_hierarchy.append(current_section)
                        sections.append(current_section)
                    # 非章节标题的head，忽略或作为当前章节的段落处理。                    
                    elif current_section:
                        text = self._extract_text_from_element(child)
                        if text:
                            current_section.paragraphs.append(text)
            
            elif child.tag == f"{{{self.NS}}}p":
                # 首先判断是否含有标题以及标题是否合法。self._compare_section_levels(numbers)
                text = self._extract_text_from_element(child)
                numbers, title = self._parse_section_title_from_p(text, 'p') if has_section_index else (None, None)
                if numbers:
                    # 情形1
                    title, text = title
                    current_section = self._update_section_hierarchy(numbers, title, child, text)
                    sections.append(current_section)
                else:
                    # 普通段
                    if current_section and has_section_index and not current_section.title and '.' in text:
                        # 情形3
                        current_section.title, text = text.split(".", 1)
                        text = text.lstrip()
                    if text:
                        # 如果有当前章节，添加到当前章节
                        if current_section:
                            current_section.paragraphs.append(text)
                        elif self.current_section_hierarchy:
                            self.current_section_hierarchy[-1].paragraphs.append(text)
                        else:
                            # 创建第一个章节
                            title, section_content = text.split(".", 1)
                            default_section = HTMLSection([1], 1, title, None, [section_content])
                            self.current_section_hierarchy.append(default_section)
                            sections.append(default_section)
                            current_section = default_section
                
            else:
                # 视为普通段落
                text = self._extract_text_from_element(child)
                if current_section:                    
                    current_section.paragraphs.append(text)
                elif self.current_section_hierarchy:
                    self.current_section_hierarchy[-1].paragraphs.append(text)

        return sections
    
    def fallback_parse_paragraphs(self, paper: Paper, body: ET.Element, references: dict):
        """
        有些文章没有章节号不能判断章节从属关系，会导致正常的判断流程出错。
        该函数为应急方案，假如章节序号出现错乱，则去掉章节号，将所有内容视为自然段。
        """        
        for div_element in body.findall('./tei:div', self.NS):
            text_buffer = ""
            for child in div_element:
                text = self._extract_text_from_element(child)
                if child.tag.endswith("head"):
                    n_attr = child.get("n")
                    if n_attr: text = f"{n_attr} {text}"
                    text_buffer += text
                else:
                    text = f"{text_buffer} {text}"
                    text_buffer = ""
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    paragraph = Paragraph(father=paper)
                    for sentence in sentences:
                        sentence = Sentence(sentence, paragraph)
                        paragraph.add_sentence(sentence)
                    paper.add_paragraph(paragraph)

            if text_buffer:
                sentences = re.split(r'(?<=[.!?])\s+', text_buffer)
                paragraph = Paragraph(father=paper)
                for sentence in sentences:
                    sentence = Sentence(sentence, paragraph)
                    paragraph.add_sentence(sentence)
                paper.add_paragraph(paragraph)
    
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
        
        def process_elem(elem: ET.Element, depth=0):
            # Add text before element
            if elem.text:
                current_text.append(elem.text)
            
            # Handle citation references
            if elem.tag == f"{{{self.NS['tei']}}}ref" and elem.get('type') == 'bibr':
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
        
        back = root.find('.//tei:text/tei:back', self.NS)
        if back is None:
            return references
        
        for biblstruct in back.findall('.//tei:listBibl/tei:biblStruct', self.NS):
            ref_id = biblstruct.get('{http://www.w3.org/XML/1998/namespace}id', '')
            
            # Extract title
            title_elem = biblstruct.find('.//tei:analytic/tei:title', self.NS)
            if title_elem is None:
                title_elem = biblstruct.find('.//tei:monogr/tei:title', self.NS)
            title = ''.join(title_elem.itertext()).strip() if title_elem is not None else ""
            
            # Extract authors
            ref_authors = []
            for author in biblstruct.findall('.//tei:analytic/tei:author', self.NS):
                persname = author.find('.//tei:persName', self.NS)
                if persname is not None:
                    forename = persname.find('.//tei:forename', self.NS)
                    surname = persname.find('.//tei:surname', self.NS)
                    name = f"{forename.text if forename is not None else ''} {surname.text if surname is not None else ''}".strip()
                    if name:
                        ref_authors.append(name)
            
            # Extract publication info
            monogr = biblstruct.find('.//tei:monogr', self.NS)
            journal = ""
            year = ""
            
            if monogr is not None:
                journal_elem = monogr.find('.//tei:title', self.NS)
                journal = ''.join(journal_elem.itertext()).strip() if journal_elem is not None else ""
                
                date_elem = monogr.find('.//tei:imprint/tei:date', self.NS)
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
        xml_content = self.process_pdf_to_xml(pdf_path)
        return self.parse_xml(xml_content)
