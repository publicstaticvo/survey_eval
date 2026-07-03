import re
import xml.etree.ElementTree as ET
from typing import List, Optional
# import sys
# sys.path.insert(0, r"P:\AI4S\survey_eval\agent\tools\utility\grobidpdf")

try:
    from .paper_elements import Paper, Section, Paragraph, Sentence
except ImportError:
    from paper_elements import Paper, Section, Paragraph, Sentence


class PaperParser:
    NS = {'tei': 'http://www.tei-c.org/ns/1.0'}
    HEAD_INDEX_RE = re.compile(r"^\s*(?P<index>(?:\d+\.)*\d+)\.?\s+(?P<name>.+?)\s*$")
    
    # Keywords that often indicate non-section headings
    NON_SECTION_KEYWORDS = [
        'figure', 'fig', 'table', 'theorem', 'lemma', 'proposition', 
        'corollary', 'definition', 'remark', 'example', 'proof',
        'algorithm', 'equation', 'appendix'
    ]
            
    def parse(self, xml_content: str, mode: str = "casual") -> Paper:
        root = ET.fromstring(xml_content)
        paper = self._parse_root(root, mode)
        if mode == "strict" and self._strict_parse_failed(paper):
            raise ValueError("Strict GROBID parsing produced no usable section structure")
        return paper

    def _parse_root(self, root: ET.Element, mode: str) -> Paper:
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
        paper.references = {x['key']: x for x in self._citation_map.values() if 'key' in x}  
        return paper

    def _section_has_content(self, section: Section) -> bool:
        return bool(section.paragraphs) or any(self._section_has_content(child) for child in section.children)

    def _strict_parse_failed(self, paper: Paper) -> bool:
        return bool(paper.paragraphs) or not any(self._section_has_content(section) for section in paper.children)

    def _format_citation_key(self, value: str) -> str:
        value = (value or "").strip()
        if not value:
            return ""
        return value if value.startswith("#") else f"#{value}"
    

    def _extract_biblstruct_info(self, biblstruct: ET.Element) -> dict:
        def text_of(xpath: str) -> str:
            elem = biblstruct.find(xpath, self.NS)
            return "".join(elem.itertext()).strip() if elem is not None else ""

        authors = []
        for author in biblstruct.findall(".//tei:author", self.NS):
            persname = author.find(".//tei:persName", self.NS)
            if persname is None:
                continue
            forenames = [
                "".join(node.itertext()).strip()
                for node in persname.findall(".//tei:forename", self.NS)
                if "".join(node.itertext()).strip()
            ]
            surname = persname.find(".//tei:surname", self.NS)
            surname_text = "".join(surname.itertext()).strip() if surname is not None else ""
            full_name = " ".join(part for part in [*forenames, surname_text] if part).strip()
            if full_name:
                authors.append({"forenames": forenames, "surname": surname_text, "name": full_name})

        ids = {}
        for idno in biblstruct.findall(".//tei:idno", self.NS):
            id_type = (idno.get("type") or "").strip()
            id_value = "".join(idno.itertext()).strip()
            if id_type and id_value:
                ids[id_type] = id_value

        date = biblstruct.find(".//tei:date", self.NS)
        year = (date.get("when") or "".join(date.itertext()).strip()) if date is not None else ""
        imprint = biblstruct.find(".//tei:imprint", self.NS)
        publisher = ""
        pub_place = ""
        if imprint is not None:
            publisher_elem = imprint.find(".//tei:publisher", self.NS)
            pub_place_elem = imprint.find(".//tei:pubPlace", self.NS)
            publisher = "".join(publisher_elem.itertext()).strip() if publisher_elem is not None else ""
            pub_place = "".join(pub_place_elem.itertext()).strip() if pub_place_elem is not None else ""

        raw_text = " ".join("".join(part.split()) for part in biblstruct.itertext() if "".join(part.split()))

        return {
            "xml_id": biblstruct.get("{http://www.w3.org/XML/1998/namespace}id", ""),
            "title": text_of(".//tei:analytic/tei:title") or text_of(".//tei:monogr/tei:title"),
            "authors": authors,
            "year": year,
            "journal": text_of(".//tei:monogr/tei:title[@level=\"j\"]") or text_of(".//tei:monogr/tei:title"),
            "booktitle": text_of(".//tei:monogr/tei:title[@level=\"m\"]"),
            "volume": text_of(".//tei:biblScope[@unit=\"volume\"]"),
            "issue": text_of(".//tei:biblScope[@unit=\"issue\"]"),
            "pages": text_of(".//tei:biblScope[@unit=\"page\"]"),
            "publisher": publisher,
            "pub_place": pub_place,
            "note": text_of(".//tei:note"),
            "doi": ids.get("doi", ""),
            "url": ids.get("url", ""),
            "ids": ids,
            "raw_text": raw_text,
        }

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
        self._last_section_index = None
        for div in body.findall('./tei:div', self.NS):
            self._parse_div_element(div, mode)

    def _parse_div_element(self, div_element: ET.Element, mode: str):
        """Parse a single TEI div."""
        current_div_section = None
        pending_head_texts: list[str] = []
        for child in div_element:
            if child.tag == f"{{{self.NS['tei']}}}head":
                n_attr = child.get('n', "")
                text = ' '.join(child.itertext()).strip()
                if mode == "strict":
                    if not re.fullmatch(r"\d+(?:\.\d+)*", n_attr or ""):
                        pending_head_texts.append(text)
                        continue
                    section_index = tuple(int(part) for part in n_attr.split("."))
                    if getattr(self, "_last_section_index", None) is not None and section_index <= self._last_section_index:
                        raise ValueError(f"Section index is not strictly increasing: {n_attr}")
                    self._last_section_index = section_index
                    while n_attr.count(".") + 1 < len(self._current_section_hierarchy):
                        self._current_section_hierarchy.pop()
                section = Section(name=text, father=self._current_section_hierarchy[-1])
                self._current_section_hierarchy[-1].add_child(section)
                if mode == "strict":
                    self._current_section_hierarchy.append(section)
                else:
                    current_div_section = section

            elif child.tag == f"{{{self.NS['tei']}}}p":
                paragraph_owner = current_div_section or self._current_section_hierarchy[-1]
                paragraph = Paragraph(father=paragraph_owner)
                prefix_text = ' '.join(pending_head_texts).strip()
                pending_head_texts.clear()
                self._parse_paragraph_element(child, paragraph, prefix_text=prefix_text)
                if paragraph.sentences:
                    paragraph_owner.add_paragraph(paragraph)

    def _parse_paragraph_element(self, p_element: ET.Element, paragraph: Paragraph, prefix_text: str = ""):
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
        if prefix_text:
            prefix_text = prefix_text.strip()
            if prefix_text and prefix_text[-1] not in ".!?":
                prefix_text += "."
            text_parts.append(('text', prefix_text + " "))
        
        def process_elem(element: ET.Element, depth=0):
            # Add text before element
            if element.text:
                current_text.append(element.text)
            
            # Handle citation references
            if element.tag == f"{{{self.NS['tei']}}}ref" and element.get('type') == 'bibr':
                citation_id = self._format_citation_key(element.get('target', ''))
                
                # Add citation marker
                if current_text:
                    text_parts.append(('text', ''.join(current_text)))
                    current_text.clear()

                if citation_id:
                    ref_id = citation_id.lstrip('#')
                    if ref_id in self._citation_map:
                        self._citation_map[ref_id]['key'] = citation_id
                        text_parts.append(('citation', self._citation_map[ref_id]))
                
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
        full_text = ""
        citation_markers = []
        for part_type, content in text_parts:
            if part_type == 'text':
                full_text += content
            elif part_type == 'citation':
                marker = f"CITMARK{len(citation_markers)}"
                citation_markers.append({"marker": marker, "citation": content})
                full_text += f" {marker} "

        for sentence_text in self._split_into_sentences(full_text):
            sentence_citations = []
            for marker_info in citation_markers:
                if marker_info["marker"] in sentence_text:
                    sentence_citations.append(marker_info["citation"])
                    sentence_text = sentence_text.replace(marker_info["marker"], " ")

            unique_citations = []
            seen_citation_keys = set()
            for citation in sentence_citations:
                key = citation.get("key") if isinstance(citation, dict) else str(citation)
                if key and key not in seen_citation_keys:
                    seen_citation_keys.add(key)
                    unique_citations.append(citation)

            sentence_text = re.sub(r"\s+", " ", sentence_text).strip()
            sentence_text = re.sub(r"\s+([,.;:!?])", r"\1", sentence_text)
            if sentence_text or unique_citations:
                paragraph.add_sentence(Sentence(sentence_text, paragraph, unique_citations))

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using the same scan strategy as the LaTeX parser."""
        text = text.strip()
        if not text:
            return []

        candidates = self._scan_sentence_candidates(text)
        result = []
        for sentence in candidates:
            sentence = sentence.strip()
            if not sentence:
                continue
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
        number_marker = re.search(r"(\d+)\.$", text[:idx + 1])
        if number_marker and idx + 1 < len(text) and text[idx + 1].isspace():
            prefix = text[:number_marker.start(1)].rstrip()
            if not prefix or prefix[-1] in ".!?:;([{":
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
        if re.match(r"(?:[ivxlcdm]+|[a-z])\)", stripped, flags=re.IGNORECASE):
            return False
        return bool(stripped) and "a" <= stripped[0] <= "z"
    
    def _extract_references(self, root: ET.Element) -> dict:
        """Extract bibliography/reference list as a dictionary."""        
        self._citation_map = {}
        back = root.find('.//tei:text/tei:back', self.NS)
        if back is None: return
        
        for biblstruct in back.findall('.//tei:listBibl/tei:biblStruct', self.NS):   
            if ref_id := biblstruct.get('{http://www.w3.org/XML/1998/namespace}id', ""):
                self._citation_map[ref_id] = self._extract_biblstruct_info(biblstruct)

    def _clean_title(self, value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip()

    def _title_record(self, section_index: str, section_name: str) -> dict[str, str] | None:
        section_index = self._clean_title(section_index)
        section_name = self._clean_title(section_name)
        if not section_name:
            return None
        return {"section_index": section_index, "section_name": section_name}

    def _split_indexed_title(self, text: str, attr_index: str = "") -> dict[str, str] | None:
        text = self._clean_title(text)
        attr_index = self._clean_title(attr_index)
        if attr_index:
            name = re.sub(rf"^\s*{re.escape(attr_index)}\s+\.?\s*", "", text).strip()
            return self._title_record(attr_index, name or text)
        match = self.HEAD_INDEX_RE.match(text)
        if match:
            return self._title_record(match.group("index"), match.group("name"))
        return self._title_record("", text)

    def _unique_title_records(self, records: List[dict[str, str]]) -> List[dict[str, str]]:
        has_index = any(item.get("section_index") for item in records)
        unique, seen_indexes, seen_names = [], set(), set()
        for item in records:
            section_index = item.get("section_index", "")
            section_name = item.get("section_name", "")
            if has_index:
                if not section_index or section_index in seen_indexes:
                    continue
                seen_indexes.add(section_index)
            else:
                key = section_name.lower()
                if key in seen_names:
                    continue
                seen_names.add(key)
            unique.append(item)
        return unique

    def get_titles(self, xml_content: str) -> List[dict[str, str]]:
        root = ET.fromstring(xml_content)
        body = root.find('.//tei:text/tei:body', self.NS)
        if body is None:
            return []

        heads = []
        for head in body.findall(".//tei:head", self.NS):
            text = self._clean_title(" ".join(head.itertext()))
            if not text:
                continue
            record = self._split_indexed_title(text, head.get("n", ""))
            if record:
                heads.append(record)

        indexed = [item for item in heads if item.get("section_index")]
        if indexed:
            records = indexed
            existing = {item["section_index"] for item in indexed}
            for paragraph in body.findall(".//tei:div/tei:p", self.NS):
                text = self._clean_title(" ".join(paragraph.itertext()))
                record = self._split_indexed_title(text)
                if record and record.get("section_index") and record["section_index"] not in existing:
                    records.append(record)
                    existing.add(record["section_index"])
        else:
            records = heads

        return self._unique_title_records(records)


if __name__ == "__main__":
    import json
    with open("P:\\AI4S\\survey_eval\\train_letor\\paper.xml", encoding="utf-8") as f:
        xml_content = f.read()
    parser = PaperParser()
    paper = parser.parse(xml_content, mode="strict").get_skeleton()
    with open("P:\\AI4S\\survey_eval\\train_letor\\paper.json", 'w', encoding="utf-8") as f:
        json.dump(paper, f, indent=2, ensure_ascii=False)
    print(f"Sections {len(paper['sections'])}")


