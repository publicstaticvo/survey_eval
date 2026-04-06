import json
from typing import Any, List, Optional, Union, Dict
from dataclasses import dataclass, field


@dataclass
class LatexSentence:
    """Represents a single sentence with its citations"""
    text: str
    citations: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            'text': self.text,
            'citations': self.citations
        }
    
    def __repr__(self):
        return self.text

    def get_skeleton(self) -> Dict[str, Union[str, List[str]]]:
        return {
            'text': self.text,
            'citations': self.citations,
            'environment_type': 'text'
        }


@dataclass
class LatexEnvironment:
    """Represents a Latex environment block (theorem, remark, tikzpicture, etc.)"""
    environment_name: str
    text: str
    citations: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            'type': 'latex_environment',
            'environment': self.environment_name,
            'content': self.text,
            'citations': self.citations
        }
    
    def __repr__(self):
        return self.text
        # return f"\\begin{{{self.environment_name}}}\n{self.text}\n\\end{{{self.environment_name}}}\n"

    def get_skeleton(self) -> Dict[str, Union[str, List[str]]]:
        return {
            'text': self.text,
            'citations': self.citations,
            'environment_type': self.environment_name
        }


def debug_sentences(sentences: List[Union[LatexSentence, LatexEnvironment]], start_id: int = 0):
    output = ""
    for i, sentence in enumerate(sentences):
        if isinstance(sentence, LatexSentence):
            sentence = sentence.to_dict()
            s = f"{i + start_id}\t{sentence['text']}\n"
            if sentence['citations']:
                s += f"-\tCitations: {', '.join(sentence['citations'])}\n"
        else:
            s = f"{i + start_id}\t{sentence.__repr__()}\n"
        output += s
    return output


@dataclass
class LatexParagraph:
    r"""Represents a paragraph (either \paragraph command or text block split by \n\n)"""
    sentences: List[Union[LatexSentence, LatexEnvironment]] = field(default_factory=list)
    name: Optional[str] = None  # Only set if defined by \paragraph
    
    def add_sentence(self, sentence: Union[LatexSentence, LatexEnvironment]):
        self.sentences.append(sentence)

    def get_skeleton(self) -> List[Dict[str, Union[str, List[str]]]]:
        return [sentence.get_skeleton() for sentence in self.sentences]
    
    def to_dict(self):
        result = {'sentences': [s.to_dict() for s in self.sentences]}
        if self.name:
            result['name'] = self.name
        return result
    
    def get_sentences(self) -> List[LatexSentence]:
        """Return all sentences in this paragraph"""
        return self.sentences
    
    def get_next_sentence_until_citation(self, current_idx: int, sentence_number: int):
        graph_environments = {'tikzpicture', 'figure', 'table', 'tabular', 'longtable'}    
        current_sentence = self.sentences[current_idx]
        if len(current_sentence.citations) >= 2: return []
        next_sentences = []
        i = current_idx + 1
        while i <= current_idx + sentence_number + 1:
            if i >= len(self.sentences): return next_sentences
            sentence = self.sentences[i]
            if isinstance(sentence, LatexSentence) and sentence.citations: return next_sentences
            if isinstance(sentence, LatexEnvironment) and sentence.environment_name in graph_environments:
                sentence_number += 1
            else: 
                next_sentences.append(sentence.text)
            i += 1
        return next_sentences
    
    def __repr__(self):
        repr_string, _ = debug_sentences(self.sentences)
        return repr_string
    

def get_paragraph_skeleton(paragraph: LatexParagraph, mode: str, accumulated_sentence_id: int = 0):
    if mode == "none" or len(paragraph) == 0: return ""
    if mode == "first":
        for s in paragraph.sentences:
            if isinstance(s, LatexSentence): return s.text
        return paragraph.sentences[0].text
    else:
        return debug_sentences(paragraph.sentences, accumulated_sentence_id)


@dataclass
class LatexSubSubSection:
    r"""Represents a subsubsection (\subsubsection)"""
    name: str
    children: List[Union[LatexParagraph, 'LatexSubSubSection']] = field(default_factory=list)
    
    def add_child(self, child: Union[LatexParagraph, 'LatexSubSubSection']):
        self.children.append(child)
    
    def to_dict(self):
        result = {
            'name': self.name,
            'children': []
        }
        for child in self.children:
            if isinstance(child, LatexParagraph):
                result['children'].append({'type': 'paragraph', 'content': child.to_dict()})
            else:
                result['children'].append({'type': 'subsubsection', 'content': child.to_dict()})
        return result
    
    def get_sentences(self) -> List[LatexSentence]:
        """Return all sentences in this subsection"""
        sentences = []
        for p in self.children:
            sentences.extend(p.get_sentences())
        return sentences

    def get_skeleton(self, section_id: str) -> Dict[str, Any]:
        paragraphs = [child.get_skeleton() for child in self.children if isinstance(child, LatexParagraph)]
        sections = []
        sub_idx = 0
        for child in self.children:
            if isinstance(child, LatexSubSubSection):
                sub_idx += 1
                child_id = f"{section_id}.{sub_idx}" if section_id else str(sub_idx)
                sections.append(child.get_skeleton(child_id))
        return {
            'title': self.name,
            'section_id': section_id,
            'paragraphs': paragraphs,
            'sections': sections
        }
    
    def __repr__(self):
        return f"SubSubSection('{self.name}', {len(self.children)} children)"


@dataclass
class LatexSubSection:
    r"""Represents a subsection (\subsection)"""
    name: str
    children: List[Union[LatexParagraph, LatexSubSubSection]] = field(default_factory=list)
    
    def add_child(self, child: Union[LatexParagraph, LatexSubSubSection]):
        self.children.append(child)
    
    def to_dict(self):
        result = {
            'name': self.name,
            'children': []
        }
        for child in self.children:
            if isinstance(child, LatexParagraph):
                result['children'].append({'type': 'paragraph', 'content': child.to_dict()})
            elif isinstance(child, LatexSubSubSection):
                result['children'].append({'type': 'subsubsection', 'content': child.to_dict()})
        return result
    
    def get_sentences(self) -> List[LatexSentence]:
        """Return all sentences in this subsection"""
        sentences = []
        for p in self.children:
            sentences.extend(p.get_sentences())
        return sentences

    def get_skeleton(self, section_id: str) -> Dict[str, Any]:
        paragraphs = []
        sections = []
        sub_idx = 0
        for child in self.children:
            if isinstance(child, LatexParagraph):
                paragraphs.append(child.get_skeleton())
            elif isinstance(child, LatexSubSubSection):
                sub_idx += 1
                child_id = f"{section_id}.{sub_idx}" if section_id else str(sub_idx)
                sections.append(child.get_skeleton(child_id))
        return {
            'title': self.name,
            'section_id': section_id,
            'paragraphs': paragraphs,
            'sections': sections
        }
    
    def __repr__(self):
        return f"SubSection('{self.name}', {len(self.children)} children)"


@dataclass
class LatexSection:
    r"""Represents a section (\section)"""
    name: str
    children: List[Union[LatexParagraph, LatexSubSection]] = field(default_factory=list)
    
    def add_child(self, child: Union[LatexParagraph, LatexSubSection]):
        self.children.append(child)
    
    def to_dict(self):
        result = {
            'name': self.name,
            'children': []
        }
        for child in self.children:
            if isinstance(child, LatexParagraph):
                result['children'].append({'type': 'paragraph', 'content': child.to_dict()})
            elif isinstance(child, LatexSubSection):
                result['children'].append({'type': 'subsection', 'content': child.to_dict()})
        return result
    
    def get_sentences(self) -> List[LatexSentence]:
        """Return all sentences in this subsection"""
        sentences = []
        for p in self.children:
            sentences.extend(p.get_sentences())
        return sentences

    def get_skeleton(self, section_id: Union[int, str]) -> Dict[str, Any]:
        paragraphs = []
        sections = []
        sub_idx = 0
        for child in self.children:
            if isinstance(child, LatexParagraph):
                paragraphs.append(child.get_skeleton())
            elif isinstance(child, LatexSubSection):
                sub_idx += 1
                child_id = f"{section_id}.{sub_idx}" if section_id else str(sub_idx)
                sections.append(child.get_skeleton(child_id))
        return {
            'title': self.name,
            'section_id': section_id,
            'paragraphs': paragraphs,
            'sections': sections
        }
    
    def __repr__(self):
        return f"Section('{self.name}', {len(self.children)} children)"


@dataclass
class LatexPaper:
    """Represents the entire academic paper"""
    title: Optional[str] = None
    author: Optional[str] = None
    abstract: Optional[LatexSubSubSection] = None
    keywords: List[str] = field(default_factory=list)
    sections: List[LatexSection] = field(default_factory=list)
    all_citation_keys: List[str] = field(default_factory=list)
    bibliography: dict = field(default_factory=dict)  # Maps citation keys to bibliography entries
    
    def add_section(self, section: LatexSection):
        self.sections.append(section)
    
    def add_keyword(self, word: str):
        self.keywords.append(word)
    
    def to_dict(self):
        result = {}
        if self.title:
            result['title'] = self.title
        if self.author:
            result['author'] = self.author
        if self.abstract:
            result['abstract'] = self.abstract.to_dict()
        
        result['sections'] = [s.to_dict() for s in self.sections]
        
        if self.all_citation_keys:
            result['citations'] = self.all_citation_keys
        
        return result
    
    def get_sentences(self) -> List[LatexSentence]:
        """Return all sentences in this subsection"""
        sentences = []
        if self.abstract is not None: sentences = self.abstract.get_sentences()
        for p in self.sections:
            sentences.extend(p.get_sentences())
        return sentences
    
    def get_skeleton(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'author': self.author,
            'abstract': self.abstract.get_skeleton("") if self.abstract else "",
            'paragraphs': [],
            'sections': [section.get_skeleton(i + 1) for i, section in enumerate(self.sections)],
            'citations': self.bibliography
        }
    
    def __str__(self):
        return json.dumps(self.get_skeleton(), indent=2, ensure_ascii=False)