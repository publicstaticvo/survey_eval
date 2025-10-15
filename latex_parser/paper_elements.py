from typing import List, Optional, Union, Dict
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
        cite_str = f" [{', '.join(self.citations)}]" if self.citations else ""
        return f"Sentence({self.text[:50]}...{cite_str})"
    

@dataclass
class LatexEnvironment:
    """Represents a Latex environment block (theorem, remark, tikzpicture, etc.)"""
    environment_name: str
    content: str  # Raw Latex content
    
    def to_dict(self):
        return {
            'type': 'latex_environment',
            'environment': self.environment_name,
            'content': self.content
        }
    
    def __repr__(self):
        return f"\\begin{{{self.environment_name}}}\n{self.content}\n\\end{{{self.environment_name}}}\n"


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
    return output, start_id + len(sentences)


@dataclass
class LatexParagraph:
    """Represents a paragraph (either \paragraph command or text block split by \n\n)"""
    sentences: List[Union[LatexSentence, LatexEnvironment]] = field(default_factory=list)
    name: Optional[str] = None  # Only set if defined by \paragraph
    
    def add_sentence(self, sentence: Union[LatexSentence, LatexEnvironment]):
        self.sentences.append(sentence)
    
    def to_dict(self):
        result = {'sentences': [s.to_dict() for s in self.sentences]}
        if self.name:
            result['name'] = self.name
        return result
    
    def get_sentences(self) -> List[LatexSentence]:
        """Return all sentences in this paragraph"""
        return self.sentences
    
    def __repr__(self):
        repr_string, _ = debug_sentences(self.sentences)
        return repr_string


@dataclass
class LatexSubSubSection:
    """Represents a subsubsection (\subsubsection)"""
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
    
    def __repr__(self):
        return f"SubSubSection('{self.name}', {len(self.children)} children)"


@dataclass
class LatexSubSection:
    """Represents a subsection (\subsection)"""
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
    
    def __repr__(self):
        return f"SubSection('{self.name}', {len(self.children)} children)"


@dataclass
class LatexSection:
    """Represents a section (\section)"""
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
    
    def __repr__(self):
        return f"Section('{self.name}', {len(self.children)} children)"


@dataclass
class LatexAbstract:
    """Represents the abstract"""
    paragraphs: List[LatexParagraph] = field(default_factory=list)
    
    def add_paragraph(self, paragraph: LatexParagraph):
        self.paragraphs.append(paragraph)
    
    def to_dict(self):
        return {'paragraphs': [p.to_dict() for p in self.paragraphs]}
    
    def get_sentences(self) -> List[LatexSentence]:
        """Return all sentences in this subsection"""
        sentences = []
        for p in self.paragraphs:
            sentences.extend(p.get_sentences())
        return sentences
    
    def __repr__(self):
        return f"Abstract({len(self.paragraphs)} paragraphs)"


@dataclass
class LatexPaper:
    """Represents the entire academic paper"""
    title: Optional[str] = None
    author: Optional[str] = None
    abstract: Optional[LatexAbstract] = None
    sections: List[LatexSection] = field(default_factory=list)
    all_citation_keys: List[str] = field(default_factory=list)
    bibliography: dict = field(default_factory=dict)  # Maps citation keys to bibliography entries
    
    def add_section(self, section: LatexSection):
        self.sections.append(section)
    
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
    
    def map_citations_to_sentence(self) -> Dict[str, any]:
        pass
    
    def __str__(self):
        sentence_id = 0
        repr_str = f"Title: {self.title}, Author: {self.author}\n"
        if self.abstract is not None:
            abstract_sentences = self.abstract.get_sentences()
            abstract_sentences, sentence_id = debug_sentences(abstract_sentences)
            repr_str += f"Abstract:\n{abstract_sentences}\n"
        
        for i, section in enumerate(self.sections):
            section_str = f"Section {i + 1} - {section.name}\n"
            subsection_count = 0
            # 只有到了Paragraph中才能获取sentences。
            for subsection in section.children:
                if isinstance(subsection, LatexParagraph):
                    chapter_0_sentences = subsection.get_sentences()
                    chapter_0, sentence_id = debug_sentences(chapter_0_sentences, sentence_id)
                    section_str += f"{chapter_0}\n"
                else:
                    subsection_count += 1
                    subsection_str = f"Section {i + 1}.{subsection_count} - {subsection.name}\n"
                    subsubsection_count = 0
                    for subsubsection in subsection.children:
                        if isinstance(subsubsection, LatexParagraph):
                            chapter_00_sentences = subsubsection.get_sentences()
                            chapter_00, sentence_id = debug_sentences(chapter_00_sentences, sentence_id)
                            subsection_str += f"{chapter_00}\n"
                        else:
                            subsubsection_count += 1
                            subsubsection_str = f"Section {i + 1}.{subsection_count}.{subsubsection_count} - {subsubsection.name}\n"
                            for paragraph in subsection.children:
                                sentences = paragraph.get_sentences()
                                gathered_sentence, sentence_id = debug_sentences(sentences, sentence_id)
                                subsubsection_str += f"{gathered_sentence}\n"

                            subsection_str += subsubsection_str

                    section_str += subsection_str

            repr_str += section_str

        return repr_str