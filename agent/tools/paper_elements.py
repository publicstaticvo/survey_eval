from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any


@dataclass
class Sentence:
    """Represents a single sentence with its citations"""
    text: str  # without citations
    father: Paragraph
    citations: List[str] = field(default_factory=list)
    
    def __str__(self):
        cite_str = f" [{', '.join(self.citations)}]" if self.citations else ""
        return f"{self.text} {cite_str}"
    
    def get_skeleton(self) -> Dict[str, Union[str, List[str]]]:
        return {"text": self.text, "citations": self.citations}


@dataclass
class Paragraph:
    """Represents a paragraph with multiple sentences"""
    father: Section
    sentences: List[Sentence] = field(default_factory=list)
    
    def add_sentence(self, sentence: Sentence):
        self.sentences.append(sentence)
    
    def get_skeleton(self) -> List[Dict[str, Union[str, List[str]]]]:
        """Return all sentences in this paragraph"""
        return [s.get_skeleton() for s in self.sentences]


@dataclass
class Section:
    """Represents a section, subsection, subsubsection or abstract"""
    name: str = ""
    father: Union[Section, Paper]
    paragraphs: List[Paragraph] = field(default_factory=list)
    children: List[Section] = field(default_factory=list)
    
    def add_paragraph(self, child: Paragraph):
        self.paragraphs.append(child)
    
    def add_child(self, child: Section):
        self.children.append(child)
        
    def get_skeleton(self) -> Dict[str, Any]:
        return {
            "title": self.name,
            "paragraphs": [paragraph.get_skeleton() for paragraph in self.paragraphs],
            "sections": [section.get_skeleton() for section in self.children],
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
            "abstract": self.abstract.get_skeleton() if self.abstract else "",
            "paragraphs": [paragraph.get_skeleton() for paragraph in self.paragraphs],
            "sections": [section.get_skeleton() for section in self.children],
            "citations": self.references
        }
