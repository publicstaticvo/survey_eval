from __future__ import annotations
from typing import List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class Sentence:
    """Represents a single sentence with its citations"""
    text: str  # without citations
    father: Paragraph
    citations: List[str] = field(default_factory=list)
    
    def __str__(self):
        cite_str = f" [{', '.join(self.citations)}]" if self.citations else ""
        return f"{self.text} {cite_str}"


@dataclass
class Paragraph:
    """Represents a paragraph with multiple sentences"""
    sentences: List[Sentence] = field(default_factory=list)
    father: Section
    
    def add_sentence(self, sentence: Sentence):
        self.sentences.append(sentence)
    
    def get_sentences(self) -> List[str]:
        """Return all sentences in this paragraph"""
        return [str(s) for s in self.sentences]
    
    def get_first_sentence(self) -> str:
        return str(self.sentences[0])


@dataclass
class Section:
    """Represents a section, subsection, subsubsection or abstract"""
    name: str
    father: Union[Section, Paper]
    paragraphs: List[Paragraph] = field(default_factory=list)
    children: List[Section] = field(default_factory=list)
    
    def add_paragraph(self, child: Paragraph):
        self.paragraphs.append(child)
    
    def add_child(self, child: Section):
        self.children.append(child)
        
    def get_skeleton(self, section_id: str = "0.", mode: str = "first"):
        repr_str = f"\n{section_id} {self.name}\n" if section_id else f"{self.name}\n\n"
        if mode != "none":
            for i, p in enumerate(self.paragraphs):
                repr_str += f"Paragraph {section_id}{i}\n{p.get_first_sentence() if mode == "first" else ' '.join(p.get_sentences())} ...\n"
            for i, s in enumerate(self.children):
                repr_str += s.get_skeleton(f"{section_id}{i}.", mode)
            repr_str += "\n"
        return repr_str


@dataclass
class Paper(Section):
    """Represents the entire academic paper"""
    title: Optional[str] = None
    author: Optional[str] = None
    abstract: Optional[Section] = None  # Abstract should not have children
    references: dict = field(default_factory=dict)  # Maps citation keys to bibliography entries
    
    def get_skeleton(self, mode: str = "first") -> str:
        repr_str = f"Title: {self.title}\nAuthor: {self.author}\n"
        if self.abstract is not None:
            repr_str += self.abstract.get_skeleton("", "all")        
        for i, section in enumerate(self.children):
            repr_str += section.get_skeleton(f"{i}.", mode)
        return repr_str
