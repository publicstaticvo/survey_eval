from __future__ import annotations
from typing import List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class Sentence:
    """Represents a single sentence with its citations"""
    text: str  # without citations
    father: Paragraph
    citations: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {'text': self.text, 'citations': self.citations}
    
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
    
    def to_dict(self):
        result = {'sentences': [s.to_dict() for s in self.sentences]}
        if self.name:
            result['name'] = self.name
        return result
    
    def get_sentences(self) -> List[Sentence]:
        """Return all sentences in this paragraph"""
        return self.sentences


@dataclass
class Section:
    """Represents a section (\section)"""
    name: str
    father: Section
    paragraphs: List[Paragraph] = field(default_factory=list)
    children: List[Section] = field(default_factory=list)
    
    def add_paragraph(self, child: Paragraph):
        self.paragraphs.append(child)
    
    def add_child(self, child: Section):
        self.children.append(child)
    
    def to_dict(self):
        return {
            'name': self.name,
            'paragraphs': [{'type': 'paragraph', 'content': p.to_dict()} for p in self.paragraphs],
            'children': [{'type': 'section', 'content': p.to_dict()} for p in self.children]
        }
    
    def get_sentences(self) -> List[Sentence]:
        """Return all sentences in this section"""
        sentences = []
        for p in self.paragraphs:
            sentences.extend(p.get_sentences())
        for p in self.children:
            sentences.extend(p.get_sentences())
        return sentences


@dataclass
class Paper(Section):
    """Represents the entire academic paper"""
    title: Optional[str] = None
    author: Optional[str] = None
    abstract: Optional[Section] = None  # Abstract should not have children
    references: dict = field(default_factory=dict)  # Maps citation keys to bibliography entries
    
    def to_dict(self):
        result = {}
        if self.title:
            result['title'] = self.title
        if self.author:
            result['author'] = self.author
        if self.abstract:
            result['abstract'] = self.abstract.to_dict()
        
        result['sections'] = [s.to_dict() for s in self.children]
        
        return result
    
    def get_sentences(self) -> List[Sentence]:
        """Return all sentences in this subsection"""
        sentences = []
        if self.abstract is not None: sentences = self.abstract.get_sentences()
        for p in self.children:
            sentences.extend(p.get_sentences())
        return sentences