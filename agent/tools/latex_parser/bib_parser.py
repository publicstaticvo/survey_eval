"""
Citation Parser for .bib and .bbl files

This module provides functions to parse bibliography files in both .bib and .bbl formats.
It uses bibtexparser for .bib files and pylatexenc for .bbl files.

Dependencies:
    pip install bibtexparser pylatexenc
"""

import os
import re
from typing import List, Dict, Any
import bibtexparser
from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexwalker import (
    LatexWalker, LatexEnvironmentNode, LatexMacroNode, LatexCharsNode, LatexGroupNode
)
from utils import detect_encoding


def parse_bib_file(filepath: str) -> Dict[str, Any]:
    """
    Parse a .bib file and extract all citations.
    
    Args:
        filepath: Path to the .bib file
        
    Returns:
        List of dictionaries containing citation information
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            bib_database = bibtexparser.load(f)
    except:
        _, encoding = detect_encoding(filepath)
        with open(filepath, 'r', encoding=encoding) as f:
            bib_database = bibtexparser.load(f)
    
    citations = {}
    for entry in bib_database.entries:
        citation_key = entry.get('ID', '')
        if citation_key and citation_key not in citations:
            citations[citation_key] = entry
    
    return citations


def _parse_standard_bibitem(content: str) -> Dict[str, Any]:
    """Parse standard \\bibitem format."""
    # Remove \href commands that would cause bugs
    content = re.sub(r"\\href\s*\{[^\}]*\}\s*\{([^\}]*)\}", r"\1", content)
    content = re.sub(r"\\href\s*\{([^\}]*)\}", "", content)
    
    nodes, _, _ = LatexWalker(content).get_latex_nodes()
    converter = LatexNodes2Text(math_mode="verbatim")

    def get_bib_node(nodes):
        for node in nodes:
            if isinstance(node, LatexEnvironmentNode):
                if node.environmentname == 'thebibliography': 
                    return node
                target_node = get_bib_node(node.nodelist)
                if target_node is not None: 
                    return target_node
                
    def get_ref_content(nodes):
        content = converter.nodelist_to_text(nodes).strip()
        if any(isinstance(node, LatexMacroNode) and node.macroname == "newblock" for node in nodes):
            content_split = content.split("\n")
            if len(content_split) > 2:
                return {
                    "author": content_split[0], 
                    "title": content_split[1], 
                    "info": "\n".join(content_split[2:])
                }
            return {"info": content}
        else:
            content = " ".join([x.strip() for x in content.split("\n")])
            return {"info": content}
            
    bib_node = get_bib_node(nodes)
    all_bib = {}
    if bib_node is not None:
        current_key, current_value, bibitem_flag = None, [], False
        for node in bib_node.nodelist:
            if isinstance(node, LatexMacroNode) and node.macroname == "bibitem":
                if current_key:
                    cite_content = get_ref_content(current_value)
                    all_bib[current_key] = cite_content
                current_key = None
                if node.nodeargd and node.nodeargd.argnlist:
                    for arg in reversed(node.nodeargd.argnlist):
                        if arg is not None:
                            current_key = converter.nodelist_to_text([arg]).strip()
                            break
                current_value = []
                bibitem_flag = True
            else:
                if isinstance(node, LatexGroupNode) and bibitem_flag:
                    current_key = converter.nodelist_to_text(node.nodelist).strip()
                elif current_key:
                    current_value.append(node)
                bibitem_flag = False
        if current_key:
            cite_content = get_ref_content(current_value)
            all_bib[current_key] = cite_content

    return all_bib


def _parse_compiled_entry(content: str) -> Dict[str, Any]:
    """Parse compiled \\entry format (from biblatex)."""
    nodes, _, _ = LatexWalker(content).get_latex_nodes()
    converter = LatexNodes2Text(math_mode="verbatim")

    def get_text_for_node(i: int) -> str:
        return converter.nodelist_to_text([nodes[i]]).strip()

    i = 0
    citations = {}
    current_key = None
    while i < len(nodes):
        node = nodes[i]
        if isinstance(node, LatexMacroNode) and node.macroname == "entry":
            current_key = get_text_for_node(i + 1)
            citations[current_key] = {"paper_type": get_text_for_node(i + 2)}
            i += 3
        elif current_key is not None and isinstance(node, LatexMacroNode):  
            if node.macroname == "name" and get_text_for_node(i + 1) == "author":
                author_text = get_text_for_node(i + 4)
                authors = []
                family_pattern = re.compile(r"family=(.+?),", re.DOTALL)
                given_pattern = re.compile(r"given=(.+?),", re.DOTALL)
                for f, g in zip(re.findall(family_pattern, author_text), re.findall(given_pattern, author_text)):
                    authors.append(f"{g} {f}")
                citations[current_key]['authors'] = authors
                i += 4
            elif node.macroname == "field":
                field_key = get_text_for_node(i + 1)
                if field_key in ['title', 'year']:
                    citations[current_key][field_key] = get_text_for_node(i + 2)
                i += 2
            elif node.macroname == "endentry":
                current_key = None
            elif node.macroname == "verb":
                if node.nodeargd and node.nodeargd.verbatim_text.endswith("\\entry"):
                    current_key = get_text_for_node(i + 1)
                    citations[current_key] = {"paper_type": get_text_for_node(i + 2)}
                    i += 3
        i += 1
        
    return citations


def parse_bbl_file(filepath: str) -> Dict[str, Any]:
    """
    Parse a .bbl file and extract all citations.
    Handles multiple citation formats including standard bibitem and compiled formats.
    
    Args:
        filepath: Path to the .bbl file
        
    Returns:
        List of dictionaries containing citation information
    """
    
    if os.path.isfile(filepath):        
        try:
            with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
        except:
            content, _ = detect_encoding(filepath)
    else: content = filepath

    if "\\bibitem" in content:
        return _parse_standard_bibitem(content)
    elif "\\entry" in content:
        return _parse_compiled_entry(content)
    return {}
