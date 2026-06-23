"""
Citation Parser for .bib and .bbl files

This module provides functions to parse bibliography files in both .bib and .bbl formats.
It uses bibtexparser for .bib files and pylatexenc for .bbl files.

Dependencies:
    pip install bibtexparser pylatexenc
"""

import os
import re
import chardet
import io
import contextlib
from typing import Dict, Any
import bibtexparser
from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexwalker import (
    LatexWalker, LatexEnvironmentNode, LatexMacroNode, LatexGroupNode
)


def detect_encoding(filepath):
    with open(filepath, 'rb') as f:
        raw_data = f.read()
    
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']
    
    if encoding == 'utf-8':
        print(f"Detect codec: UTF-8. Confidence: {confidence:.2f}. Pass.")
    else:
        print(f"Detect codec: {encoding}. Confidence: {confidence:.2f}.")
    
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            content = f.read()
        return content, encoding
    except:
        print(f"Not codec {encoding}, Try others")
        return "", None


def parse_bib_file(filepath: str) -> Dict[str, Any]:
    """
    Parse a .bib file and extract all citations.
    
    Args:
        filepath: Path to the .bib file
        
    Returns:
        List of dictionaries containing citation information
    """
    def _safe_load(handle):
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return bibtexparser.load(handle)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            bib_database = _safe_load(f)
    except:
        _, encoding = detect_encoding(filepath)
        with open(filepath, 'r', encoding=encoding) as f:
            bib_database = _safe_load(f)
    
    citations = {}
    for entry in bib_database.entries:
        citation_key = entry.get('ID', '')
        if citation_key and citation_key not in citations:
            citations[citation_key] = entry

    add_ref_strings(citations)
    return citations


def add_ref_strings(citations: Dict[str, Any]) -> Dict[str, Any]:
    for entry in citations.values():
        if isinstance(entry, dict):
            entry["ref_string"] = format_ref_string(entry)
    return citations


def format_ref_string(entry: Dict[str, Any]) -> str:
    authors = extract_entry_authors(entry)
    if not authors:
        return "?"
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return f"{authors[0]} et al."


def extract_entry_authors(entry: Dict[str, Any]) -> list[str]:
    raw_authors = entry.get("author") or entry.get("authors")
    split_as_bibtex = bool(raw_authors)
    if not raw_authors:
        raw_authors = extract_info_author_prefix(entry.get("info", ""))
        split_as_bibtex = False

    if isinstance(raw_authors, list):
        return [last_name(author) for author in raw_authors if last_name(author)]
    if isinstance(raw_authors, str):
        authors = split_author_names(raw_authors, split_as_bibtex=split_as_bibtex)
        return [last_name(author) for author in authors if last_name(author)]
    return []


def extract_info_author_prefix(info: str) -> str:
    if not info:
        return ""
    match = re.search(r"\b(?:19|20)\d{2}\b", info)
    if match:
        return info[:match.start()].strip(" .")
    return info.split(".", 1)[0].strip()


def split_author_names(raw_authors: str, split_as_bibtex: bool = False) -> list[str]:
    raw_authors = raw_authors.replace("\xa0", " ")
    raw_authors = re.sub(r"\s+", " ", raw_authors).strip()
    if not raw_authors:
        return []
    if split_as_bibtex and " and " in raw_authors:
        return [author.strip(" ,") for author in re.split(r"\s+and\s+", raw_authors) if author.strip(" ,")]
    if "," in raw_authors:
        return [author.strip(" ,") for author in re.split(r",\s*(?:and\s+)?", raw_authors) if author.strip(" ,")]
    if " and " in raw_authors:
        return [author.strip(" ,") for author in re.split(r"\s+and\s+", raw_authors) if author.strip(" ,")]
    return [raw_authors]


def last_name(author: str) -> str:
    author = LatexNodes2Text(math_mode="verbatim").latex_to_text(str(author))
    author = re.sub(r"\{|\}", "", author)
    author = re.sub(r"\s+", " ", author).strip()
    if not author or author.lower() == "others":
        return ""
    if "," in author:
        name = author.split(",", 1)[0].strip()
    else:
        parts = author.split()
        name = parts[-1] if parts else ""
    return re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ'\\-]", "", name)


def _parse_standard_bibitem(content: str) -> Dict[str, Any]:
    """Parse standard \\bibitem format."""
    # Remove \href commands that would cause bugs
    content = re.sub(r"\\href\s*\{[^\}]*\}\s*\{([^\}]*)\}", r"\1", content)
    content = re.sub(r"\\href\s*\{([^\}]*)\}", "", content)
    
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        nodes, _, _ = LatexWalker(content).get_latex_nodes()
    converter = LatexNodes2Text(math_mode="verbatim")

    def get_bib_node(nodes):
        for node in nodes or []:
            if isinstance(node, LatexEnvironmentNode):
                if node.environmentname == 'thebibliography': 
                    return node
                target_node = get_bib_node(node.nodelist)
                if target_node is not None: 
                    return target_node
                
    def get_ref_content(nodes):
        nodes = nodes or []
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

    def parse_bibitem_regex(raw_content: str) -> Dict[str, Any]:
        raw_content = re.sub(r"\\begin\s*\{thebibliography\}\s*\{[^}]*\}", "", raw_content)
        raw_content = re.sub(r"\\end\s*\{thebibliography\}", "", raw_content)
        pattern = re.compile(
            r"\\bibitem(?:\[[^\]]*\])?\{(?P<key>[^}]+)\}(?P<body>.*?)(?=\\bibitem(?:\[[^\]]*\])?\{|$)",
            flags=re.DOTALL,
        )
        parsed = {}
        for match in pattern.finditer(raw_content):
            key = match.group("key").strip()
            body = match.group("body").strip()
            if not key:
                continue
            body = re.sub(r"\\newblock\b", "\n", body)
            body = re.sub(r"\\emph\s*\{([^}]*)\}", r"\1", body)
            text = converter.latex_to_text(body)
            text = " ".join(part.strip() for part in text.splitlines() if part.strip())
            if text:
                parsed[key] = {"info": text}
        return parsed
            
    bib_node = get_bib_node(nodes)
    all_bib = {}
    if bib_node is not None and bib_node.nodelist is not None:
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

    if not all_bib:
        all_bib = parse_bibitem_regex(content)

    add_ref_strings(all_bib)
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
        
    add_ref_strings(citations)
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
