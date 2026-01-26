"""
Quick reference guide for using the LaTeX parser.
Copy-paste ready code snippets for common tasks.
"""

from latex_parser import parse_paper, EnvironmentType
from pathlib import Path
import json

# ============================================================================
# QUICK REFERENCE: Common LaTeX Parser Tasks
# ============================================================================

# 1. BASIC PARSING
# ============================================================================
"""Parse a LaTeX paper file"""
paper = parse_paper("path/to/main.tex")


# 2. ACCESSING PAPER METADATA
# ============================================================================
"""Get title and author"""
title = paper.title
author = paper.author
num_sections = len(paper.children)
num_references = len(paper.references)
has_abstract = paper.abstract is not None


# 3. TRAVERSING SECTIONS
# ============================================================================
"""Iterate through all sections"""
for section in paper.children:
    print(f"Section: {section.name} (Level {section.level})")
    for subsection in section.children:
        print(f"  Subsection: {subsection.name}")


"""Get specific section by name"""
def find_section(paper, name):
    def search(section):
        if section.name == name:
            return section
        for child in section.children:
            result = search(child)
            if result:
                return result
        return None
    
    return search(paper)

introduction = find_section(paper, "Introduction")


# 4. EXTRACTING CONTENT
# ============================================================================
"""Get all text from a section"""
def get_section_text(section):
    sentences = []
    for para in section.paragraphs:
        for sentence in para.sentences:
            sentences.append(sentence.text)
    for child in section.children:
        sentences.extend(get_section_text(child))
    return " ".join(sentences)

full_text = get_section_text(paper.children[0])


"""Get all sentences from a section"""
def get_section_sentences(section):
    sentences = []
    for para in section.paragraphs:
        sentences.extend(para.sentences)
    for child in section.children:
        sentences.extend(get_section_sentences(child))
    return sentences

all_sentences = get_section_sentences(paper.children[0])


# 5. CITATION ANALYSIS
# ============================================================================
"""Count citations in each section"""
citation_counts = {}
for section in paper.children:
    citations = []
    for para in section.paragraphs:
        for sentence in para.sentences:
            citations.extend(sentence.citations)
    if citations:
        citation_counts[section.name] = len(citations)


"""Find all unique citations"""
all_citations = set()
for section in paper.children:
    for para in section.paragraphs:
        for sentence in para.sentences:
            all_citations.update(sentence.citations)


"""Find which sections cite a specific paper"""
def find_citations(paper, citation_key):
    locations = []
    for section in paper.children:
        for para in section.paragraphs:
            for sentence in para.sentences:
                if citation_key in sentence.citations:
                    locations.append((section.name, sentence.text[:100]))
    return locations

locations = find_citations(paper, "transformer")


"""Calculate citation density per section"""
def calculate_citation_density(section):
    total_sentences = sum(len(p.sentences) for p in section.paragraphs)
    total_citations = sum(
        len(s.citations) 
        for p in section.paragraphs 
        for s in p.sentences
    )
    return total_citations / total_sentences if total_sentences > 0 else 0


# 6. FILTERING CONTENT
# ============================================================================
"""Get all equations (or figures, tables, etc.)"""
def get_equations(paper):
    equations = []
    for section in paper.children:
        for para in section.paragraphs:
            if para.environment_type == EnvironmentType.EQUATION:
                equations.append((section.name, para.sentences[0].text if para.sentences else ""))
    return equations

equations = get_equations(paper)


"""Get all text-type paragraphs"""
def get_text_paragraphs(paper):
    paragraphs = []
    for section in paper.children:
        for para in section.paragraphs:
            if para.environment_type == EnvironmentType.TEXT:
                text = " ".join(s.text for s in para.sentences)
                paragraphs.append((section.name, text))
    return paragraphs

text_content = get_text_paragraphs(paper)


# 7. SEARCHING CONTENT
# ============================================================================
"""Search for sentences containing a keyword"""
def search_keyword(paper, keyword):
    results = []
    for section in paper.children:
        for para in section.paragraphs:
            for sentence in para.sentences:
                if keyword.lower() in sentence.text.lower():
                    results.append((section.name, sentence.text))
    return results

matches = search_keyword(paper, "deep learning")


"""Find sentences with specific citations"""
def sentences_with_citation(paper, citation_key):
    results = []
    for section in paper.children:
        for para in section.paragraphs:
            for sentence in para.sentences:
                if citation_key in sentence.citations:
                    results.append((section.name, sentence.text))
    return results

cited_sentences = sentences_with_citation(paper, "bert")


# 8. STATISTICS AND ANALYSIS
# ============================================================================
"""Get paper statistics"""
def analyze_paper(paper):
    stats = {
        "title": paper.title,
        "author": paper.author,
        "total_sections": len(paper.children),
        "total_paragraphs": sum(len(s.paragraphs) for s in paper.children),
        "total_sentences": 0,
        "total_citations": 0,
        "unique_citations": set(),
        "avg_sentences_per_paragraph": 0,
    }
    
    total_paras = 0
    for section in paper.children:
        for para in section.paragraphs:
            total_paras += 1
            stats["total_sentences"] += len(para.sentences)
            for sentence in para.sentences:
                stats["total_citations"] += len(sentence.citations)
                stats["unique_citations"].update(sentence.citations)
    
    stats["unique_citations"] = len(stats["unique_citations"])
    stats["avg_sentences_per_paragraph"] = (
        stats["total_sentences"] / total_paras if total_paras > 0 else 0
    )
    
    return stats

analysis = analyze_paper(paper)
print(f"Sentences: {analysis['total_sentences']}")
print(f"Unique citations: {analysis['unique_citations']}")


# 9. EXPORTING DATA
# ============================================================================
"""Export paper skeleton to JSON"""
import json

skeleton = paper.get_skeleton()
with open("paper.json", "w", encoding="utf-8") as f:
    json.dump(skeleton, f, indent=2, default=str, ensure_ascii=False)


"""Export content pairs for training"""
def export_content_pairs(paper, output_file):
    pairs = []
    for section in paper.children:
        for para in section.paragraphs:
            text = " ".join(s.text for s in para.sentences)
            citations = []
            for s in para.sentences:
                citations.extend(s.citations)
            
            pairs.append({
                "section": section.name,
                "text": text,
                "citations": list(set(citations)),
            })
    
    with open(output_file, "w") as f:
        json.dump(pairs, f, indent=2, default=str)

export_content_pairs(paper, "content_pairs.json")


# 10. ADVANCED: BUILDING INDICES
# ============================================================================
"""Build citation-to-section index"""
def build_citation_index(paper):
    index = {}
    for section in paper.children:
        for para in section.paragraphs:
            for sentence in para.sentences:
                for citation in sentence.citations:
                    if citation not in index:
                        index[citation] = []
                    index[citation].append(section.name)
    return index

citation_index = build_citation_index(paper)
# Now look up which sections cite a specific paper:
# sections_citing_transformer = citation_index.get("transformer", [])


"""Build section content cache"""
def build_section_cache(paper):
    cache = {}
    for section in paper.children:
        cache[section.name] = {
            "text": get_section_text(section),
            "num_paragraphs": len(section.paragraphs),
            "num_sentences": sum(len(p.sentences) for p in section.paragraphs),
            "citations": set(
                c for p in section.paragraphs 
                for s in p.sentences 
                for c in s.citations
            ),
            "subsections": [s.name for s in section.children],
        }
    return cache

section_cache = build_section_cache(paper)


# ============================================================================
# EXECUTION EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Parse a paper
    paper_path = Path("P:\\AI4S\\survey_paper\\cs\\arXiv-2503.24377v1\\main.tex")
    paper = parse_paper(paper_path)
    
    # Print basic info
    print(f"Paper: {paper.title}")
    print(f"Sections: {len(paper.children)}")
    print(f"References: {len(paper.references)}")
    
    # Get statistics
    stats = analyze_paper(paper)
    print(f"\nStatistics:")
    print(f"  Sentences: {stats['total_sentences']}")
    print(f"  Unique citations: {stats['unique_citations']}")
    print(f"  Avg sentences per paragraph: {stats['avg_sentences_per_paragraph']:.2f}")
    
    # Find top cited papers
    citation_counts = {}
    for section in paper.children:
        for para in section.paragraphs:
            for sentence in para.sentences:
                for cite in sentence.citations:
                    citation_counts[cite] = citation_counts.get(cite, 0) + 1
    
    top_cited = sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 cited papers:")
    for cite, count in top_cited:
        print(f"  {cite}: {count} citations")
