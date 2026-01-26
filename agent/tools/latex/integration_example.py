"""
Integration example: Using the LaTeX parser with paper_elements.py data structures.
This demonstrates how to use the parsed output in various NLP/ML pipelines.
"""

from pathlib import Path
from latex_parser import parse_paper, EnvironmentType
import json


def analyze_paper_structure(paper_path):
    """
    Analyze and display paper structure comprehensively.
    
    Args:
        paper_path: Path to main.tex file
    
    Returns:
        Dictionary with analysis results
    """
    paper = parse_paper(paper_path)
    
    analysis = {
        "metadata": {
            "title": paper.title,
            "author": paper.author,
            "filepath": str(paper_path),
        },
        "structure": {
            "total_sections": len(paper.children),
            "max_section_level": max([s.level for s in paper.children] or [0]),
            "total_paragraphs": sum(len(s.paragraphs) for s in paper.children) + len(paper.paragraphs),
            "has_abstract": paper.abstract is not None,
        },
        "content": {
            "total_sentences": 0,
            "total_citations": 0,
            "environment_type_counts": {},
        },
        "references": {
            "total_entries": len(paper.references),
            "sample_keys": list(paper.references.keys())[:5] if paper.references else [],
        }
    }
    
    # Count content statistics
    def count_content(section):
        for para in section.paragraphs:
            for sentence in para.sentences:
                analysis["content"]["total_sentences"] += 1
                analysis["content"]["total_citations"] += len(sentence.citations)
                
                env_type = sentence.environment_type.value
                analysis["content"]["environment_type_counts"][env_type] = \
                    analysis["content"]["environment_type_counts"].get(env_type, 0) + 1
        
        for child in section.children:
            count_content(child)
    
    count_content(paper)
    
    return analysis, paper


def extract_section_summaries(paper_path, max_sentences=3):
    """
    Extract first N sentences from each section as a summary.
    
    Args:
        paper_path: Path to main.tex file
        max_sentences: Maximum sentences to extract per section
    
    Returns:
        Dictionary mapping section names to summaries
    """
    paper = parse_paper(paper_path)
    summaries = {}
    
    def extract_summaries(section, summaries):
        section_key = f"{section.name} (Level {section.level})"
        sentences = []
        
        for para in section.paragraphs:
            for sentence in para.sentences:
                if len(sentences) < max_sentences:
                    sentences.append(sentence.text)
        
        if sentences:
            summaries[section_key] = {
                "sentences": sentences,
                "full_text": " ".join(sentences),
                "num_paragraphs": len(section.paragraphs),
            }
        
        for child in section.children:
            extract_summaries(child, summaries)
    
    extract_summaries(paper, summaries)
    return summaries


def build_citation_graph(paper_path):
    """
    Build a citation graph showing which sections cite which papers.
    
    Args:
        paper_path: Path to main.tex file
    
    Returns:
        Dictionary mapping sections to their cited papers
    """
    paper = parse_paper(paper_path)
    citation_graph = {}
    
    def build_graph(section):
        cited_papers = set()
        
        for para in section.paragraphs:
            for sentence in para.sentences:
                cited_papers.update(sentence.citations)
        
        if cited_papers:
            citation_graph[section.name] = {
                "cited_papers": list(cited_papers),
                "num_citations": len(cited_papers),
                "unique_citations": len(set(cited_papers)),
            }
        
        for child in section.children:
            build_graph(child)
    
    build_graph(paper)
    return citation_graph


def export_paper_skeleton(paper_path, output_path):
    """
    Export paper skeleton in JSON format for further processing.
    
    Args:
        paper_path: Path to main.tex file
        output_path: Path to save JSON file
    """
    paper = parse_paper(paper_path)
    skeleton = paper.get_skeleton()
    
    # Convert to JSON-serializable format
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(skeleton, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"Paper skeleton exported to: {output_path}")
    return skeleton


def extract_section_content_pairs(paper_path):
    """
    Extract (section_title, paragraph_text) pairs for training/evaluation.
    Useful for topic modeling, section classification, etc.
    
    Args:
        paper_path: Path to main.tex file
    
    Returns:
        List of (section_name, paragraph_text, citations) tuples
    """
    paper = parse_paper(paper_path)
    content_pairs = []
    
    def extract_pairs(section):
        for para in section.paragraphs:
            # Combine all sentences in paragraph
            full_text = " ".join(s.text for s in para.sentences)
            citations = []
            for s in para.sentences:
                citations.extend(s.citations)
            
            if full_text.strip():
                content_pairs.append({
                    "section": section.name,
                    "section_level": section.level,
                    "text": full_text,
                    "citations": list(set(citations)),
                    "num_sentences": len(para.sentences),
                    "environment_type": para.environment_type.value,
                })
        
        for child in section.children:
            extract_pairs(child)
    
    extract_pairs(paper)
    return content_pairs


def calculate_citation_density(paper_path):
    """
    Calculate citation density (citations per sentence) for each section.
    
    Args:
        paper_path: Path to main.tex file
    
    Returns:
        Dictionary with citation density metrics per section
    """
    paper = parse_paper(paper_path)
    density_metrics = {}
    
    def calculate_density(section):
        total_sentences = 0
        total_citations = 0
        
        for para in section.paragraphs:
            for sentence in para.sentences:
                total_sentences += 1
                total_citations += len(sentence.citations)
        
        if total_sentences > 0:
            citation_density = total_citations / total_sentences
            density_metrics[section.name] = {
                "total_sentences": total_sentences,
                "total_citations": total_citations,
                "citation_density": citation_density,
                "avg_citations_per_sentence": citation_density,
            }
        
        for child in section.children:
            calculate_density(child)
    
    calculate_density(paper)
    return density_metrics


def main():
    """Demonstrate all analysis functions"""
    
    paper_path = Path("P:\\AI4S\\survey_paper\\cs\\arXiv-2503.24377v1\\main.tex")
    
    print("\n" + "="*80)
    print("INTEGRATED LaTeX PARSER ANALYSIS")
    print("="*80)
    
    # 1. Basic Analysis
    print("\n[1] Paper Structure Analysis")
    print("-" * 80)
    analysis, paper = analyze_paper_structure(paper_path)
    
    print(f"Title: {analysis['metadata']['title']}")
    print(f"Author: {analysis['metadata']['author'][:50]}...")
    print(f"\nStructure:")
    print(f"  - Sections: {analysis['structure']['total_sections']}")
    print(f"  - Paragraphs: {analysis['structure']['total_paragraphs']}")
    print(f"  - Abstract: {analysis['structure']['has_abstract']}")
    print(f"\nContent Statistics:")
    print(f"  - Total Sentences: {analysis['content']['total_sentences']}")
    print(f"  - Total Citations: {analysis['content']['total_citations']}")
    print(f"  - Content Types: {analysis['content']['environment_type_counts']}")
    print(f"\nReferences:")
    print(f"  - Total Entries: {analysis['references']['total_entries']}")
    
    # 2. Section Summaries
    print("\n[2] Section Summaries (first 2 sentences)")
    print("-" * 80)
    summaries = extract_section_summaries(paper_path, max_sentences=2)
    
    for i, (section_name, summary) in enumerate(list(summaries.items())[:5], 1):
        print(f"\n{i}. {section_name}")
        print(f"   Paragraphs: {summary['num_paragraphs']}")
        print(f"   Summary: {summary['full_text'][:100]}...")
    
    if len(summaries) > 5:
        print(f"\n... and {len(summaries) - 5} more sections")
    
    # 3. Citation Analysis
    print("\n[3] Citation Graph (top 5 cited sections)")
    print("-" * 80)
    citation_graph = build_citation_graph(paper_path)
    
    # Sort by number of unique citations
    sorted_sections = sorted(
        citation_graph.items(),
        key=lambda x: x[1]['unique_citations'],
        reverse=True
    )[:5]
    
    for i, (section, data) in enumerate(sorted_sections, 1):
        print(f"\n{i}. {section}")
        print(f"   Unique papers cited: {data['unique_citations']}")
        print(f"   Total citations: {data['num_citations']}")
        print(f"   Sample citations: {', '.join(data['cited_papers'][:3])}")
    
    # 4. Citation Density
    print("\n[4] Citation Density Analysis (top 5)")
    print("-" * 80)
    density = calculate_citation_density(paper_path)
    
    sorted_by_density = sorted(
        density.items(),
        key=lambda x: x[1]['citation_density'],
        reverse=True
    )[:5]
    
    for i, (section, metrics) in enumerate(sorted_by_density, 1):
        print(f"{i}. {section}")
        print(f"   Sentences: {metrics['total_sentences']}, " +
              f"Citations: {metrics['total_citations']}, " +
              f"Density: {metrics['citation_density']:.2f}")
    
    # 5. Content Pairs
    print("\n[5] Sample Content Pairs for ML Training")
    print("-" * 80)
    content_pairs = extract_section_content_pairs(paper_path)
    
    print(f"Total content pairs extracted: {len(content_pairs)}")
    print(f"\nSample pairs:")
    for i, pair in enumerate(content_pairs[:3], 1):
        print(f"\n{i}. Section: {pair['section']}")
        print(f"   Type: {pair['environment_type']}")
        print(f"   Sentences: {pair['num_sentences']}")
        print(f"   Text: {pair['text'][:80]}...")
        print(f"   Citations: {pair['citations'][:3]}")
    
    # 6. Export
    print("\n[6] Export to JSON")
    print("-" * 80)
    export_path = "paper_analysis.json"
    skeleton = export_paper_skeleton(paper_path, export_path)
    print(f"Exported skeleton with {len(skeleton['sections'])} sections")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
