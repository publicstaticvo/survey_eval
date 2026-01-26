"""
Example usage of the LaTeX paper parser.
Demonstrates how to use the parser to extract paper structure and content.
"""

from latex_parser import parse_paper, Paper, Section, Paragraph, Sentence, EnvironmentType
from pathlib import Path
import json


def example_basic_usage():
    """Example 1: Basic parsing and structure access"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Parse a paper
    paper_path = Path("P:\\AI4S\\survey_paper\\cs\\arXiv-2503.24377v1\\main.tex")
    paper = parse_paper(paper_path)
    
    # Access basic information
    print(f"\nTitle: {paper.title}")
    print(f"Author: {paper.author}")
    print(f"Number of top-level sections: {len(paper.children)}")
    print(f"Number of references: {len(paper.references)}")
    
    # Access abstract
    if paper.abstract:
        print(f"\nAbstract has {len(paper.abstract.paragraphs)} paragraphs")
        if paper.abstract.paragraphs:
            para = paper.abstract.paragraphs[0]
            print(f"First paragraph has {len(para.sentences)} sentences")


def example_section_traversal():
    """Example 2: Traverse sections and subsections"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Section Traversal")
    print("="*70)
    
    paper_path = Path("P:\\AI4S\\survey_paper\\cs\\arXiv-2503.24377v1\\main.tex")
    paper = parse_paper(paper_path)
    
    def print_section_tree(section, indent=0):
        """Recursively print section hierarchy"""
        prefix = "  " * indent
        print(f"{prefix}- {section.name}")
        print(f"{prefix}  Paragraphs: {len(section.paragraphs)}, Subsections: {len(section.children)}")
        
        # Print subsections
        for child in section.children[:2]:  # Limit to first 2 for brevity
            print_section_tree(child, indent + 1)
        
        if len(section.children) > 2:
            print(f"{prefix}  ... and {len(section.children) - 2} more subsections")
    
    print("\nSection Structure (first 3 sections):")
    for section in paper.children[:3]:
        print_section_tree(section)


def example_content_extraction():
    """Example 3: Extract and analyze content"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Content Extraction")
    print("="*70)
    
    paper_path = Path("P:\\AI4S\\survey_paper\\cs\\arXiv-2503.24377v1\\main.tex")
    paper = parse_paper(paper_path)
    
    # Count total sentences and citations
    total_sentences = 0
    total_citations = 0
    env_types = {}
    
    def analyze_section(section):
        nonlocal total_sentences, total_citations, env_types
        
        for para in section.paragraphs:
            for sentence in para.sentences:
                total_sentences += 1
                total_citations += len(sentence.citations)
                
                # Count environment types
                env_type = sentence.environment_type.value
                env_types[env_type] = env_types.get(env_type, 0) + 1
        
        for child in section.children:
            analyze_section(child)
    
    analyze_section(paper)
    
    print(f"\nContent Statistics:")
    print(f"  Total sentences: {total_sentences}")
    print(f"  Total citations: {total_citations}")
    print(f"  Environment types: {env_types}")


def example_citation_analysis():
    """Example 4: Analyze citation patterns"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Citation Analysis")
    print("="*70)
    
    paper_path = Path("P:\\AI4S\\survey_paper\\cs\\arXiv-2503.24377v1\\main.tex")
    paper = parse_paper(paper_path)
    
    # Collect all citations
    citation_counts = {}
    
    def collect_citations(section):
        for para in section.paragraphs:
            for sentence in para.sentences:
                for cite in sentence.citations:
                    citation_counts[cite] = citation_counts.get(cite, 0) + 1
        
        for child in section.children:
            collect_citations(child)
    
    collect_citations(paper)
    
    # Get top cited papers
    top_citations = sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\nTop 10 Most Cited References:")
    for i, (cite, count) in enumerate(top_citations, 1):
        print(f"  {i}. {cite}: {count} citations")
    
    print(f"\nTotal unique references in citations: {len(citation_counts)}")


def example_serialize():
    """Example 5: Serialize to JSON"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Serialization to JSON")
    print("="*70)
    
    paper_path = Path("P:\\AI4S\\survey_paper\\cs\\arXiv-2503.24377v1\\main.tex")
    paper = parse_paper(paper_path)
    
    # Get the skeleton (JSON-serializable format)
    skeleton = paper.get_skeleton()
    
    print(f"\nPaper skeleton keys: {list(skeleton.keys())}")
    print(f"Title: {skeleton['title']}")
    print(f"Author: {skeleton['author'][:50]}..." if skeleton['author'] else "No author")
    print(f"Number of sections: {len(skeleton['sections'])}")
    print(f"Number of references: {len(skeleton['citations'])}")
    
    # Save to file
    output_path = "paper_skeleton.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        # Convert to a simple dict to avoid EnvironmentType enum issues
        simple_skeleton = json.dumps(skeleton, default=str, ensure_ascii=False, indent=2)
        f.write(simple_skeleton)
    
    print(f"\nSerialized structure saved to: {output_path}")


def example_search():
    """Example 6: Search for specific content"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Content Search")
    print("="*70)
    
    paper_path = Path("P:\\AI4S\\survey_paper\\cs\\arXiv-2503.24377v1\\main.tex")
    paper = parse_paper(paper_path)
    
    # Search for sentences containing specific keywords
    keywords = ["efficient", "reasoning"]
    matches = []
    
    def search_section(section):
        for para in section.paragraphs:
            for sentence in para.sentences:
                if any(kw.lower() in sentence.text.lower() for kw in keywords):
                    matches.append((section.name, sentence.text[:100]))
        
        for child in section.children:
            search_section(child)
    
    search_section(paper)
    
    print(f"\nSearching for keywords: {keywords}")
    print(f"Found {len(matches)} matching sentences")
    print(f"\nFirst 5 matches:")
    for section_name, sentence in matches[:5]:
        print(f"  [{section_name}] {sentence}...")


def example_filter_by_environment():
    """Example 7: Filter content by environment type"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Filter by Environment Type")
    print("="*70)
    
    paper_path = Path("P:\\AI4S\\survey_paper\\cs\\arXiv-2209.00796v15\\main.tex")
    paper = parse_paper(paper_path)
    
    # Collect sentences by environment type
    env_sentences = {}
    
    def collect_by_env(section):
        for para in section.paragraphs:
            for sentence in para.sentences:
                env = sentence.environment_type.value
                if env not in env_sentences:
                    env_sentences[env] = []
                env_sentences[env].append(sentence.text[:60])
        
        for child in section.children:
            collect_by_env(child)
    
    collect_by_env(paper)
    
    print(f"\nContent by environment type:")
    for env_type, sentences in sorted(env_sentences.items()):
        print(f"\n  {env_type}: {len(sentences)} items")
        if sentences:
            print(f"    Sample: {sentences[0]}...")


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_section_traversal()
    example_content_extraction()
    example_citation_analysis()
    example_serialize()
    example_search()
    example_filter_by_environment()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")
