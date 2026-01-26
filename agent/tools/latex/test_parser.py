"""Test script for LaTeX paper parser"""

import json
from pathlib import Path
from latex_parser import parse_paper

# Test papers
test_papers = [
    Path("P:\\AI4S\\survey_paper\\cs\\arXiv-2209.00796v15\\main.tex"),
    Path("P:\\AI4S\\survey_paper\\cs\\arXiv-2503.24377v1\\main.tex"),
]

def test_parse():
    """Test parsing of sample papers"""
    for paper_path in test_papers:
        if not paper_path.exists():
            print(f"[FAIL] File not found: {paper_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing: {paper_path.name}")
        print('='*60)
        
        try:
            paper = parse_paper(paper_path)
            
            # Print basic info
            print(f"\n[TITLE] {paper.title}")
            print(f"[AUTHOR] {paper.author}")
            
            # Print abstract info
            if paper.abstract:
                print(f"\n[ABSTRACT] {len(paper.abstract.paragraphs)} paragraphs")
                if paper.abstract.paragraphs:
                    first_para = paper.abstract.paragraphs[0]
                    if first_para.sentences:
                        preview = first_para.sentences[0].text[:100]
                        print(f"   Preview: {preview}...")
            
            # Print section info
            print(f"\n[SECTIONS] {len(paper.children)} total")
            for i, section in enumerate(paper.children[:5], 1):
                print(f"   {i}. {section.name}")
                print(f"      - Paragraphs: {len(section.paragraphs)}")
                print(f"      - Subsections: {len(section.children)}")
                if section.children:
                    for j, subsec in enumerate(section.children[:2], 1):
                        print(f"        {j}. {subsec.name}")
            
            if len(paper.children) > 5:
                print(f"   ... and {len(paper.children) - 5} more sections")
            
            # Print reference info
            print(f"\n[REFERENCES] {len(paper.references)} entries")
            if paper.references:
                sample_keys = list(paper.references.keys())[:3]
                for key in sample_keys:
                    print(f"   - {key}")
            
            # Check paragraph content
            total_paragraphs = len(paper.paragraphs)
            for section in paper.children:
                total_paragraphs += len(section.paragraphs)
            
            print(f"\n[PARAGRAPHS] {total_paragraphs} total")
            
            # Check for citations in content
            total_citations = 0
            for section in paper.children:
                for para in section.paragraphs:
                    for sentence in para.sentences:
                        total_citations += len(sentence.citations)
            
            print(f"[CITATIONS_IN_CONTENT] {total_citations} found")
            
            # Show sample content
            if paper.children and paper.children[0].paragraphs:
                first_para = paper.children[0].paragraphs[0]
                if first_para.sentences:
                    print(f"\n[SAMPLE_CONTENT]")
                    print(f"   {first_para.sentences[0].text[:150]}...")
                    if first_para.sentences[0].citations:
                        print(f"   Citations: {first_para.sentences[0].citations}")
            
            print(f"\n[SUCCESS] Successfully parsed!")
            
        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_parse()
