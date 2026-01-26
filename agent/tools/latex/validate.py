#!/usr/bin/env python3
"""
Validation script to verify LaTeX parser installation and functionality.
Run this to ensure everything is working correctly.
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n[1] Checking Dependencies")
    print("-" * 60)
    
    dependencies = {
        "pylatexenc": "LaTeX parsing library",
        "json": "JSON serialization",
    }
    
    all_ok = True
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            print(f"✓ {dep:20} - {description}")
        except ImportError:
            print(f"✗ {dep:20} - MISSING")
            all_ok = False
    
    return all_ok

def check_files():
    """Check if all required files exist"""
    print("\n[2] Checking Files")
    print("-" * 60)
    
    required_files = [
        "latex_parser.py",
        "test_parser.py",
        "example_usage.py",
        "integration_example.py",
        "QUICK_REFERENCE.py",
        "LATEX_PARSER_README.md",
        "IMPLEMENTATION_SUMMARY.md",
        "DELIVERABLES.md",
    ]
    
    base_dir = Path(__file__).parent
    all_ok = True
    
    for filename in required_files:
        filepath = base_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"✓ {filename:30} ({size_kb:6.1f} KB)")
        else:
            print(f"✗ {filename:30} - NOT FOUND")
            all_ok = False
    
    return all_ok

def test_import():
    """Test if the parser module imports correctly"""
    print("\n[3] Testing Module Import")
    print("-" * 60)
    
    try:
        from latex_parser import (
            LaTeXParser, Paper, Section, Paragraph, Sentence, 
            EnvironmentType, parse_paper
        )
        print("✓ Successfully imported all classes")
        print("  - LaTeXParser")
        print("  - Paper")
        print("  - Section")
        print("  - Paragraph")
        print("  - Sentence")
        print("  - EnvironmentType")
        print("  - parse_paper()")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_parsing():
    """Test parsing on sample papers"""
    print("\n[4] Testing Paper Parsing")
    print("-" * 60)
    
    from latex_parser import parse_paper
    
    test_papers = [
        Path("P:\\AI4S\\survey_paper\\cs\\arXiv-2209.00796v15\\main.tex"),
        Path("P:\\AI4S\\survey_paper\\cs\\arXiv-2503.24377v1\\main.tex"),
    ]
    
    all_ok = True
    for paper_path in test_papers:
        if not paper_path.exists():
            print(f"⊘ {paper_path.name} - Test file not found")
            continue
        
        try:
            paper = parse_paper(paper_path)
            status = "✓"
            details = (
                f"Sections: {len(paper.children)}, "
                f"Refs: {len(paper.references)}"
            )
            print(f"{status} {paper_path.name:30} - {details}")
        except Exception as e:
            print(f"✗ {paper_path.name:30} - ERROR: {str(e)[:50]}")
            all_ok = False
    
    return all_ok

def test_features():
    """Test key features of the parser"""
    print("\n[5] Testing Key Features")
    print("-" * 60)
    
    try:
        from latex_parser import parse_paper, EnvironmentType
        
        paper_path = Path("P:\\AI4S\\survey_paper\\cs\\arXiv-2503.24377v1\\main.tex")
        if not paper_path.exists():
            print("⊘ Cannot test features - sample paper not found")
            return True
        
        paper = parse_paper(paper_path)
        
        # Test 1: Metadata extraction
        assert paper.title, "Title not extracted"
        print("✓ Metadata extraction")
        
        # Test 2: Section hierarchy
        assert len(paper.children) > 0, "No sections found"
        print("✓ Section hierarchy parsing")
        
        # Test 3: Citation extraction
        total_citations = sum(
            len(s.citations)
            for sec in paper.children
            for para in sec.paragraphs
            for s in para.sentences
        )
        assert total_citations > 0, "No citations found"
        print(f"✓ Citation extraction ({total_citations} found)")
        
        # Test 4: Reference parsing
        assert len(paper.references) > 0, "No references found"
        print(f"✓ Reference parsing ({len(paper.references)} entries)")
        
        # Test 5: JSON skeleton export
        skeleton = paper.get_skeleton()
        assert "title" in skeleton, "Skeleton missing title"
        assert "sections" in skeleton, "Skeleton missing sections"
        print("✓ JSON skeleton export")
        
        # Test 6: Environment type detection
        has_text = any(
            s.environment_type == EnvironmentType.TEXT
            for sec in paper.children
            for para in sec.paragraphs
            for s in para.sentences
        )
        assert has_text, "No text environment detected"
        print("✓ Environment type detection")
        
        return True
        
    except AssertionError as e:
        print(f"✗ Feature test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Feature test error: {e}")
        return False

def print_summary(results):
    """Print validation summary"""
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_ok = all(results.values())
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name:30} {status}")
    
    print("="*60)
    
    if all_ok:
        print("\n✓ All checks passed! LaTeX parser is ready to use.\n")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the issues above.\n")
        return 1

def main():
    """Run all validation checks"""
    print("\n" + "="*60)
    print("LaTeX Parser Validation")
    print("="*60)
    
    results = {
        "Dependencies": check_dependencies(),
        "Files": check_files(),
        "Import": test_import(),
        "Parsing": test_parsing(),
        "Features": test_features(),
    }
    
    exit_code = print_summary(results)
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
