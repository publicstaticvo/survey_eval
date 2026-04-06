import os
import glob
import json
import tqdm
from tex_parser import LatexPaperParser, process_input_commands, construct_citation_info
from utils import detect_encoding


def is_main_paper(paper: str) -> bool:
    return ("\\begin{document}" in paper and "\\section" in paper)


def test_skeleton():
    sample_base = "P:\\AI4S\\survey_eval\\crawled_papers\\cs\\2209.00796"
    main_tex = os.path.join(sample_base, "main.tex")
    if not os.path.exists(main_tex):
        print("Sample test file not found:", main_tex)
        return

    try:
        with open(main_tex, encoding='utf-8') as f:
            paper_text = f.read()
    except UnicodeDecodeError:
        paper_text, _ = detect_encoding(main_tex)

    paper_text = process_input_commands(paper_text, sample_base)
    parser = LatexPaperParser(paper_text, sample_base)
    paper = parser.parse()
    if paper is None:
        print("Parser returned None for sample paper.")
        return

    skeleton = paper.get_skeleton()
    assert isinstance(skeleton, dict), "get_skeleton() should return a dict"
    assert skeleton.get('title') == paper.title, "Title mismatch in skeleton"
    assert 'sections' in skeleton and isinstance(skeleton['sections'], list), "Sections missing or invalid"
    print("get_skeleton() is available and returned JSON structure.")
    print("Skeleton keys:", list(skeleton.keys()))
    print("Top-level section count:", len(skeleton['sections']))
    print("Sample skeleton output:")
    json.dumps(skeleton, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    test_skeleton()
