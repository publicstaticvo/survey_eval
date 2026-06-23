import os
import json
from tex_parser import LatexPaperParser


def is_main_paper(paper: str) -> bool:
    return ("\\begin{document}" in paper and "\\section" in paper)


def test_skeleton():
    sample_base = "P:\\AI4S\\survey_eval\\crawled_papers\\cs\\2209.00796"
    main_tex = os.path.join(sample_base, "main.tex")
    if not os.path.exists(main_tex):
        print("Sample test file not found:", main_tex)
        return

    parser = LatexPaperParser()
    paper = parser.parse(main_tex)
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
