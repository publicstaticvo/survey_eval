import os
import glob
import json
import tqdm
from tex_parser import LatexPaperParser, process_input_commands, construct_citation_info
from utils import detect_encoding


def is_main_paper(paper: str) -> bool:
    return ("\\begin{document}" in paper and "\\section" in paper)


subjects = ["cs", "econ", "eess", "math", "phy", "q-bio", "q-fin", "stat"]

for s in subjects:
    root = f"P:\\AI4S\\survey_eval\\crawled_papers\\{s}"
    paper_projects = [os.path.join(root, x) for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
    # paper_projects = ["P:\\AI4S\\survey_eval\\crawled_papers\\cs\\2502.11211"]
    for base_path in tqdm.tqdm(paper_projects, desc="Processing files"):
        citation_output = os.path.join(base_path, "citation.jsonl")
        if os.path.exists(citation_output): continue
        main_tex = os.path.join(base_path, "main.tex")
        try:
            if os.path.isfile(main_tex):
                try:
                    with open(main_tex, encoding='utf-8') as f: paper = f.read()
                except UnicodeDecodeError:
                    paper, _ = detect_encoding(main_tex)
                longest_paper = process_input_commands(paper, base_path)
            else:
                tex_paths = glob.glob(os.path.join(base_path, "*.tex"))
                longest_paper_length, longest_paper, longest_fp = 0, "", ""
                for tex in tex_paths:
                    with open(tex, encoding='utf-8') as f: 
                        paper = process_input_commands(f.read(), base_path)
                    if not is_main_paper(paper): continue
                    if len(paper) > longest_paper_length:
                        longest_paper_length, longest_paper, longest_fp = len(paper), paper, tex
                if not longest_paper: continue
            print(base_path)
            parser = LatexPaperParser(longest_paper, base_path)
            paper = parser.parse()
            if paper is not None:
                citations = construct_citation_info(paper, parser)
                if citations:
                    with open(citation_output, "w+", encoding='utf-8') as f: 
                        for citation in citations: f.write(json.dumps(citation) + "\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
    break
