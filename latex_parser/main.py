import os
import glob
import tqdm
from tex_parser import LatexPaperParser


domain = ""

for f in tqdm.tqdm(glob.glob("P:\\AI4S\\survey_paper\\*\\*.tex"), desc="Processing files"):
    base_path = os.path.join(*f.split("\\")[:-1])
    with open(f, encoding='utf-8') as fp: paper = fp.read()
    if "\\begin{document}" not in paper or "\\end{document}" not in paper: continue
    if base_path == domain: print(f"Warning: multiple documents in project {domain}")
    domain = base_path
    try:
        parser = LatexPaperParser(paper, base_path)
        paper = parser.parse()
        if paper is not None:
            with open(os.path.join(base_path, "parse.tex"), "w+", encoding='utf-8') as fp: 
                fp.write(str(paper))
    except Exception as e:
        print(f"Top parse exception: {e}")