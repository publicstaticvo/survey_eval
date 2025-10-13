import os
import glob
import tqdm
from tex_parser import LaTeXPaperParser


for f in tqdm.tqdm(glob.glob("P:\\AI4S\\survey_paper\\*\\main.tex"), desc="Processing files"):
    base_path = os.path.join(*f.split("\\")[:-1])
    with open(f, encoding='utf-8') as fp: paper = fp.read()
    parser = LaTeXPaperParser(paper, base_path)
    paper = parser.parse()
    with open(os.path.join(base_path, "parse.tex"), "w+", encoding='utf-8') as fp: fp.write(str(paper))