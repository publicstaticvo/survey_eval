import os
import glob
import tqdm
from tex_parser import LatexPaperParser
from paper_elements import *


def debug_sentences(
        sentences: List[Union[LatexSentence, LatexEnvironment]], 
        parser: LatexPaperParser, 
        start_id: int,
        file_point
    ):
    for i, sentence in enumerate(sentences):
        file_point.write(f"{i + start_id}\t{sentence.__repr__()}\n")
        if isinstance(sentence, dict): print(sentence)
        if sentence.citations:
            citations = [f"{x}: {parser.get_bibliography_entry(x)}" for x in sentence.citations]
            citations = '\n\t-- '.join(citations)
            file_point.write(f"-\tCitations: {citations}\n")
    return start_id + len(sentences)


def is_main_paper(paper: str) -> bool:
    return ("\\begin{document}" in paper and "\\section" in paper)


def debug_paper():
    domain = ""
    root = "P:\\AI4S\\survey_paper\\mat"
    paper_projects = [os.path.join(root, x) for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
    # paper_projects = ["P:\\AI4S\\survey_paper\\mat\\arXiv-2408.12171v1"]

    for base_path in tqdm.tqdm(paper_projects, desc="Processing papers"):
        if os.path.isfile(os.path.join(base_path, "main.tex")):
            tex_paths = [os.path.join(base_path, "main.tex")]
        else:
            tex_paths = glob.glob(os.path.join(base_path, "*.tex"))
        for f in tex_paths:
            if "parse" in f or "FULL" in f: continue
            with open(f, encoding='utf-8') as fp: paper = fp.read()
            parser = LatexPaperParser(paper, base_path)
            if not is_main_paper(parser.latex_content): continue
            if base_path == domain: print(f"Warning: multiple documents in project {domain}")
            domain = base_path
            paper = parser.parse()
            if paper is not None:
                with open(os.path.join(base_path, "parse.tex"), "w+", encoding='utf-8') as f: 
                    sentence_id = 0
                    f.write(f"Title: {paper.title}, Author: {paper.author}\n")
                    if paper.abstract is not None:
                        f.write("0 Abstract:\n")
                        for i, paragraph in enumerate(paper.abstract.children):
                            f.write(f"0-{i + 1} {paragraph.name}\n")
                            abstract_sentences = paragraph.get_sentences()
                            sentence_id = debug_sentences(abstract_sentences, parser, sentence_id, f)

                    for i, section in enumerate(paper.sections):
                        f.write(f"{i + 1} {section.name}\n")
                        subsection_start_idx = -1
                        for j, subsection in enumerate(section.children):
                            if isinstance(subsection, LatexParagraph):
                                assert subsection_start_idx == -1
                                f.write(f"Paragraph {i + 1}-{j + 1} {subsection.name}\n")
                                sentences = subsection.get_sentences()
                                sentence_id = debug_sentences(sentences, parser, sentence_id, f)
                            else:
                                if subsection_start_idx == -1: subsection_start_idx = j
                                subsubsection_start_idx = -1
                                f.write(f"SubSection {i + 1}.{j - subsection_start_idx + 1} {subsection.name}\n")
                                for k, subsubsection in enumerate(subsection.children):
                                    if isinstance(subsubsection, LatexParagraph):
                                        assert subsubsection_start_idx == -1
                                        f.write(f"Paragraph {i + 1}.{j - subsection_start_idx + 1}-{k + 1} {subsubsection.name}\n")
                                        sentences = subsubsection.get_sentences()
                                        sentence_id = debug_sentences(sentences, parser, sentence_id, f)
                                    else:
                                        if subsubsection_start_idx == -1: subsubsection_start_idx = k
                                        f.write(f"SubSection {i + 1}.{j - subsection_start_idx + 1}.{k - subsubsection_start_idx + 1} {subsubsection.name}\n")
                                        for l, paragraph in enumerate(subsubsection.children):
                                            f.write(f"Paragraph {i + 1}.{j - subsection_start_idx + 1}.{k - subsubsection_start_idx + 1}-{l + 1} {paragraph.name}\n")
                                            sentences = subsubsection.get_sentences()
                                            sentence_id = debug_sentences(sentences, parser, sentence_id, f)


def debug_citation():
    with open("P:\\AI4S\\survey_eval\\crawled_papers\\cs\\2209.00796\\main.tex", encoding='utf-8') as f:
        content = f.read()
    parser = LatexPaperParser(content, "P:\\AI4S\\survey_eval\\crawled_papers\\cs\\2209.00796")
    paper = parser.parse()
    all_citations = paper.map_citations_to_sentence()
    print(len(all_citations))
    for i in range(10):
        print(all_citations[i])


if __name__ == "__main__":
    debug_citation()
