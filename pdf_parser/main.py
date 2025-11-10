import os
import glob
import tqdm
import logging
import multiprocessing
from pdf_parser import GROBIDParser
logging.basicConfig(filename="../logs/pdfparse.log", level=logging.INFO)
ROOT = "/data/tsyu/survey_eval"


def parse_pdf_with_parser(filename: str):
    paper_name = filename.split("/")[-1].replace("{", "").replace("}", "").replace(".pdf", "")
    output = f"{ROOT}/crawled_papers/papers_full/{paper_name}.txt"
    # if os.path.exists(output): return
    parser = GROBIDParser()    
    paper_xml = parser.process_pdf_to_xml(filename)
    if not paper_xml: return
    try:
        paper = parser.parse_xml(paper_xml)
    except:
        print(filename)
        return
    with open(output, "w+") as f:
        f.write(paper.get_skeleton("all"))


n_workers = 8
pdf_files = glob.glob(f"{ROOT}/crawled_papers/pdf/*.pdf")
with multiprocessing.Pool(processes=n_workers) as pool:
    pending_results = []
    for filename in pdf_files:
        pending_results.append(pool.apply_async(parse_pdf_with_parser, (filename,)))
    logging.info("finish pending results")
    for async_result in tqdm.tqdm(pending_results): async_result.get()
