import glob
import tqdm
import logging
import multiprocessing
from pdf_parser import GROBIDParser
logging.basicConfig(filename="../logs/pdfparse.log", level=logging.INFO)


def parse_pdf_with_parser(filename: str):
    parser = GROBIDParser()    
    paper_xml = parser.process_pdf_to_xml(filename)
    if not paper_xml: return
    try:
        paper = parser.parse_xml(paper_xml)
    except:
        print(filename)
        return
    paper_name = filename.split("/")[-1].replace("{", "").replace("}", "").replace(".pdf", "")
    with open(f"../crawled_papers/paper_info/{paper_name}.txt", "w+") as f:
        f.write(paper.get_skeleton())


n_workers = 10
pdf_files = glob.glob("../crawled_papers/pdf/*.pdf")
with multiprocessing.Pool(processes=n_workers) as pool:
    pending_results = []
    for filename in pdf_files:
        pending_results.append(pool.apply_async(parse_pdf_with_parser, (filename,)))
    logging.info("finish pending results")
    for async_result in tqdm.tqdm(pending_results): async_result.get()
