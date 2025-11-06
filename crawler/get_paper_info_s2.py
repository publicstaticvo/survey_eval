import json
import os
import time
import tqdm
import logging
import multiprocessing
from request_utils import s2api_search_paper
logging.basicConfig(filename="../logs/info.log", level=logging.INFO)
fields = ",".join(['title', 'venue', 'publicationVenue', 's2FieldsOfStudy', "externalIds"])
fin = "../../paper2025_with_abs.jsonl"
foutput = "paper2025_details.jsonl"
dont_parallel = False


def request_and_output(paper):
    x = s2api_search_paper(arxiv_id=paper['arxiv_url'], fields=fields)
    venue = x.get("venue", "")
    publication_venue = x.get("publicationVenue", {})
    fields_of_study = x.get("s2FieldsOfStudy", [])
    if (venue or publication_venue) and fields_of_study:
        external_ids = x.get("externalIds", {})            
        paper['venue'] = publication_venue if publication_venue else venue
        paper['external_ids'] = external_ids
        paper['fields_of_study'] = fields_of_study
        with open(foutput, "a+", encoding='utf-8') as f:
            f.write(json.dumps(paper) + "\n")


def parallel_main():
    t = time.time()
    n_workers = 10
    logging.info(f"使用并行进程数: {n_workers}")
    with multiprocessing.Pool(processes=n_workers) as pool:
        pending_results = []
        with open(fin, encoding='utf-8') as f:
            for line in f:
                paper = json.loads(line.strip())
                if paper['title'].endswith("."): paper['title'] = paper['title'][:-1]
                paper['title'] = paper['title'].strip()
                pending_results.append(pool.apply_async(request_and_output, (paper,)))
        logging.info("finish pending results")
        for async_result in tqdm.tqdm(pending_results): async_result.get()
    logging.info(f"Time: {time.time() - t:.4f}")


def parallel_main_windows():
    t = time.time()
    n_workers = 10
    logging.info(f"使用并行进程数: {n_workers}")
    with multiprocessing.Pool(processes=n_workers) as pool:
        pending_results = []
        with open(fin, encoding='utf-8') as f:
            for line in f:
                paper = json.loads(line.strip())
                job = pool.apply_async(s2api_search_paper, (paper['title'], paper['arxiv_url'], fields))
                pending_results.append([job, paper])
        logging.info("finish pending results")
        with open(foutput, "w+", encoding='utf-8') as f:
            for async_result, paper in tqdm.tqdm(pending_results): 
                x = async_result.get()
                venue = x.get("venue", "")
                publication_venue = x.get("publicationVenue", {})
                fields_of_study = x.get("s2FieldsOfStudy", [])
                if (venue or publication_venue) and fields_of_study:
                    external_ids = x.get("externalIds", {})            
                    paper['venue'] = publication_venue if publication_venue else venue
                    paper['external_ids'] = external_ids
                    paper['fields_of_study'] = fields_of_study
                    f.write(json.dumps(paper) + "\n")
    logging.info(f"Time: {time.time() - t:.4f}")


def main():
    t = time.time()
    with open(fin, encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
            paper = json.loads(line.strip())
            if paper['title'].endswith("."): paper['title'] = paper['title'][:-1]
            paper['title'] = paper['title'].strip()
            request_and_output(paper)
    logging.info(f"Time: {time.time() - t:.4f}")


if __name__ == "__main__":
    if dont_parallel: main()
    elif os.name == "nt": parallel_main_windows()
    else: parallel_main()