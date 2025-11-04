import json
import os
import time
import tqdm
import logging
import multiprocessing
from request_utils import s2api_search_paper
logging.basicConfig(filename="../logs/info.log", level=logging.INFO)
fields = ['paperId', 'title', 'abstract', 'venue', 'fieldsOfStudy', "externalIds"]
foutput = "paper2025_details.jsonl"


def request_and_output(paper):
    response = s2api_search_paper(paper['title'], fields)
    data = response.get("data", [])
    for x in data:
        if paper['title'] not in x['title']: continue
        venue = x.get("venue", "")
        if "arxiv" in venue.lower(): continue
        fields_of_study = x.get("fieldsOfStudy", [])
        if venue and fields_of_study:
            paper['venue'] = venue
            paper['fieldsOfStudy'] = fields_of_study
            with open(foutput, "a+") as f:
                f.write(json.dumps(paper) + "\n")
            return


def parallel_main():
    t = time.time()
    n_workers = 10
    logging.info(f"使用并行进程数: {n_workers}")
    with multiprocessing.Pool(processes=n_workers) as pool:
        pending_results = []
        with open("../../paper2025_with_abs_and_topic.jsonl", encoding='utf-8') as f:
            for line in f:
                paper = json.loads(line.strip())
                if paper['title'].endswith("."): paper['title'] = paper['title'][:-1]
                paper['title'] = paper['title'].strip()
                pending_results.append(pool.apply_async(request_and_output, (paper,)))
        logging.info("finish pending results")
        for async_result in tqdm.tqdm(pending_results): async_result.get()
    logging.info(f"Time: {time.time() - t:.4f}")


def main():
    t = time.time()
    with open("../../paper2025_with_abs_and_topic.jsonl", encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
            paper = json.loads(line.strip())
            if paper['title'].endswith("."): paper['title'] = paper['title'][:-1]
            paper['title'] = paper['title'].strip()
            request_and_output(paper)
    logging.info(f"Time: {time.time() - t:.4f}")


if __name__ == "__main__":
    parallel_main()