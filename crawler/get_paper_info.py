import json
import os
import time
import tqdm
import logging
import multiprocessing
from request_utils import s2api_search_paper
logging.basicConfig(filename="../logs/info.log", level=logging.INFO)
fields = ['paperId', 'title', 'abstract', 'venue', 'fieldsOfStudy', "externalIds"]


if __name__ == "__main__":
    t = time.time()
    n_workers = 20
    logging.info(f"使用并行进程数: {n_workers}")
    # with open("paper2025.json", encoding='utf-8') as f:
    #     papers = json.load(f)['cs']
    with multiprocessing.Pool(processes=n_workers) as pool:
        pending_results = []
        with open("../../paper2025_with_abs_and_topic.jsonl", encoding='utf-8') as f:
            for line in f:
                paper = json.loads(line.strip())
                pending_results.append([pool.apply_async(s2api_search_paper, (paper['title'], fields)), paper])
        logging.info("finish pending results")
        with open("paper2025_details.jsonl", "w+") as f:
            for async_result, paper in tqdm.tqdm(pending_results): 
                result = async_result.get()
                data = result.get("data", [])
                for x in data:
                    venue = x.get("venue", "")
                    fields_of_study = x.get("fieldsOfStudy", [])
                    if venue and fields_of_study:
                        paper['venue'] = venue
                        paper['fieldsOfStudy'] = fields_of_study
                        f.write(json.dumps(paper) + "\n")
    logging.info(f"Time: {time.time() - t:.4f}")
