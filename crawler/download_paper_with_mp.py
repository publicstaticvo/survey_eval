import json
import os
import glob
import time
import tqdm
import argparse
import regex
import random
import requests
import multiprocessing
from requests.exceptions import ConnectionError, Timeout, RequestException


def download_paper(url):
    pass    


if __name__ == "__main__":
    t = time.time()
    n_workers = 60
    print(f"使用并行进程数: {n_workers}")
    with multiprocessing.Pool(processes=n_workers) as pool:
        pending_results = []
        with open(args.inputs, encoding="utf-8") as f_in:
            for line in f_in:
                x = json.loads(line.strip())
                if not x: continue
                pending_results.append(pool.apply_async(download_paper, (x, args, args.output)))
        print("finish pending results")
        for async_result in tqdm.tqdm(pending_results): async_result.get()
    print(f"Time: {time.time() - t:.4f}")
