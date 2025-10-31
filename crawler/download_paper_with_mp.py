import json
import os
import time
import tqdm
import logging
import requests
import multiprocessing
from requests.exceptions import RequestException
logging.basicConfig(filename="../logs/download.log", level=logging.INFO)


def download_paper(url, fn):
    save_path = f"../crawled_papers/pdf/{fn}.pdf"
    if os.path.exists(save_path): return
    retry = 3
    while retry > 0:        
        try:                
            # Download the file
            response = requests.get(url, timeout=600, stream=True)
            response.raise_for_status()
            
            # Save to file
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = os.path.getsize(save_path) / (1024 * 1024)  # Size in MB
            logging.info(f"Successfully downloaded {fn} to: {save_path} ({file_size:.2f} MB)")
            return True
            
        except RequestException as e:
            retry -= 1
            logging.error(f"Error downloading paper {fn}: {e}. Retry: {retry}")
            if retry > 0: time.sleep(4 ** (3 - retry))
        except Exception as e:
            retry -= 1
            logging.error(f"Unknown error downloading paper {fn}: {e}. Retry: {retry}")
            if retry > 0: time.sleep(1)


if __name__ == "__main__":
    t = time.time()
    n_workers = 10
    logging.info(f"使用并行进程数: {n_workers}")
    with open("../crawled_papers/citations/urls.json") as f: urls = json.load(f)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pending_results = []
        for title, url in urls.items():
            if "arxiv.org" in url: title = url.replace("https://arxiv.org/pdf/", "")
            pending_results.append(pool.apply_async(download_paper, (url, title.replace(" ", "+").replace(":", "--"))))
        logging.info("finish pending results")
        for async_result in tqdm.tqdm(pending_results): async_result.get()
    logging.info(f"Time: {time.time() - t:.4f}")
