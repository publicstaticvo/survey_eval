import json
import os
import re
import time
import tqdm
import logging
import requests
import multiprocessing
logging.basicConfig(filename="../logs/download2.log", level=logging.INFO)


def download_paper(url, fn):
    save_path = f"../crawled_papers/pdf/{fn}.pdf"
    # if os.path.exists(save_path): return
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
            
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError) and response.status_code in [400, 401, 403, 404]: return
            retry -= 1
            logging.error(f"Error downloading paper {fn}: {e}. Retry: {retry}")
            if retry > 0: time.sleep(2)
        except Exception as e:
            retry -= 1
            logging.error(f"Unknown error downloading paper {fn}: {e}. Retry: {retry}")
            if retry > 0: time.sleep(1)


def normalize_title(title: str, url: str) -> tuple[str, str]:
    if "https://arxiv.org/pdf/" in url: title = url.replace("https://arxiv.org/pdf/", "")
    elif "https://arxiv.org/abs/" in url: 
        title = url.replace("https://arxiv.org/abs/", "")
        url = url.replace("abs", "pdf")
    if "/" in title: title = title.split("/")[-1]
    title = title.replace(" ", "+").replace(":", "--")
    title = re.sub(r"[\{\}\[\]\(\)\n]", "", title)
    return title, url


if __name__ == "__main__":
    t = time.time()
    n_workers = 10
    logging.info(f"使用并行进程数: {n_workers}")
    with open("../crawled_papers/citations/redownload.json") as f: urls = json.load(f)
    # with open("../crawled_papers/citations/inlinearXiv_redownload.json") as f: urls.update(json.load(f))
    # with open("../crawled_papers/citations/find.json") as f: urls.update(json.load(f))
    # print(len(urls))
    with multiprocessing.Pool(processes=n_workers) as pool:
        pending_results = []
        for title, url in urls.items():
            # title, url = normalize_title(title, url)
            pending_results.append(pool.apply_async(download_paper, (url, title)))
        logging.info("finish pending results")
        for async_result in tqdm.tqdm(pending_results): async_result.get()
    logging.info(f"Time: {time.time() - t:.4f}")
