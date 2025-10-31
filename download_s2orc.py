import os
import re
import json
import tqdm
import wget
import urllib
import requests
s2orc_urls, release_id = {}, None
if os.path.exists("s2orc_info"):
    with open("s2orc_info") as f: s2orc_urls = json.load(f)
while True:
    if "files" not in s2orc_urls:
        release_id = requests.get("https://api.semanticscholar.org/datasets/v1/release/latest").json()['release_id']
        s2orc_urls = requests.get(f"https://s2api.ominiai.cn/generalProxy/datasets/v1/release/{release_id}/dataset/s2orc/", 
                                headers={"Authorization": f"Bearer sk-7CpRrtyqbVZwcb5k6b4eB6E176264374A2029e456c516a8b", 
                                        "OMINI-API-Model": "semantic"}).json()
        with open("s2orc_info", "w+") as f: json.dump(s2orc_urls, f)
    for i, url in tqdm.tqdm(enumerate(s2orc_urls['files'])):
        match = re.match(r"https://ai2-s2ag.s3.amazonaws.com/staging/(.*)/s2orc/(.*).gz(.*)", url)
        if not match: continue
        if release_id: assert match.group(1) == release_id
        shard_id = match.group(2)
        target = f"../s2orc/allenai/{shard_id}.gz"
        if not os.path.exists(target): 
            try:
                wget.download(url, out=target)
            except Exception as e:
                if isinstance(e, urllib.error.HTTPError) and e.code == 400:
                    s2orc_urls = {}
                    print("Receive 400 error. quitting and rerequest.")
                    break
                print(e)
    else:
        break
