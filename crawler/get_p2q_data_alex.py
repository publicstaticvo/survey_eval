import os
import json
import tqdm
import random
import logging
from typing import Any
from request_utils import openalex_search_paper
from concurrent.futures import ThreadPoolExecutor as TPE
logging.basicConfig(filename="../logs/alex.log", level=logging.INFO)
domain = "https://openalex.org/"
with open("openalex_level0_concepts.jsonl") as f:
    all_level0_concepts = [json.loads(line)['id'].replace(domain, "") for line in f]


def index_to_abstract(indexes):
    abstract_length = max(v[-1] for v in indexes.values())
    abstract = ["<mask>" for _ in range(abstract_length + 1)]
    for k, v in indexes.items():
        for i in v:
            abstract[i] = k
    return " ".join(abstract)


def is_valid_paper(paper: dict) -> bool:
    return paper.get('abstract_inverted_index', None) and \
           paper.get('referenced_works', []) and \
           paper.get('related_works', []) and \
           paper.get('doi', None) and \
           paper.get('language', "") == "en"


def is_easy_negative(paper: dict, concepts: set[str]) -> bool:
    return paper.get('abstract_inverted_index', None) and \
           paper.get('language', "") == "en" and \
           paper.get('concepts', []) and \
           all(x['id'].replace(domain, "") not in concepts for x in paper['concepts'] if x['level'] == 0)


def keep_critical_details(x):
    return {
        "title": x['title'],
        "abstract": index_to_abstract(x['abstract_inverted_index']),
        "year": x['publication_year'],
        "citation_count": x['cited_by_count'], 
        "doi": x['doi'],
        "concepts": sorted(x['concepts'], key=lambda y: y['score']),
        "topics": x['topics'],
        "referenced_works": [y.replace(domain, "") for y in x['referenced_works']],
        "related_works": [y.replace(domain, "") for y in x['related_works']]
    }


def lvalue_update(d: dict, update: dict) -> dict:
    d.update(update)
    return d


num_seeds, per_page = 50000, 200
step1_path = "../paper_to_query/seeds.json"
step2_path = "../paper_to_query/hard_neg.json"
step3_path = "../paper_to_query/easy_neg.json"

# 第一步：随机选择一些文章开始，可以选择publication_year=2025的。
def step1():
    seed_set = {}
    bar = tqdm.trange(num_seeds, desc="step 1")
    while len(seed_set) < num_seeds:
        results = openalex_search_paper("works", max_results=per_page)
        for x in results.get('results'):
            if not is_valid_paper(x): continue            
            if (paper_id := x['id'].replace(domain, "")) not in seed_set:
                try:
                    seed_set[paper_id] = keep_critical_details(x)
                    bar.update(1)
                except Exception as e:
                    logging.error(str(e))
    with open(step1_path, "w+") as f:
        json.dump(seed_set, f)
    return seed_set


# 第二步：取引用文献的论文
def step2(seed: dict[str, dict[str, Any]]):
    hard_negative_set = {}
    for paper_id in tqdm.tqdm(seed, desc="step 2"):
        paper = seed[paper_id]
        references = [x for x in paper['referenced_works'] if x not in seed and x not in hard_negative_set]
        for sample in references:
            if sample in hard_negative_set: continue
            assert sample.startswith("W"), sample
            paper_info = openalex_search_paper(f"works/{sample}")
            if is_valid_paper(paper_info): 
                hard_negative_set[sample] = keep_critical_details(paper_info).update({"related_id": paper_id})
    with open(step2_path, "w+") as f:
        json.dump(hard_negative_set, f)
    return hard_negative_set


def parallel_step2(seed: dict[str, dict[str, Any]], n_workers=20):
    hard_negative_set = {}

    def step2_inner(paper):
        paper_id, paper = paper
        references = [x for x in paper['referenced_works'] if x not in seed]
        for sample in references:
            if sample in hard_negative_set: continue
            assert sample.startswith("W"), sample
            paper_info = openalex_search_paper(f"works/{sample}")
            if is_valid_paper(paper_info): 
                return lvalue_update(keep_critical_details(paper_info), {"sample": sample, "related_id": paper_id})
    
    processed = set()
    while len(hard_negative_set) < num_seeds:
        has_updated = False
        pending_results = [(k, v) for k, v in seed.items() if k not in processed]
        with TPE(max_workers=n_workers) as executor:
            for paper_info in tqdm.tqdm(executor.map(step2_inner, pending_results), desc="step 2", total=len(pending_results)):
                if not paper_info: continue
                hard_negative_set[paper_info['sample']] = paper_info
                processed.add(paper_info['related_id'])
                has_updated = True
        if not has_updated: break

    with open(step2_path, "w+") as f:
        json.dump(hard_negative_set, f)
    return hard_negative_set


# 第三步：取完全不同领域的论文
def step3(seed: dict[str, dict[str, Any]], hard_negative: dict[str, dict[str, Any]]):
    easy_negative_set = {}

    def search_easy_negative_single_round(concepts: set[str]):
        response = openalex_search_paper("works", max_results=per_page)
        for paper_info in response.get("results"):
            if is_easy_negative(paper_info, concepts) and (neg := paper_info['id'].replace(domain, "")) \
                not in hard_negative and neg not in easy_negative_set: return paper_info

    for paper_id in tqdm.tqdm(seed, desc="step 3"):
        paper = seed[paper_id]
        level0_concepts = [x['id'].replace(domain, "") for x in paper['concepts'] if x['level'] == 0]
        if len(level0_concepts) == len(all_level0_concepts): level0_concepts = level0_concepts[:-1]
        exist_concepts = set(all_level0_concepts) - set(level0_concepts)
        while not (paper_info := search_easy_negative_single_round(concepts=exist_concepts)): pass
        neg_paper_id = paper_info['id'].replace(domain, "")
        easy_negative_set[neg_paper_id] = keep_critical_details(paper_info).update({"related_id": paper_id})
    with open(step3_path, "w+") as f:
        json.dump(easy_negative_set, f)
    return easy_negative_set


def parallel_step3(seed: dict[str, dict[str, Any]], hard_negative: dict[str, dict[str, Any]], n_workers=20):
    easy_negative_set = {}

    def search_easy_negative_single_round(concepts: set[str]):
        response = openalex_search_paper("works", max_results=per_page)
        for paper_info in response.get("results"):
            if is_easy_negative(paper_info, concepts) and paper_info['id'].replace(domain, "") not in hard_negative:
                return paper_info
                
    def step3_inner(paper):
        paper_id, paper = paper
        level0_concepts = [x['id'].replace(domain, "") for x in paper['concepts'] if x['level'] == 0]
        if len(level0_concepts) == len(all_level0_concepts): level0_concepts = level0_concepts[:-1]
        exist_concepts = set(all_level0_concepts) - set(level0_concepts)
        while not (paper_info := search_easy_negative_single_round(concepts=exist_concepts)): pass        
        return lvalue_update(keep_critical_details(paper_info), {"related_id": paper_id})

    processed = set()
    while len(easy_negative_set) < num_seeds:
        has_updated = False
        pending_results = [(k, v) for k, v in seed.items() if k not in processed]
        with TPE(max_workers=n_workers) as executor:
            for paper_info in tqdm.tqdm(executor.map(step3_inner, pending_results), desc="step 3", total=len(pending_results)):
                if not paper_info or (neg_paper_id := paper_info['id'].replace(domain, "")) not in easy_negative_set: continue
                easy_negative_set[neg_paper_id] = paper_info
                processed.add(paper_info['related_id'])
                has_updated = True
        if not has_updated: break
    
    with open(step3_path, "w+") as f:
        json.dump(easy_negative_set, f)
    return easy_negative_set


if __name__ == "__main__":
    if os.path.exists(step2_path):
        with open(step2_path) as f: hard_negative_set = json.load(f)
    else:
        if os.path.exists(step1_path):
            with open(step1_path) as f: seed_set = json.load(f)
        else: seed_set = step1()
        hard_negative_set = step2(seed_set)
    easy_negative_set = step3(seed_set, hard_negative_set)
