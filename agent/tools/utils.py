import re
import json
import time
import random
import logging
import requests
import unidecode
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor as TPE

from tool_config import LLMServerInfo

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}
email_pool = [
    "dailyyulun@gmail.com",
    "fqpcvtjj@hotmail.com",
    "ts.yu@siat.ac.cn",
    "yutianshu.yts@alibaba-inc.com",
    "yts17@mails.tsinghua.edu.cn"
    "yutianshu2025@ia.ac.cn",
    "yutianshu25@ucas.ac.cn",
    "dailyyulun@163.com",
    "lundufiles@163.com",
    "lundufiles123@163.com"
]
URL_DOMAIN = "https://openalex.org/"


def request_template(method: str, url: str, headers: dict, parameters: dict, timeout: int = 60, sleep: bool = True) -> dict:
    if sleep: time.sleep(2 * random.random())
    if method == "post": 
        headers['Content-type'] = "application/json"
        request_method = requests.post
    else:
        request_method = requests.get
    response = request_method(url, headers=headers, params=parameters, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data


def callLLM(
        llm: LLMServerInfo, 
        messages: list | str, 
        sampling_params: dict, 
        return_reasoning: bool = False,
        retry: int = 5
    ) -> str:
    if not llm.base_url or llm.api_key is None: return ""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    sampling_params.update({"model": llm.model, "messages": messages})
    while retry > 0:
        try:            
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {llm.api_key}'}
            url = f"{llm.base_url}/v1/chat/completions"
            response = request_template("post", url, headers, json.dumps(sampling_params), 600, False)
            message = response['choices'][0]['message']
            text = message['content']     
            think = message.get("reasoning_content", "")
            if not think and "</think>" in text:
                think = text[:text.index("</think>")]
                text = text[text.index("</think>") + 8:]
            if return_reasoning: return text, think
            return text
        except Exception as e:
            retry -= 1
            logging.error(f"Error: {e}, Retry: {retry}")
            time.sleep(10)
    return ""


def openalex_search_paper(
        endpoint: str, 
        filter: dict = None, 
        do_sample: bool = False,
        per_page: int = -1, 
        add_email: bool | str = True, 
        retry: int = 3, 
        **request_kwargs
    ) -> dict:
    assert per_page <= 200, "Per page is at most 200"
    # 整理参数
    url = f"https://api.openalex.org/{endpoint}"
    if filter:
        # filter
        filter_string = ",".join([f"{k}:{v}" for k, v in filter.items()])
        request_kwargs["filter"] = filter_string
    if do_sample:
        # use per_page as num_samples
        request_kwargs['sample'] = per_page
        request_kwargs['seed'] = random.randint(0, 32767)        
    if add_email:
        request_kwargs['mailto'] = add_email if isinstance(add_email, str) else random.choice(email_pool)
    if per_page > 25: 
        request_kwargs['per-page'] = per_page
    # Go!
    while retry != 0:
        try:            
            return request_template("get", url, None, request_kwargs)
        except requests.exceptions.RequestException as e:
            what = str(e)
            logging.error(f"OpenAlex {e}, Retry: {retry}")
            if any(f"{code} Client Error" in what for code in [400, 401, 403, 404]): return {}
            retry -= 1
            time.sleep(1)
    return {}


def index_to_abstract(indexes: dict | None):
    if not indexes: return None
    abstract_length = max(v[-1] for v in indexes.values())
    abstract = ["<mask>" for _ in range(abstract_length + 1)]
    for k, v in indexes.items():
        for i in v:
            abstract[i] = k
    return " ".join(abstract)


def valid_check(query: str, target: str) -> bool:
    def normalize(text: str) -> str:
        text = unidecode.unidecode(text)
        text = re.sub(r"[^0-9a-zA-Z\s]", "", text)
        return text.lower()
    
    return normalize(query) == normalize(target)


def split_content_to_paragraph(content: dict | list):
    if isinstance(content, list): return content
    paragraphs = content['paragraphs']
    for section in content['sections']:
        paragraphs.extend(split_content_to_paragraph(section))
    return paragraphs


def prepare_paragraphs_for_clarity_2(paper_content: dict):
    paragraphs = []
    for i, p in enumerate(paper_content['paragraphs']):
        current = " ".join(x['text'] for x in p)
        if i >= 1: paragraphs.append({"text": current, "pre_text": paragraphs[-1]['text']})
        else: paragraphs.append({"text": current, "pre_text": "This is the first paragraph in this section."})
    for s in paper_content['sections']:
        sub_paragraphs = prepare_paragraphs_for_clarity(s)
        if sub_paragraphs and s['title']: 
            sub_paragraphs[0]['text'] = f"{s['title']}\n\n" + sub_paragraphs[0]['text']
        paragraphs.extend(sub_paragraphs)
    return paragraphs


def prepare_paragraphs_for_clarity(paper_content: dict):
    paragraphs = [" ".join(x['text'] for x in p) for p in paper_content['paragraphs']]
    for s in paper_content['sections']:
        sub_paragraphs = prepare_paragraphs_for_clarity(s)
        if sub_paragraphs and s['title']: 
            sub_paragraphs[0] = f"{s['title']}\n\n" + sub_paragraphs[0]
        paragraphs.extend(sub_paragraphs)
    paragraph_inputs = []
    for i, p in paragraphs:
        if i == 0: paragraph_inputs.append({"text": p, "pre_text": "This is the first paragraph in this section."})
        else: paragraph_inputs.append({"text": p, "pre_text": paragraphs[i - 1]})
    return paragraphs
