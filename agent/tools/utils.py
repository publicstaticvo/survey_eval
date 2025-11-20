import json
import time
import random
import logging
import requests
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


def request_template(url: str, headers: dict, parameters: dict, timeout: int = 60, sleep: bool = True) -> dict:
    if sleep: time.sleep(2 * random.random())
    response = requests.get(url, headers=headers, params=parameters, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data


def callLLM(base_url: str, api_key: str, messages: list | str, model: str, sampling_params: dict, retry: int = 5) -> str:
    if not base_url or not api_key: return
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    sampling_params.update({"model": model, "messages": messages})
    while retry > 0:
        try:            
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
            url = f"{base_url}/v1/chat/completions"
            response = request_template(url, headers, json.dumps(sampling_params), 600, False)
            message = response['choices'][0]['message']
            text = message['content']     
            think = message.get("reasoning_content", "")
            if not think and "</think>" in text:
                think = text[:text.index("</think>")]
                text = text[text.index("</think>") + 8:]
            return text
        except Exception as e:
            retry -= 1
            logging.error(f"Error: {e}, Retry: {retry}")
            time.sleep(10)


def openalex_search_paper(endpoint: str, filter: dict = None, max_results: int = -1, add_mail: bool | str = True, retry: int = 3) -> dict:
    assert max_results <= 200, "Per page is at most 200"
    while retry != 0:
        try:
            url = f"https://api.openalex.org/{endpoint}"
            if filter is not None:
                # filter
                filter_string = ",".join([f"{k}:{v}" for k, v in filter.items()])
                request_parameters = {"filter": filter_string}
            else:
                request_parameters = {}
                if max_results >= 0:
                    # sample
                    request_parameters['sample'] = max_results
                    request_parameters['seed'] = random.randint(0, 32767)
                    if max_results > 25:
                        request_parameters['per-page'] = 200
            if add_mail:
                request_parameters['mailto'] = add_mail if isinstance(add_mail, str) else random.choice(email_pool)
            return request_template(url, None, request_parameters)
        except requests.exceptions.RequestException as e:
            what = str(e)
            logging.error(f"OpenAlex {e}, Retry: {retry}")
            if any(f"{code} Client Error" in what for code in [400, 401, 403, 404]): return {}
            retry -= 1
            time.sleep(1)
    return {}
