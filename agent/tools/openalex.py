import asyncio, aiohttp
import random
import json
import re
from tenacity import (
    retry,
    stop_after_attempt,           # 最大重试次数
    wait_exponential,             # 指数退避
    retry_if_exception,           # 遇到什么异常才重试
    retry_if_result,              # 返回None的时候也要重试
)

from .utils import index_to_abstract
from .request_utils import async_request_template, RateLimit

EMAIL_POOL = [
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
RETRY_EXCEPTION_TYPES = [
    aiohttp.ClientError, 
    asyncio.TimeoutError, 
    aiohttp.ServerDisconnectedError, 
    json.JSONDecodeError,
    AssertionError,
    KeyError,
]
OPENALEX_SELECT = 'id,cited_by_count,counts_by_year,referenced_works,publication_date,created_date,abstract_inverted_index,title'
URL_DOMAIN = "https://openalex.org/"
# with open("redirect.json") as f: redirect = json.load(f)


class OpenAlexRedirect:

    redirect_file: str = "redirect.json"
    REDIRECT_LOCK = asyncio.Lock()
    redirect: dict = {}

    @classmethod
    async def init(cls, redirect_file="redirect.json"):
        async with cls.REDIRECT_LOCK:
            with open(redirect_file) as f: cls.redirect = json.load(f)
        cls.redirect_file = redirect_file

    @classmethod
    async def close(cls):
        async with cls.REDIRECT_LOCK:
            with open(cls.redirect_file, "w+") as f: json.dump(cls.redirect, f, indent=2)

    @classmethod
    async def get(cls, old, default=None):
        async with cls.REDIRECT_LOCK:
            return cls.redirect.get(old, default)

    @classmethod
    async def update(cls, old, new):
        async with cls.REDIRECT_LOCK:
            cls.redirect[old] = new


def openalex_should_retry(exception: BaseException) -> bool:
    if any(isinstance(exception, x) for x in RETRY_EXCEPTION_TYPES): return True
    if isinstance(exception, aiohttp.ClientResponseError) and exception.status not in [400, 401, 403, 404]: return True
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, exp_base=2, min=1, max=10),
    retry=retry_if_exception(openalex_should_retry),
    reraise=True
)
async def openalex_search_paper(
        endpoint: str,
        filter: list[tuple] | dict = {},
        do_sample: bool = False,
        per_page: int = 1,
        add_email: bool | str = True,
        select: str = OPENALEX_SELECT,
        **request_kwargs
    ) -> dict:
    """使用 async_request_template，间接使用全局 session"""
    assert per_page <= 200, "Per page is at most 200"
    # 整理参数
    url = f"https://api.openalex.org/{endpoint}"
    if filter:
        # filter
        if isinstance(filter, dict): filter = list(filter.items())
        filter_string = ",".join([f"{k}:{v}" for k, v in filter])
        request_kwargs["filter"] = filter_string
    if do_sample:
        # use per_page as num_samples
        request_kwargs['sample'] = per_page
        request_kwargs['seed'] = random.randint(0, 32767)        
    if add_email:
        request_kwargs['mailto'] = add_email if isinstance(add_email, str) else random.choice(EMAIL_POOL)
    if per_page > 25: 
        request_kwargs['per-page'] = per_page
    if select: request_kwargs['select'] = select
    # Go!
    async with RateLimit.OPENALEX_SEMAPHORE:
        results = await async_request_template("get", url, parameters=request_kwargs)
    
    if endpoint != "works": results = {"results": [results]}
    papers = []
    for x in results['results']:
        x['id'] = x['id'].replace(URL_DOMAIN, "")
        x['title'] = re.sub(r"\s+", " ", x['title'])
        if 'abstract_inverted_index' in x:
            x['abstract'] = index_to_abstract(x['abstract_inverted_index'])
            del x['abstract_inverted_index']
        if 'publication_date' in x and 'created_date' in x and not x['publication_date']:
            x['publication_date'] = x['created_date']
            del x['created_date']
        x['referenced_works'] = [y.replace(URL_DOMAIN, "") for y in x['referenced_works']]
        # x['referenced_works'] = [redirect[y] if y in redirect else y for y in x['referenced_works']]
        papers.append(x)
    return {"count": results['meta']['count'], "results": papers}


def strip_outer_parentheses(s: str) -> str:
    """
    递归去掉包住整个表达式的最外层括号
    """
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        valid = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and i != len(s) - 1:
                valid = False
                break
        if valid:
            s = s[1:-1].strip()
        else:
            break
    return s


def clean_term(term: str) -> str:
    term = term.replace('"', '').replace("'", '')
    term = re.sub(r"[^\w\s\-_]", " ", term)
    term = re.sub(r"\s+", " ", term).strip()
    return term


def split_top_level_and(query: str):
    parts = []
    depth = 0
    buffer = []

    tokens = re.split(r"(\bAND\b)", query, flags=re.IGNORECASE)

    for tok in tokens:
        if "(" in tok:
            depth += tok.count("(")
        if ")" in tok:
            depth -= tok.count(")")

        if tok.upper() == "AND" and depth == 0:
            parts.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(tok)

    if buffer:
        parts.append("".join(buffer).strip())

    return parts


def split_or(group: str):
    group = strip_outer_parentheses(group)
    terms = re.split(r"\bOR\b", group, flags=re.IGNORECASE)
    return [clean_term(t) for t in terms if clean_term(t)]


def to_openalex(query: str) -> str:
    query = strip_outer_parentheses(query)
    and_groups = split_top_level_and(query)

    blocks = []
    for group in and_groups:
        or_terms = split_or(group)
        if or_terms:
            blocks.append("|".join(or_terms))

    return ",default.search:".join(blocks)
    # return text.replace('default.search:', '', 1)
