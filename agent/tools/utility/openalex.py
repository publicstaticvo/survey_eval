from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import aiohttp
import Levenshtein

from .request_utils import HEADERS, OpenAlexBudgetExceeded, RateLimit, SessionManager
from .tool_config import ToolConfig
from .utils import normalize_text, valid_check


OPENALEX_SELECT = "id,cited_by_count,counts_by_year,referenced_works,publication_date,created_date,abstract_inverted_index,title,authorships"
URL_DOMAIN = "https://openalex.org/"
OPENALEX_API_URL = "https://api.openalex.org"
OPENALEX_CONTENT_URL = "https://contents.openalex.org"
FREE_CREDITS_PER_DAY = 10000

TRANSIENT_EXCEPTION_TYPES = (
    aiohttp.ClientError,
    asyncio.TimeoutError,
    aiohttp.ServerDisconnectedError,
    json.JSONDecodeError,
    AssertionError,
    KeyError,
)


def index_to_abstract(indexes: dict | None):
    if not indexes:
        return None
    abstract_length = max(v[-1] for v in indexes.values())
    abstract = ["<mask>" for _ in range(abstract_length + 1)]
    for token, positions in indexes.items():
        for i in positions:
            abstract[i] = token
    return " ".join(abstract)


@dataclass
class CredentialState:
    label: str
    api_key: str | None
    credits_remaining: int
    cooling_until: datetime | None = None
    initialized: bool = False

    @property
    def requires_api_key(self) -> bool:
        return self.api_key is not None

    def is_available(self, estimated_cost: int = 0) -> bool:
        if self.cooling_until is not None and datetime.now(UTC) < self.cooling_until:
            return False
        return self.credits_remaining >= max(0, estimated_cost)


class OpenAlex:
    def __init__(self, config: ToolConfig):
        self.config = config
        self._init_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._initialized = False
        self.request_count = 0
        self.no_key_state = CredentialState("anonymous", None, FREE_CREDITS_PER_DAY, initialized=True)
        self.api_key_states = [
            CredentialState(f"api_key_{idx}", api_key, 0, initialized=False)
            for idx, api_key in enumerate(config.openalex_api_keys or [], 1)
            if str(api_key).strip()
        ]

    async def ensure_ready(self):
        if self._initialized: return
        async with self._init_lock:
            if self._initialized: return
            for state in self.api_key_states:
                try:
                    state.credits_remaining = await self.get_balance(state.api_key)
                    state.initialized = True
                except Exception as e:
                    state.credits_remaining = 0
                    state.cooling_until = self._next_utc_midnight()
                    state.initialized = True
            self._initialized = True

    def reset_request_count(self):
        self.request_count = 0

    def get_request_count(self) -> int:
        return self.request_count

    def _next_utc_midnight(self) -> datetime:
        now = datetime.now(UTC)
        tomorrow = (now + timedelta(days=1)).date()
        return datetime.combine(tomorrow, datetime.min.time(), tzinfo=UTC)

    def _normalize_openalex_id(self, value: str) -> str:
        return (value or "").replace(URL_DOMAIN, "").strip()

    def _normalize_work(self, paper: dict) -> dict:
        paper = dict(paper or {})
        if not paper.get("title"):
            return {}
        if paper.get("id"):
            paper["id"] = self._normalize_openalex_id(paper["id"])
        paper["title"] = re.sub(r"\s+", " ", paper.get("title", ""))
        if "abstract_inverted_index" in paper:
            paper["abstract"] = index_to_abstract(paper["abstract_inverted_index"])
            del paper["abstract_inverted_index"]
        if paper.get("publication_date") is None and paper.get("created_date"):
            paper["publication_date"] = paper["created_date"]
        if "created_date" in paper:
            paper.pop("created_date", None)
        paper["referenced_works"] = [
            self._normalize_openalex_id(work_id)
            for work_id in paper.get("referenced_works", []) or []
        ]
        paper["authorships"] = self._normalize_authorships(paper.get("authorships", []))
        if paper.get("id") and not paper.get("ids"):
            paper["ids"] = [paper["id"]]
        return paper

    def _normalize_authorships(self, authorships: list[dict] | list[str] | None) -> list[str]:
        authors = []
        for item in authorships or []:
            if isinstance(item, str):
                name = item
            elif isinstance(item, dict):
                author = item.get("author") or {}
                name = author.get("display_name") or ""
            else:
                name = ""
            name = re.sub(r"\s+", " ", name or "").strip()
            if name: authors.append(name)
        return list(dict.fromkeys(authors))

    def _paper_year(self, paper: dict) -> int | None:
        publication_date = paper.get("publication_date") or ""
        if isinstance(publication_date, str) and len(publication_date) >= 4 and publication_date[:4].isdigit():
            return int(publication_date[:4])
        return None

    def _same_work(self, left: dict, right: dict) -> bool:
        left_title = normalize_text(left.get("title", ""))
        right_title = normalize_text(right.get("title", ""))
        if not left_title or not right_title:
            return False
        max_title_len = max(len(left_title), len(right_title), 1)
        if Levenshtein.distance(left_title, right_title) > max(1, int(0.1 * max_title_len)):
            return False

        left_authors = {normalize_text(author) for author in left.get("authorships", []) if normalize_text(author)}
        right_authors = {normalize_text(author) for author in right.get("authorships", []) if normalize_text(author)}
        if left_authors and right_authors:
            overlap = len(left_authors & right_authors) / max(1, min(len(left_authors), len(right_authors)))
            if overlap < 0.8:
                return False

        left_year = self._paper_year(left)
        right_year = self._paper_year(right)
        if left_year is not None and right_year is not None and abs(left_year - right_year) > 1:
            return False
        return True

    def _minhash_buckets(self, title: str, num_hashes: int = 24, band_size: int = 4) -> list[tuple[int, tuple[int, ...]]]:
        tokens = set(re.findall(r"[a-z0-9]+", normalize_text(title)))
        if not tokens:
            tokens = {normalize_text(title)}
        signature = []
        for seed in range(num_hashes):
            values = []
            for token in tokens:
                digest = hashlib.blake2b(f"{seed}:{token}".encode("utf-8"), digest_size=8).hexdigest()
                values.append(int(digest, 16))
            signature.append(min(values) if values else 0)
        return [
            (band_idx, tuple(signature[band_idx: band_idx + band_size]))
            for band_idx in range(0, num_hashes, band_size)
        ]

    def merge_duplicate_works(self, papers: list[dict], original_title: str = "") -> dict | None:
        merged = self.deduplicate_works(papers, original_title=original_title)
        if not merged:
            return None
        if len(merged) == 1:
            return merged[0]
        return self._merge_work_cluster(merged, original_title)

    def deduplicate_works(self, papers: list[dict], original_title: str = "") -> list[dict]:
        papers = [dict(paper) for paper in papers if paper and paper.get("title")]
        if not papers:
            return []

        parent = list(range(len(papers)))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(left: int, right: int):
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        buckets = {}
        for idx, paper in enumerate(papers):
            for bucket in self._minhash_buckets(paper.get("title", "")):
                buckets.setdefault(bucket, []).append(idx)
        for indexes in buckets.values():
            for pos, left in enumerate(indexes):
                for right in indexes[pos + 1:]:
                    if self._same_work(papers[left], papers[right]):
                        union(left, right)

        clusters = {}
        for idx in range(len(papers)):
            clusters.setdefault(find(idx), []).append(papers[idx])
        return [self._merge_work_cluster(cluster, original_title) for cluster in clusters.values()]

    def _merge_work_cluster(self, papers: list[dict], original_title: str = "") -> dict:
        base = max(papers, key=lambda paper: paper.get("cited_by_count", 0) or 0)
        merged = dict(base)
        ids = []
        referenced_works = set()
        authors = []
        locations = []
        seen_locations = set()
        for paper in papers:
            ids.extend(paper.get("ids") or ([paper["id"]] if paper.get("id") else []))
            referenced_works.update(paper.get("referenced_works", []) or [])
            authors.extend(paper.get("authorships", []) or [])
            for location in paper.get("locations", []) or []:
                key = json.dumps(location, sort_keys=True, ensure_ascii=False) if isinstance(location, dict) else str(location)
                if key not in seen_locations:
                    seen_locations.add(key)
                    locations.append(location)

        ids = list(dict.fromkeys(ids))
        merged["ids"] = ids
        if ids:
            merged["id"] = ids[0]
        merged["referenced_works"] = sorted(referenced_works)
        merged["authorships"] = list(dict.fromkeys(authors))
        if locations:
            merged["locations"] = locations

        dates = [paper.get("publication_date") for paper in papers if paper.get("publication_date")]
        if dates:
            merged["publication_date"] = min(dates)

        title_anchor = original_title or papers[0].get("title", "")
        merged["title"] = min(
            (paper.get("title", "") for paper in papers if paper.get("title")),
            key=lambda value: Levenshtein.distance(normalize_text(title_anchor), normalize_text(value)),
        )
        cited_source = max(papers, key=lambda paper: paper.get("cited_by_count", 0) or 0)
        merged["cited_by_count"] = cited_source.get("cited_by_count", 0) or 0
        merged["counts_by_year"] = cited_source.get("counts_by_year", []) or []
        return merged

    def _wrap_search_results(self, entity_type: str, payload: dict) -> dict:
        results = payload.get("results", []) if isinstance(payload, dict) else []
        if entity_type == "works":
            normalized = [paper for item in results if (paper := self._normalize_work(item))]
        else:
            normalized = results
        return {"count": payload.get("meta", {}).get("count", len(normalized)), "results": normalized}

    def _format_filter(self, filter_value: list[tuple] | dict | None) -> str | None:
        if not filter_value:
            return None
        if isinstance(filter_value, dict):
            filter_value = list(filter_value.items())
        return ",".join(f"{key}:{value}" for key, value in filter_value)

    def _filter_has_search_key(self, filter_value: list[tuple] | dict | None) -> bool:
        if not filter_value: return False
        items = filter_value.items() if isinstance(filter_value, dict) else filter_value
        return any(str(key).endswith(".search") for key, _ in items)

    def _estimate_search_cost(self, search: str, filter_value: list[tuple] | dict | None) -> int:
        return 10 if (search or self._filter_has_search_key(filter_value)) else 1

    async def _refresh_daily_state(self, state: CredentialState):
        now = datetime.now(UTC)
        if state.cooling_until is None or now < state.cooling_until:
            return
        if state.api_key is None:
            state.credits_remaining = FREE_CREDITS_PER_DAY
            state.cooling_until = None
            return
        try:
            state.credits_remaining = await self.get_balance(state.api_key)
            state.cooling_until = None if state.credits_remaining > 0 else self._next_utc_midnight()
        except Exception:
            state.credits_remaining = 0
            state.cooling_until = self._next_utc_midnight()

    async def _choose_credential(self, estimated_cost: int, require_api_key: bool = False) -> CredentialState:
        await self.ensure_ready()
        async with self._state_lock:
            if not require_api_key:
                await self._refresh_daily_state(self.no_key_state)
                if self.no_key_state.is_available(estimated_cost):
                    return self.no_key_state
            for state in self.api_key_states:
                await self._refresh_daily_state(state)
                if state.is_available(estimated_cost):
                    return state
        raise OpenAlexBudgetExceeded({"message": "No OpenAlex credential has remaining credits"})

    def _mark_exhausted(self, state: CredentialState, payload: dict | None = None):
        state.credits_remaining = 0
        state.cooling_until = self._next_utc_midnight()

    def _deduct_credits(self, state: CredentialState, credits: int):
        if credits <= 0:
            return
        state.credits_remaining = max(0, state.credits_remaining - credits)
        if state.credits_remaining == 0:
            state.cooling_until = self._next_utc_midnight()

    def _extract_cost_credits(self, payload: dict) -> int:
        meta = payload.get("meta") or {}
        cost_usd = meta.get("cost_usd")
        if cost_usd is None:
            return 0
        return max(0, int(round(float(cost_usd) * 10000)))

    def _is_transient_error(self, exc: BaseException) -> bool:
        if isinstance(exc, OpenAlexBudgetExceeded):
            return False
        if isinstance(exc, aiohttp.ClientResponseError):
            return exc.status not in {400, 401, 403, 404}
        return isinstance(exc, TRANSIENT_EXCEPTION_TYPES)

    async def _single_json_request(self, url: str, params: dict[str, Any], credential: CredentialState) -> dict:
        session = SessionManager.get()
        last_exc = None
        for attempt in range(3):
            try:
                await RateLimit.wait_openalex_slot()
                self.request_count += 1
                async with RateLimit.OPENALEX_SEMAPHORE:
                    async with session.get(
                        url,
                        headers=HEADERS,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        text = await resp.text()
                        payload = json.loads(text)
                        if resp.status >= 400:
                            if payload.get("error") == "Rate limit exceeded":
                                raise OpenAlexBudgetExceeded(payload)
                            resp.raise_for_status()
                        return payload
            except Exception as exc:
                last_exc = exc
                if not self._is_transient_error(exc) or attempt == 2:
                    raise
                await asyncio.sleep(min(10, 2 ** attempt))
        raise last_exc

    async def _single_bytes_request(self, url: str, params: dict[str, Any], credential: CredentialState) -> bytes:
        session = SessionManager.get()
        last_exc = None
        for attempt in range(3):
            try:
                await RateLimit.wait_openalex_slot()
                self.request_count += 1
                async with RateLimit.OPENALEX_SEMAPHORE:
                    async with session.get(
                        url,
                        headers=HEADERS,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as resp:
                        content = await resp.read()
                        if resp.status >= 400:
                            try:
                                payload = json.loads(content.decode("utf-8"))
                            except Exception:
                                payload = {"raw_text": content.decode("utf-8", errors="ignore")}
                            if payload.get("error") == "Rate limit exceeded":
                                raise OpenAlexBudgetExceeded(payload)
                            resp.raise_for_status()
                        return content
            except Exception as exc:
                last_exc = exc
                if not self._is_transient_error(exc) or attempt == 2:
                    raise
                await asyncio.sleep(min(10, 2 ** attempt))
        raise last_exc

    async def _request_json(
        self,
        url: str,
        params: dict[str, Any],
        estimated_cost: int = 0,
        fixed_cost: int | None = None,
        require_api_key: bool = False,
    ) -> tuple[dict, CredentialState]:
        while True:
            credential = await self._choose_credential(estimated_cost, require_api_key=require_api_key)
            request_params = dict(params)
            if credential.api_key is not None:
                request_params["api_key"] = credential.api_key
            try:
                payload = await self._single_json_request(url, request_params, credential)
            except OpenAlexBudgetExceeded as exc:
                async with self._state_lock:
                    self._mark_exhausted(credential, exc.payload)
                continue
            credits = fixed_cost if fixed_cost is not None else self._extract_cost_credits(payload)
            async with self._state_lock:
                self._deduct_credits(credential, credits)
            return payload, credential

    async def _request_bytes(
        self,
        url: str,
        params: dict[str, Any],
        fixed_cost: int,
        require_api_key: bool = True,
    ) -> tuple[bytes, CredentialState]:
        while True:
            credential = await self._choose_credential(fixed_cost, require_api_key=require_api_key)
            request_params = dict(params)
            if credential.api_key is not None:
                request_params["api_key"] = credential.api_key
            try:
                payload = await self._single_bytes_request(url, request_params, credential)
            except OpenAlexBudgetExceeded as exc:
                async with self._state_lock:
                    self._mark_exhausted(credential, exc.payload)
                continue
            async with self._state_lock:
                self._deduct_credits(credential, fixed_cost)
            return payload, credential

    async def get_balance(self, api_key: str) -> int:
        session = SessionManager.get()
        params = {"api_key": api_key}
        last_exc = None
        for attempt in range(3):
            try:
                await RateLimit.wait_openalex_slot()
                self.request_count += 1
                async with RateLimit.OPENALEX_SEMAPHORE:
                    async with session.get(
                        f"{OPENALEX_API_URL}/rate-limit",
                        headers=HEADERS,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        text = await resp.text()
                        payload = json.loads(text)
                        if resp.status >= 400:
                            if payload.get("error") == "Rate limit exceeded":
                                raise OpenAlexBudgetExceeded(payload)
                            resp.raise_for_status()
                        return int((payload.get("rate_limit") or {}).get("credits_remaining", 0))
            except Exception as exc:
                last_exc = exc
                if not self._is_transient_error(exc) or attempt == 2:
                    raise
                await asyncio.sleep(min(10, 2 ** attempt))
        raise last_exc

    async def get_entity(
        self,
        entity_id: str,
        entity_type: str = "works",
        select: str | None = OPENALEX_SELECT,
        **request_kwargs,
    ) -> dict:
        params = dict(request_kwargs)
        if select:
            params["select"] = select
        payload, _ = await self._request_json(
            f"{OPENALEX_API_URL}/{entity_type}/{entity_id}",
            params,
            estimated_cost=0,
            fixed_cost=0,
            require_api_key=False,
        )
        return self._normalize_work(payload) if entity_type == "works" else payload

    async def search_works(
        self,
        entity_type: str = "works",
        search: str = "",
        filter: list[tuple] | dict | None = None,
        do_sample: bool = False,
        per_page: int = 1,
        select: str | None = OPENALEX_SELECT,
        **request_kwargs,
    ) -> dict:
        assert per_page <= 200, "Per page is at most 200"
        params = dict(request_kwargs)
        filter_string = self._format_filter(filter)
        if filter_string: params["filter"] = filter_string
        if search: params["search"] = search
        if do_sample:
            params["sample"] = per_page
            params["seed"] = random.randint(0, 32767)
        elif per_page: params["per-page"] = per_page
        if select: params["select"] = select
        estimated_cost = self._estimate_search_cost(search, filter)
        payload, _ = await self._request_json(
            f"{OPENALEX_API_URL}/{entity_type}",
            params,
            estimated_cost=estimated_cost,
            fixed_cost=None,
            require_api_key=False,
        )
        return self._wrap_search_results(entity_type, payload)

    async def autocomplete(self, entity_type: str = "works", title: str = "") -> dict:
        payload, _ = await self._request_json(
            f"{OPENALEX_API_URL}/autocomplete/{entity_type}",
            {"q": title},
            estimated_cost=1,
            fixed_cost=1,
            require_api_key=False,
        )
        results = payload.get("results", []) or []
        for item in results:
            if item.get("id"):
                item["id"] = self._normalize_openalex_id(item["id"])
        return {"count": len(results), "results": results}

    async def find_work_by_title(self, title: str, select: str | None = OPENALEX_SELECT) -> dict | None:
        results = await self.autocomplete("works", title)
        if not results.get("results"): return
        entity_ids = []
        for item in results.get("results", []):
            candidate_title = item.get("display_name") or item.get("title") or item.get("name") or ""
            if not valid_check(title, candidate_title): continue
            entity_id = self._normalize_openalex_id(item.get("id", ""))
            if entity_id: entity_ids.append(entity_id)
        if not entity_ids:
            return

        async def _single(entity_id: str):
            try:
                return await self.get_entity(entity_id, entity_type="works", select=select)
            except Exception:
                return None

        tasks = [asyncio.create_task(_single(entity_id)) for entity_id in dict.fromkeys(entity_ids)]
        papers = [paper for paper in await asyncio.gather(*tasks) if paper]
        return self.merge_duplicate_works(papers, original_title=title)

    async def download_work_content(
        self,
        work_id: str,
        download_type: str = "grobid_xml",
        output_target: str = "",
    ) -> str:
        from .paper_download import parse_with_grobid

        payload, _ = await self._request_bytes(
            f"{OPENALEX_CONTENT_URL}/{work_id}.{download_type}",
            {},
            fixed_cost=100,
            require_api_key=True,
        )
        if download_type == "pdf":
            xml_content = await parse_with_grobid(self.config.grobid_url, io.BytesIO(payload))
            if not xml_content:
                raise RuntimeError(f"OpenAlex content download for {work_id} returned no XML")
            if output_target:
                with open(output_target, "w", encoding="utf-8") as f:
                    f.write(xml_content)
            return xml_content

        xml_content = gzip.decompress(payload).decode("utf-8")
        if output_target:
            with open(output_target, "w", encoding="utf-8") as f:
                f.write(xml_content)
        return xml_content


_OPENALEX_CLIENT: OpenAlex | None = None


def get_openalex_client(config: ToolConfig | None = None) -> OpenAlex:
    global _OPENALEX_CLIENT
    if _OPENALEX_CLIENT is None:
        _OPENALEX_CLIENT = OpenAlex(config or ToolConfig())
    elif config is not None:
        current_keys = tuple(_OPENALEX_CLIENT.config.openalex_api_keys or [])
        next_keys = tuple(config.openalex_api_keys or [])
        if (
            current_keys != next_keys
            or _OPENALEX_CLIENT.config.openalex_requests_per_second != config.openalex_requests_per_second
            or _OPENALEX_CLIENT.config.openalex_max_concurrency != config.openalex_max_concurrency
            or _OPENALEX_CLIENT.config.grobid_url != config.grobid_url
        ):
            _OPENALEX_CLIENT = OpenAlex(config)
    return _OPENALEX_CLIENT


async def openalex_search_paper(
    endpoint: str,
    filter: list[tuple] | dict = {},
    do_sample: bool = False,
    per_page: int = 1,
    add_email: bool | str = True,
    select: str = OPENALEX_SELECT,
    **request_kwargs,
) -> dict:
    client = get_openalex_client()
    if "/" in endpoint:
        entity_type, entity_id = endpoint.split("/", 1)
        payload = await client.get_entity(entity_id, entity_type=entity_type, select=select, **request_kwargs)
        return {"results": [payload]}
    return await client.search_works(
        entity_type=endpoint,
        filter=filter,
        do_sample=do_sample,
        per_page=per_page,
        select=select,
        **request_kwargs,
    )


def strip_outer_parentheses(s: str) -> str:
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
    term = term.replace('"', "").replace("'", "")
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
    return [clean_term(term) for term in terms if clean_term(term)]


def to_openalex(query: str) -> list[str]:
    query = strip_outer_parentheses(query)
    and_groups = split_top_level_and(query)
    blocks = []
    for group in and_groups:
        or_terms = split_or(group)
        if or_terms:
            blocks.append("|".join(or_terms))
    return blocks
