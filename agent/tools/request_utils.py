import json
import time
import asyncio, aiohttp
from typing import Optional

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36',
}


class AsyncRequestRateLimiter:
    def __init__(self, requests_per_second: float = 0.0, enabled: bool = True):
        self._lock = asyncio.Lock()
        self._next_allowed_time = 0.0
        self.configure(requests_per_second=requests_per_second, enabled=enabled)

    def configure(self, requests_per_second: float = 0.0, enabled: bool = True):
        self.enabled = bool(enabled) and requests_per_second is not None and float(requests_per_second) > 0
        self.requests_per_second = float(requests_per_second or 0.0)
        self.interval = 1.0 / self.requests_per_second if self.enabled else 0.0
        self._next_allowed_time = 0.0

    async def acquire(self):
        if not self.enabled:
            return
        async with self._lock:
            now = time.monotonic()
            scheduled_time = max(now, self._next_allowed_time)
            self._next_allowed_time = scheduled_time + self.interval
        wait_seconds = scheduled_time - now
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)


class OpenAlexBudgetExceeded(RuntimeError):
    def __init__(self, payload: dict | None = None):
        self.payload = payload or {}
        retry_after = self.payload.get("retryAfter")
        message = self.payload.get("message", "OpenAlex budget exceeded")
        if retry_after is not None:
            message = f"{message} (retryAfter={retry_after})"
        super().__init__(message)


# =============== Global Semaphore ===============
class RateLimit:
    OPENALEX_REQUEST_COUNT = 0
    OPENALEX_THROTTLER = AsyncRequestRateLimiter(5.0, enabled=True)
    OPENALEX_SEMAPHORE = asyncio.Semaphore(3)              # 搜索 API
    AGENT_SEMAPHORE = asyncio.Semaphore(100)                # LLM
    DOWNLOAD_SEMAPHORE = asyncio.Semaphore(4)
    SBERT_SEMAPHORE = asyncio.Semaphore(20)                 # LLM
    PARSE_SEMAPHORE = asyncio.Semaphore(4)                 # GROBID docker镜像本地解析
    WEBSEARCH_SEMAPHORE = asyncio.Semaphore(50)

    @classmethod
    def configure_openalex(cls, requests_per_second: float = 5.0, enabled: bool = True, max_concurrency: int = 3):
        cls.OPENALEX_SEMAPHORE = asyncio.Semaphore(max(1, int(max_concurrency)))
        cls.OPENALEX_THROTTLER.configure(requests_per_second=requests_per_second, enabled=enabled)

    @classmethod
    async def wait_openalex_slot(cls):
        await cls.OPENALEX_THROTTLER.acquire()

    @classmethod
    def increment_openalex_count(cls):
        cls.OPENALEX_REQUEST_COUNT += 1

    @classmethod
    def reset_openalex_count(cls):
        cls.OPENALEX_REQUEST_COUNT = 0

    @classmethod
    def get_openalex_count(cls) -> int:
        return cls.OPENALEX_REQUEST_COUNT


class SessionManager:
    _global_session: Optional[aiohttp.ClientSession] = None
    
    @classmethod
    async def init(cls):
        """进入上下文时调用"""
        if cls._global_session is None:
            connector = aiohttp.TCPConnector(limit=200, limit_per_host=100, ttl_dns_cache=300)
            cls._global_session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=600)
            )
    
    @classmethod
    async def close(cls):
        """退出上下文时调用"""
        if cls._global_session and not cls._global_session.closed:
            await cls._global_session.close()
            cls._global_session = None
    
    @classmethod
    def get(cls) -> aiohttp.ClientSession:
        """获取全局 session"""
        if cls._global_session is None:
            raise RuntimeError("SessionManager not initialized")
        return cls._global_session


async def async_request_template(
    method: str,
    url: str,
    headers: dict = HEADERS,
    parameters: dict = None,
    return_json: bool = True,
    timeout: int = 30
) -> dict:
    """使用全局 session"""
    session = SessionManager.get()
    
    headers = headers or {}
    parameters = parameters or {}

    async def _handle_error_response(resp: aiohttp.ClientResponse):
        text = await resp.text()
        try:
            payload = json.loads(text)
        except Exception:
            payload = {"raw_text": text}
        if payload.get("error") == "Rate limit exceeded":
            raise OpenAlexBudgetExceeded(payload)
        resp.raise_for_status()
    
    if method.lower() == "post":
        headers.setdefault("Content-Type", "application/json")
        async with session.post(url, headers=headers, json=parameters, timeout=timeout) as resp:
            if resp.status >= 400:
                await _handle_error_response(resp)
            return await resp.json()
    else:
        async with session.get(url, headers=headers, params=parameters, timeout=timeout) as resp:
            if resp.status >= 400:
                await _handle_error_response(resp)
            return await resp.json()
