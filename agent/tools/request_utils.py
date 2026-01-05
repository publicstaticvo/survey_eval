import json
import asyncio, aiohttp
from typing import Optional

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}
GROBID_URL = "https://localhost:8070"


# =============== Global Semaphore ===============
class RateLimit:
    OPENALEX_SEMAPHORE = asyncio.Semaphore(5)              # 搜索 API
    AGENT_SEMAPHORE = asyncio.Semaphore(100)                # LLM
    HTTP_SEMAPHORE = asyncio.Semaphore(10)
    SBERT_SEMAPHORE = asyncio.Semaphore(20)                 # LLM
    PARSE_SEMAPHORE = asyncio.Semaphore(10)                 # GROBID docker镜像本地解析


class SessionManager:
    _global_session: Optional[aiohttp.ClientSession] = None
    
    @classmethod
    async def init(cls):
        """进入上下文时调用"""
        if cls._global_session is None:
            connector = aiohttp.TCPConnector(limit=200, limit_per_host=100, ttl_dns_cache=300)
            cls._global_session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=60)
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
    parameters: dict = None
) -> dict:
    """使用全局 session"""
    session = SessionManager.get()
    
    headers = headers or {}
    parameters = parameters or {}
    
    if method.lower() == "post":
        headers.setdefault("Content-Type", "application/json")
        async with session.post(url, headers=headers, json=parameters) as resp:
            resp.raise_for_status()
            return await resp.json()
    else:
        async with session.get(url, headers=headers, params=parameters) as resp:
            resp.raise_for_status()
            return await resp.json()
