from abc import ABC, abstractmethod
from tenacity import (
    retry,
    stop_after_attempt,           # 最大重试次数
    wait_exponential,             # 指数退避
    retry_if_exception,           # 遇到什么异常才重试
    retry_if_result,              # 返回None的时候也要重试
)

from .tool_config import LLMServerInfo
from .request_utils import RateLimit, HEADERS, async_request_template


def llm_should_retry(exception: BaseException) -> bool:
    if isinstance(exception, KeyboardInterrupt): return False
    if isinstance(exception, NotImplementedError): return False
    return True


class AsyncLLMClient(ABC):

    PROMPT: str = ""

    def __init__(self, llm: LLMServerInfo, sampling_params: dict = {}):
        self.llm = llm
        self.timeout = 600
        self.sampling_params = sampling_params
        
    @abstractmethod
    def _availability(self, response):
        raise NotImplementedError
    
    def _organize_inputs(self, inputs: dict):
        return self.PROMPT.format(**inputs)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1.5, min=1, max=10),
        retry=retry_if_exception(llm_should_retry) | retry_if_result(lambda x: not x)
    )
    async def _call_llm(self, endpoint: str = "chat/completions", request_args: dict = {}):
        try:
            url = f"{self.llm.base_url.rstrip('/')}/v1/{endpoint}"
            headers = {"Authorization": f"Bearer {self.llm.api_key}"} | HEADERS
            async with RateLimit.AGENT_SEMAPHORE:
                data = await async_request_template("post", url, headers, request_args, self.timeout)
            
            if not data: 
                print("No return by LLM")
                return
            if endpoint == "chat/completions":
                data = data["choices"][0]["message"]["content"]

            return self._availability(data)
        except Exception as e:
            print(f"LLM call error: {e} {type(e)}")
            raise

    async def call(self, endpoint: str = "chat/completions", **kwargs) -> dict | None:
        self._context = kwargs.get("context", {})
        payload = {"model": self.llm.model}

        if endpoint == "chat/completions":
            payload.update(self.sampling_params)
            if "messages" not in kwargs:
                if "inputs" not in kwargs:
                    raise AttributeError("Must have messages or inputs for chat/completions")
                kwargs['messages'] = self._organize_inputs(kwargs['inputs'])
            messages = kwargs['messages']
            if isinstance(messages, str):
                messages = [{'role': 'user', "content": messages}]
            payload['messages'] = messages
        
        payload.update({k: v for k, v in kwargs.items() if k not in ["context", 'messages']})  
        return await self._call_llm(endpoint, payload)
