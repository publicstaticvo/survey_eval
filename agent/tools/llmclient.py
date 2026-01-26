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

    def __init__(self, llm: LLMServerInfo, sampling_params: dict = {}):
        self.llm = llm
        self.timeout = 600
        self.sampling_params = sampling_params
        
    @abstractmethod
    def _availability(self, response, context):
        raise NotImplementedError
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=1, max=10),
        retry=retry_if_exception(llm_should_retry) | retry_if_result(lambda x: not x)
    )
    async def _post(self, endpoint: str = "chat/completions", payload: dict = None, context: dict = None):
        try:
            url = f"{self.llm.base_url.rstrip('/')}/v1/{endpoint}"
            headers = {"Authorization": f"Bearer {self.llm.api_key}"} | HEADERS
            async with RateLimit.AGENT_SEMAPHORE:
                data = await async_request_template("post", url, headers, payload, self.timeout)            
            if endpoint == "chat/completions":
                data = data["choices"][0]["message"]["content"]
            return self._availability(data, context)
        except Exception as e:
            print(f"LLM call error: {type(e)}")
            raise


class AsyncChat(AsyncLLMClient):

    PROMPT: str = ""
    
    def _organize_inputs(self, inputs: dict):
        return self.PROMPT.format(**inputs), {}
    
    async def call(self, inputs=None, messages=None, context=None, **kwargs):        
        assert inputs or messages, "Must have messages or inputs for chat/completions"        
        if messages is None:
            messages, new_context = self._organize_inputs(kwargs['inputs'])
        context = {**context, **new_context}
        if isinstance(messages, str):
            messages = [{'role': 'user', "content": messages}]
        payload = {"model": self.llm.model, "messages": messages, **self.sampling_params, **kwargs}
        return await self._post("chat/completions", payload, context)
