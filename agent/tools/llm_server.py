from concurrent.futures import ThreadPoolExecutor as TPE

from tool_config import LLMServerInfo
from utils import callLLM


class ConcurrentLLMClient:

    def __init__(
            self, 
            llm: LLMServerInfo, 
            sampling_params: dict, 
            n_workers: int, 
            retry: int = 5,
            return_reasoning: bool = False, 
        ):
        self.llm = llm
        self.sampling_params = sampling_params
        self.n_workers = n_workers
        self.return_reasoning = return_reasoning
        self.retry = retry

    def run_llm(self, message: list | str):
        return callLLM(self.llm, message, self.sampling_params, self.retry)
    
    def run_parallel(self, messages):
        with TPE(max_workers=self.n_workers) as executor:
            return list(executor.map(self.run_llm, messages))  
