
from .agent_state import AgentState
from .llmclient import AsyncLLMClient
from .prompts import FINAL_AGGREGATION_PROMPT


class FinalAggregate(AsyncLLMClient):

    prompt: str = FINAL_AGGREGATION_PROMPT

    def __init__(self, llm, sampling_params, config):
        super().__init__(llm, sampling_params)
        self.config = config

    def _availability(self, response):
        return super()._availability(response)
    
    def _organize_inputs(self, inputs):
        return super()._organize_inputs(inputs)

    async def __call__(self, state: AgentState):
        return
