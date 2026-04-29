import jsonschema
from .prompts import FINAL_AGGREGATION_PROMPT, FINAL_AGGREGATION_SCHEMA
from .utility.agent_state import AgentState
from .utility.llmclient import AsyncChat
from .eval.utils import extract_json


class FinalAggregate(AsyncChat):

    prompt: str = FINAL_AGGREGATION_PROMPT

    def __init__(self, llm, sampling_params, config):
        super().__init__(llm, sampling_params)
        self.config = config

    def _availability(self, response):
        json = extract_json(response)
        jsonschema.validate(json, FINAL_AGGREGATION_SCHEMA)
        return json
    
    def _organize_inputs(self, inputs):
        return super()._organize_inputs(inputs)

    async def __call__(self, state: AgentState):
        """
        All results in state
        - 
        """
        # Fact Check
        supported, refuted, neutral = 0, [], {}
        for x in state.fact_checks:
            if x['judgment'] == 'refuted': refuted.append(x)
            elif x['judgment'] == 'neutral':
                value = neutral.get(x['reason'], [])
                value.append(x)
                neutral[x['reason']] = value
            else: supported += 1
        pass
