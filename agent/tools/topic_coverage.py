import json
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from dynamic_oracle_generator import DynamicOracleGenerator


class InitialToolInput(BaseModel):
    """Input for InitialTool."""
    review_text: str = Field(description="The full text of the generated literature review.")
    dynamic_oracle_data: str = Field(description="The JSON string output from the 'DynamicOracleGenerator' tool.")


class TopicCoverageCritic(BaseTool):
    """
    First tool that must be called. Loads document and creates context state.
    """
    name: str = "topic_coverage_critic"
    description: str = """
    Evaluates the topical coverage of the *written review*. It checks if the semantic 
    content of the review text successfully covers the 'essential_subtopics' provided 
    by the 'DynamicOracleGenerator'.

    Returns a JSON string containing a 'coverage_score' (a float 0-1 representing the 
    percentage of essential topics covered) and 'missing_topics' (a list of strings).
    """
    args_schema: type[BaseModel] = InitialToolInput
    oracle_generator: DynamicOracleGenerator
    
    def __init__(self, oracle, **kwargs):
        super().__init__(**kwargs)
        self.oracle_generator = oracle
        self.model = None
    
    def _run(self, review_text: str, dynamic_oracle_data: str, run_manager=None) -> str:
        # Return context_id for other tools to use    
        dynamic_oracle_data = json.loads(dynamic_oracle_data)
        return json.dumps({
            "coverage_score": 0,
            "missing_topics": ["Introduction", "Methodology", "Results"]
        }, indent=2)
    
    async def _arun(self, review_text: str, dynamic_oracle_data: str, run_manager=None) -> str:
        return self._run(review_text, dynamic_oracle_data, run_manager)
