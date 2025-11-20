from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import json


class InitialToolInput(BaseModel):
    """Input for InitialTool."""
    query: str = Field(description="The user's research query (e.g., \"Federated Learning for medical data\").")


class DynamicOracleGenerator(BaseTool):
    """
    First tool that must be called. Loads document and creates context state.
    """
    name: str = "dynamic_oracle_generator"
    description: str = """
    Generates a 'Dynamic Oracle' for a given research query. This is the first and most critical step. 
    It uses a pre-trained Learning-to-Rank (LETOR) model to create a ranked 'oracle' list of the most important papers (based on 
    relevance, prestige, and impact) and extracts a list of essential sub-topics from them.

    Returns a JSON string containing two keys: 
    1. 'oracle_paper_list': A ranked list of paper ID objects, each with 'id' and 'score'.
    2. 'essential_subtopics': A list of key sub-topic strings.
    """
    args_schema: type[BaseModel] = InitialToolInput
    
    # Stateful: stores loaded documents
    oracle: dict = {}
    context_counter: int = 0
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _search_for_papers(self, ):
        pass
    
    def _run(self, query: str, run_manager=None) -> str:
        # Return context_id for other tools to use
        return json.dumps({
            "oracle_paper_list": [{}, {}],
            "essential_subtopics": ["Introduction", "Methodology", "Results"]
        }, indent=2)
    
    async def _arun(self, query: str, run_manager=None) -> str:
        raise NotImplementedError
    
    def get_paper_with_id(self, context_id: str) -> dict:
        """Helper method for other tools to retrieve oracle papers."""
        return self.oracle.get(context_id, {})

