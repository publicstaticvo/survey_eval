import operator
from langchain_core.messages import BaseMessage
from typing import Annotated, List, Dict, Any
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    # Inputs
    query: str
    review_paper: Dict[str, Any]
    
    # Shared Resources (Written once)
    """
    What's in paper_content_map
    Key: paper_citation_key
    Value:
        - metadatas: List of metadata found in openAlex
        - status: 0-3
        - title: title
        - abstract: abstract
        - full_content: full_content (Dict[str, str]) / abstract (str) / title (str)
    """ 
    paper_content_map: Dict[str, str] = Field(default_factory=dict)     # From Retriever (Map: ID -> Text)
    oracle_data: Dict[str, Any] = Field(default_factory=dict)           # From Oracle
    claims: List[Dict] = Field(default_factory=list)                    # From Segmentation

    # REQUIRED for standard tool calling:
    # messages: Annotated[List[BaseMessage], operator.add]
    errors: Annotated[Dict[str, Any], lambda l, r: l | r if r else l]

    # Critic Outputs (Reducers allow parallel writing)
    # Each critic appends its result to these lists
    invalid_claims: int = 0
    fact_checks: Annotated[List[Dict], operator.add] 
    source_evals: Dict[str, Any] = Field(default_factory=dict)
    topic_evals: Dict[str, Any] = Field(default_factory=dict)
    quality_evals: Dict[str, Any] = Field(default_factory=dict)
    summary_fact_check_metrics: Dict[str, Any] = Field(default_factory=dict)
