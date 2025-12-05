import operator
from langchain_core.messages import BaseMessage
from typing import Annotated, TypedDict, List, Dict, Any


class AgentState(TypedDict):
    # Inputs
    query: str
    review_paper: Dict[str, Any]
    
    # Shared Resources (Written once)
    oracle_data: Dict[str, Any]  # From Oracle
    claims: List[Dict]                   # From Segmentation
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
    paper_content_map: Dict[str, str]    # From Retriever (Map: ID -> Text)

    # REQUIRED for standard tool calling:
    messages: Annotated[List[BaseMessage], operator.add]

    # Critic Outputs (Reducers allow parallel writing)
    # Each critic appends its result to these lists
    invalid_claims: int
    fact_checks: Annotated[List[Dict], operator.add] 
    source_evals: Dict[str, Any]
    topic_evals: Dict[str, Any]
    quality_evals: Dict[str, Any]
    summary_fact_check_metrics: Dict[str, Any]


class InputState(TypedDict):
    query: str
    review_paper: Dict[str, Any]
