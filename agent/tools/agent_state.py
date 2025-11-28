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
    paper_content_map: Dict[str, str]    # From Retriever (Map: ID -> Text)

    # REQUIRED for standard tool calling:
    messages: Annotated[List[BaseMessage], operator.add]

    # Critic Outputs (Reducers allow parallel writing)
    # Each critic appends its result to these lists
    fact_checks: Annotated[List[Dict], operator.add] 
    source_evals: Dict[str, Any]
    topic_evals: Dict[str, Any]
    quality_evals: Dict[str, Any]
    summary_fact_check_metrics: Dict[str, Any]
