import asyncio
import logging
import operator
from typing import Annotated, TypedDict, List, Dict, Any

from pydantic import BaseModel, Field
from langgraph.types import Send
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel

from tools import (
    ClaimSegmentation,
    DynamicOracleGenerator,
    FactualCorrectnessCritic,
    CitationParser,
    ClarityCritic,
    ProgrammaticReadabilityCritic,
    ProgrammaticRedundancyCritic,
    SourceSelectionCritic,
    SynthesisCorrectnessCritic,
    TopicCoverageCritic,
    SentenceTransformerClient,
    ToolConfig,
    FactualLLMClient,
    QualityLLMClient,
    SubtopicLLMClient,
    SynthesisLLMClient,
    ClaimSegmentationLLMClient,
)


# --- The State ---
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
    
    # Final Report
    final_report: str


class AgentNode:

    def __init__(self, llm: BaseChatModel, tools: List[BaseTool]):
        # Initialize classes (assuming they are wrapped as simple async functions or classes)
        self.tools = tools
        self.main_llm = llm.bind_tools(tools)

    async def __call__(self, state: AgentState):
        """
        The 'Brain' node.
        """
        logging.info("--- Judge Agent Reasoning ---")

        # 4. Invoke. The LLM will return an AIMessage.
        # If it wants to call a tool, AIMessage.tool_calls will be populated.
        # 光有action不够，还要有input。
        prompt = f"""
        You are a Literature Review Judge. 
        Current Data Status:
        - Oracle Data: {"READY" if state.get('dynamic_oracle_data') else "MISSING"}
        - Parsed Papers: {len(state.get('paper_content_map', {}))} papers
        - Facts Checked: {len(state.get('fact_checks', []))} claims
        
        Task:
        1. If Oracle/Parsing is missing, you MUST call 'run_parallel_data_prep'.
        2. Analyze the 'claims'. If you see suspicious claims, call 'factual_correctness_critic'.
        3. If the writing seems repetitive, call 'programmatic_redundancy_critic'.
        4. If you have sufficient evidence to score the paper, respond with FINAL_REPORT.
        """
        response = await self.main_llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Current State Query: {state.query}"}
        ])
        
        return {"messages": [response]}


# --- 1. The "Super Node" for Synchronization ---
class ParallelDataPreperation:
    """
    Runs Oracle, Parser, and Segmentation in parallel.
    This acts as a synchronization barrier: We don't move to critics
    until ALL data (Oracle list, Parsed PDFs, Segmented Claims) is ready.
    """

    def __init__(self, config: ToolConfig):
        # Initialize classes (assuming they are wrapped as simple async functions or classes)
        sbert = SentenceTransformerClient(config.sbert_server_url)
        self.oracle_tool = DynamicOracleGenerator(
            num_oracle_papers=config.num_oracle_papers, 
            llm_model=SubtopicLLMClient(config.llm_server_info, config.sampling_params, config.llm_num_workers),
            sentence_transformer=sbert,
            eval_date=config.evaluation_date,
        )
        self.parser_tool = CitationParser(config.grobid_url, config.grobid_num_workers)
        self.segment_tool = ClaimSegmentation(
            llm=ClaimSegmentationLLMClient(config.llm_server_info, config.sampling_params, config.llm_num_workers)
        )

    async def __call__(self, state: AgentState):
        logging.info("--- Starting Parallel Data Prep ---")
        # 1. Create Async Tasks
        # Note: segment_tool only needs the review_text, not the parsed papers!
        task_oracle = self.oracle_tool.ainvoke(state.query)
        task_parse = self.parser_tool.ainvoke(state.review_paper['citations']) 
        task_segment = self.segment_tool.ainvoke(state.review_paper['content'])

        # 2. Run them all at once
        oracle_res, parse_res, segment_res = await asyncio.gather(task_oracle, task_parse, task_segment)

        # 3. Return combined state updates
        return {
            "oracle_data": oracle_res,
            "paper_content_map": parse_res,
            "claims": segment_res,
        }


# --- 2. Document-Level Critics (Standard Nodes) ---

class RunSourceCritic:

    def __init__(self, config: ToolConfig):
        # Initialize classes (assuming they are wrapped as simple async functions or classes)
        self.tool = SourceSelectionCritic(
            sentence_transformer=SentenceTransformerClient(config.sbert_server_url),
            letor_model=...,
            eval_date=config.evaluation_date,
            topn=config.topn
        )

    async def __call__(self, state: AgentState):
        score = await self.tool.ainvoke(state.review_paper['citations'], state.query, state.oracle_data)
        return {"source_evals": score}


class RunTopicCritic:

    def __init__(self, config: ToolConfig):
        # Initialize classes (assuming they are wrapped as simple async functions or classes)
        self.tool = TopicCoverageCritic(
            sentence_transformer=SentenceTransformerClient(config.sbert_server_url),
            threshold=config.topic_similarity_threshold,
        )

    async def __call__(self, state: AgentState):
        score = await self.tool.ainvoke(state.review_paper['content'], state.oracle_data)
        return {"topic_evals": score}


class QualityEvaluation:
    """
    Runs Clarity, Readability, Redundancy in one node
    """

    def __init__(self, config: ToolConfig):
        # Initialize classes (assuming they are wrapped as simple async functions or classes)
        self.redundancy_tool = ProgrammaticRedundancyCritic(
            sentence_transformer=SentenceTransformerClient(config.sbert_server_url), 
            threshold=config.redundancy_similarity_threshold, 
            n_gram=config.redundancy_ngram,
        )
        self.readability_tool = ProgrammaticReadabilityCritic()
        llm = QualityLLMClient(config.llm_server_info, config.sampling_params, config.llm_num_workers, True)
        self.clarity_tool = ClarityCritic(llm)

    async def __call__(self, state: AgentState):
        logging.info("--- Starting Parallel Data Prep ---")
        # 1. Create Async Tasks
        # Note: segment_tool only needs the review_text, not the parsed papers!
        task_redundancy = self.redundancy_tool.ainvoke(state['query'])
        task_readability = self.readability_tool.ainvoke(state['citation_list']) 
        task_clarity = self.clarity_tool.ainvoke(state['review_text'])

        # 2. Run them all at once
        redundancy_res, readability_res, clarity_res = await asyncio.gather(
            task_redundancy, task_readability, task_clarity
        )

        # 3. Return combined state updates
        return {"quality_evals": {
            "redundancy_evals": redundancy_res,
            "readability_evals": readability_res,
            "clarity_evals": clarity_res,
        }}


# --- 3. Fact-Checking Critics (Map-Reduce Workers) ---

# Define a tiny state just for the worker payload
class FactCheckPayload(TypedDict):
    claim: str
    cited_paper: str


class SynthesisPayload(TypedDict):
    claim: str
    cited_papers: Dict[str, str]


class RunFactualCritic:

    def __init__(self, config: ToolConfig):
        # Initialize classes (assuming they are wrapped as simple async functions or classes)
        llm = FactualLLMClient(config.llm_server_info, config.sampling_params, config.llm_num_workers, True)
        self.tool = FactualCorrectnessCritic(llm)

    async def __call__(self, payload: FactCheckPayload):
        score = await self.tool.ainvoke(payload.claim, payload.cited_paper)
        return {"topic_evals": score}
    

class RunSynthesisCritic:

    def __init__(self, config: ToolConfig):
        # Initialize classes (assuming they are wrapped as simple async functions or classes)
        llm = SynthesisLLMClient(config.llm_server_info, config.sampling_params, config.llm_num_workers, True)
        self.tool = SynthesisCorrectnessCritic(llm)

    async def __call__(self, payload: SynthesisPayload):
        score = await self.tool.ainvoke(payload.claim, payload.cited_papers)
        return {"topic_evals": score}


def map_claims_to_critics(state: AgentState):
    """
    Generator function that creates Send objects.
    It inspects the 'claims' list and spawns the correct worker for each.
    """
    tasks = []
    content_map = state.paper_content_map
    
    for claim in state.claims:
        c_type = claim['type']
        
        if c_type == "SINGLE_CLAIM":
            # Spawn 1 Factual Critic
            cid = claim['citations'][0]
            tasks.append(Send("factual_critic", {
                "claim": claim['claim_text'],
                "cited_paper": content_map[cid]
            }))
            
        elif c_type == "SYNTHESIS_CLAIM":
            # Spawn 1 Synthesis Critic
            cids = claim['citations']
            tasks.append(Send("synthesis_critic", {
                "claim": claim['text'],
                "cited_paper": {cid: content_map[cid] for cid in cids}
            }))
            
        elif c_type == "SERIAL_CLAIMS":
            # EXPLODE: Spawn multiple Factual Critics
            for sub in claim['sub_claims']:
                cid = sub['citation']
                tasks.append(Send("factual_critic", {
                    "claim_text": sub['text'],
                    "paper_text": content_map[cid]
                }))
                
    return tasks


class FinalAggregation:

    def __init__(self, llm: BaseChatModel):

        class FinalReport(BaseModel):
            final_score: int
            decision: str
            summary: str

        self.structured_llm = llm.with_structured_output(FinalReport)


    async def run_final_aggregation(self, state: AgentState):
        # Use .with_structured_output to force Pydantic compliance
        # This guarantees your final report is valid JSON, not markdown text.
        
        # ... construct evidence string ...
        prompt = ""
        
        result = await self.structured_llm.ainvoke(prompt)
        
        # Result is now a FinalReport object, not a string
        return {"final_report": result.model_dump()}
    

def tools_condition(state: AgentState):
    pass


def build_agent(config: ToolConfig):

    llm = ChatOpenAI(
        model=config.agent_info.model,
        openai_api_base=config.agent_info.base_url,
        openai_api_key=config.agent_info.api_key,
        temperature=0,
        max_tokens=config.agent_max_tokens,
    )
    agent = AgentNode(llm, tools_to_bind)

    run_parallel_data_prep = ParallelDataPreperation(config)
    run_quality_group = QualityEvaluation(config)
    run_source_critic = RunSourceCritic(config)
    run_topic_critic = RunTopicCritic(config)
    run_factual_critic = RunFactualCritic(config)
    run_synthesis_critic = RunSynthesisCritic(config)
    run_final_aggregation = FinalAggregation(llm)

    tools_to_bind = [
        run_parallel_data_prep, 
        run_quality_group, 
        run_source_critic,
        run_topic_critic,
        run_factual_critic,
        run_synthesis_critic,
        run_final_aggregation
    ]
    
    tools = ToolNode(tools_to_bind)

    workflow = StateGraph(AgentState)
    workflow.add_node("judge", agent)
    workflow.add_node("tools", tools)
    workflow.add_conditional_edges("judge", tools_condition, {"tools": "tools", "__end__": END})
    workflow.add_edge("tools", "judge")

    # # Add Nodes
    # workflow.add_node("data_prep", run_parallel_data_prep)       # The Sync Node
    # workflow.add_node("quality_critics", run_quality_group)      # Independent
    # workflow.add_node("source_critic", run_source_critic)
    # workflow.add_node("topic_critic", run_topic_critic)
    # workflow.add_node("factual_critic", run_factual_critic)      # Worker
    # workflow.add_node("synthesis_critic", run_synthesis_critic)  # Worker
    # workflow.add_node("aggregator", run_final_aggregation)

    # # --- EDGES ---

    # # Branch 1: Independent Quality Check
    # workflow.add_edge(START, "quality_critics")
    # workflow.add_edge("quality_critics", "aggregator")

    # # Branch 2: The Main Pipeline
    # workflow.add_edge(START, "data_prep")

    # # Fan-Out from Data Prep (Now safe because all data is ready)
    # workflow.add_edge("data_prep", "source_critic")
    # workflow.add_edge("data_prep", "topic_critic")

    # # Retrieval -> Map-Reduce for Facts
    # # This is the Conditional Edge that spawns workers
    # workflow.add_conditional_edges("data_prep", map_claims_to_critics, ["factual_critic", "synthesis_critic"])

    # # Fan-In: All critics -> Aggregator
    # workflow.add_edge("source_critic", "aggregator")
    # workflow.add_edge("topic_critic", "aggregator")
    # workflow.add_edge("quality_critics", "aggregator")
    # workflow.add_edge("factual_critic", "aggregator")
    # workflow.add_edge("synthesis_critic", "aggregator")

    # workflow.add_edge("aggregator", END)

    # Compile
    app = workflow.compile()
    return app
