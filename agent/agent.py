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
from langgraph.prebuilt import tools_condition

from tools import (
    AgentState,
    ClaimSegmentation,
    DynamicOracleGenerator,
    FactualCorrectnessCritic,
    CitationParser,
    QualityCritic,
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
        response = await self.main_llm.ainvoke(state.messages)
        
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
        self.parser_tool = CitationParser(grobid_url=config.grobid_url, n_workers=config.grobid_num_workers)
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
        self.tool = QualityCritic(
            llm=QualityLLMClient(config.llm_server_info, config.sampling_params, config.llm_num_workers),
            sentence_transformer=SentenceTransformerClient(config.sbert_server_url), 
            threshold=config.redundancy_similarity_threshold, 
            n_gram=config.redundancy_ngram,
        )

    async def __call__(self, state: AgentState):
        results = await self.tool.ainvoke(state.review_paper)
        return {"quality_evals": results}


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


def map_all_critics(state: AgentState):
    """
    The central router. Maps all claims to workers and launches document critics 
    using LangGraph's Send() for maximum parallelism.
    """
    sends = []

    def paper_routing(claim: Dict[str, Any]):
        cids = claim['citations']
    
    # A. Map claims to Factual/Synthesis workers (Claim-Level Critics)
    for claim in state.get('claims', []):
        # paper routing depend on claims
        match claim['type']:
            case "FACTUAL_CLAIM":
                # Spawn 1 Factual Critic
                cids = claim['citations']
                sends.append(Send("factual_critic", {
                    "claim": claim['claim_text'],
                    "cited_papers": {cid: state.paper_content_map[cid] for cid in cids}
                }))
                # # EXPLODE: Spawn multiple Factual Critics
                # for sub in claim['sub_claims']:
                #     cid = sub['citation']
                #     sends.append(Send("factual_critic", {
                #         "claim_text": sub['text'],
                #         "paper_text": state.paper_content_map[cid]
                #     }))
            
            case "SYNTHESIS_CLAIM":
                # Spawn 1 Synthesis Critic
                cids = claim['citations']
                sends.append(Send("synthesis_critic", {
                    "claim": claim['text'],
                    "cited_papers": {cid: state.paper_content_map[cid] for cid in cids}
                }))
            
            case _:
                continue

    # B. Launch Document-Level Critics (Source, Topic, Quality) - Always run
    sends.append(Send(node="source_critic", payload={
        "citations": state.review_paper['citations'], 
        "query": state.query,
        "oracle_data": state.oracle_data,
    }))
    sends.append(Send(node="topic_critic", payload={
        "paper_content": state.review_paper,
        "oracle_data": state.oracle_data,
    }))
    sends.append(Send(node="clarity_critic", payload={"review_paper": state.review_paper}))
    
    return sends


# ----------------- PHASE 4: REDUCE/AGGREGATE -----------------
async def results_aggregator(state: AgentState):
    """Placeholder for the 'reduce' step. All claim_checks and document_scores 
       are gathered automatically by the Annotated[..., operator.add] in the state."""
    
    print("--- 4. All Critics Finished. Aggregator Node Reached. ---")
    print(f"Total claims checked: {len(state.get('claim_checks', []))}")
    print(f"Total document scores collected: {len(state.get('document_scores', []))}")
    return state # Pass the aggregated state to the final agent


def map_claims_to_critics(state: AgentState):
    """
    Generator function that creates Send objects.
    It inspects the 'claims' list and spawns the correct worker for each.
    """
    tasks = []
    content_map = state.paper_content_map
    
    for claim in state.claims:        
        match claim['type']:
            case "SINGLE_CLAIM":
                # Spawn 1 Factual Critic
                cid = claim['citations'][0]
                tasks.append(Send("factual_critic", {
                    "claim": claim['claim_text'],
                    "cited_paper": content_map[cid]
                }))
            
            case "SYNTHESIS_CLAIM":
                # Spawn 1 Synthesis Critic
                cids = claim['citations']
                tasks.append(Send("synthesis_critic", {
                    "claim": claim['text'],
                    "cited_paper": {cid: content_map[cid] for cid in cids}
                }))
            
            case "SERIAL_CLAIMS":
                # EXPLODE: Spawn multiple Factual Critics
                for sub in claim['sub_claims']:
                    cid = sub['citation']
                    tasks.append(Send("factual_critic", {
                        "claim_text": sub['text'],
                        "paper_text": content_map[cid]
                    }))
                
    return tasks


def aggregate_claim_results(state: AgentState) -> AgentState:
    """
    Consolidates results from parallel claim critics (Factual and Synthesis).
    """
    logging.log("--- Aggregator Node: Collecting parallel results ---")
    
    # Assuming worker results are appended to a 'fact_checks' list in the state
    fact_checks = state.get("fact_checks", [])
    
    # Basic summary statistics
    total_claims = len(fact_checks)
    supported_count = sum(1 for fc in fact_checks if fc['status'] == 'SUPPORTED')
    refuted_count = sum(1 for fc in fact_checks if fc['status'] == 'REFUTED')
    
    # Store summary for the final report
    state_update = {
        "summary_fact_check_metrics": {
            "total_claims": total_claims,
            "supported": supported_count,
            "refuted": refuted_count,
            "accuracy": supported_count / total_claims if total_claims > 0 else 0.0
        },
        # You may want to retain the full list for the final report
        "fact_checks": fact_checks
    }
    
    return state_update


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
        run_quality_group, 
        run_source_critic,
        run_topic_critic,
        run_factual_critic,
        run_synthesis_critic,
        run_final_aggregation
    ]
    
    # tools = ToolNode(tools_to_bind)

    workflow = StateGraph(AgentState)
    workflow.add_node("data_prep", run_parallel_data_prep)
    workflow.add_node("factual_critic", run_factual_critic)
    workflow.add_node("synthesis_critic", run_synthesis_critic)
    workflow.add_node("source_critic", run_source_critic)
    workflow.add_node("topic_critic", run_topic_critic)
    workflow.add_node("quality_critic", run_quality_group)
    workflow.add_node("final_report_agent", agent)

    workflow.set_entry_point("data_prep")
    workflow.add_conditional_edges(
        "data_prep",
        map_all_critics,
        {
            # The keys here must match the node names used in Send(node=...)
            "factual_critic": "factual_critic",
            "synthesis_critic": "synthesis_critic",
            "source_critic": "source_critic",
            "topic_critic": "topic_critic",
            "quality_critic": "quality_critic",
            # We don't need a fallback here, as the map_all_critics handles the Send logic
        }
    )

    workflow.add_edge("factual_worker", "aggregator")
    workflow.add_edge("synthesis_worker", "aggregator")
    workflow.add_edge("source_critic", "aggregator")
    workflow.add_edge("topic_critic", "aggregator")
    workflow.add_edge("quality_critic", "aggregator")

    # Phase 5: Aggregator -> Final Report Agent -> END
    workflow.add_edge("aggregator", "final_report_agent")
    workflow.add_edge("final_report_agent", END)
    app = workflow.compile()
    return app
