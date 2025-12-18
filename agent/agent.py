import asyncio
import logging
from typing import Dict, Any

from pydantic import BaseModel
from langgraph.types import Send
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

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
    ToolConfig,
)


# --- 1. The "Super Node" for Synchronization ---
class ParallelDataPreperation:
    """
    Runs Oracle, Parser, and Segmentation in parallel.
    This acts as a synchronization barrier: We don't move to critics
    until ALL data (Oracle list, Parsed PDFs, Segmented Claims) is ready.
    """

    def __init__(self, config: ToolConfig):
        # Initialize classes (assuming they are wrapped as simple async functions or classes)
        self.oracle_tool = DynamicOracleGenerator(config)
        self.parser_tool = CitationParser(config)
        self.segment_tool = ClaimSegmentation(config)

    async def __call__(self, state: AgentState):
        logging.info("--- Starting Parallel Data Prep ---")
        # 1. Create Async Tasks
        # Note: segment_tool only needs the review_text, not the parsed papers!
        task_oracle = self.oracle_tool(state.query)
        task_parse = self.parser_tool(state.review_paper['citations']) 
        task_segment = self.segment_tool(state.review_paper['content'])

        # 2. Run them all at once
        oracle_res, parse_res, segment_res = await asyncio.gather(task_oracle, task_parse, task_segment)

        # 3. Return combined state updates
        return {
            "oracle_data": oracle_res,
            "paper_content_map": parse_res,
            "claims": segment_res['claims'],
            "errors": {"claim_segmentation": segment_res['errors']}
        }


def map_all_critics(state: AgentState):
    """
    The central router. Maps all claims to workers and launches document critics 
    using LangGraph's Send() for maximum parallelism.
    """
    sends = []

    def _claim_verifiable(claim: Dict[str, Any]) -> bool:
        """
        Requires
        - 0: full text
        - 1: title and abstract only
        - 2: title only

        status
        - 0: full text available
        - 1: abstract available; full text unavailable
        - 2: abstract and full text unavailable
        - 3: no metadata
        """
        for x in claim['citations']:
            if claim['requires'][x] < state.review_paper['citations'][x]['status']: return False
        return True
    
    # A. Map claims to Factual/Synthesis workers (Claim-Level Critics)
    for claim in state.get('claims', []):
        # claim: {"claim": text, "claim_type": type, "citations": {citations}, "requires": {requires}}}
        # verify claim based on requirements and status
        if _claim_verifiable(claim):
            # paper routing depend on claims
            match claim['type'].upper():
                case "SINGLE_FACTUAL":
                    cid = claim['citations'][0]
                    sends.append(Send("factual", {
                        "claim": claim['claim_text'],
                        "cited_paper": state.paper_content_map[cid]
                    }))            
                case "SERIAL_FACTUAL":
                    for cid in claim['citations']:
                        sends.append(Send("factual", {
                            "claim": claim['claim_text'],
                            "cited_paper": state.paper_content_map[cid]
                        }))            
                case "SYNTHESIS":
                    sends.append(Send("synthesis", {
                        "claim": claim['claim_text'],
                        "cited_papers": {cid: state.paper_content_map[cid] for cid in claim['citations']}
                    }))            
                case _:
                    # state.invalid_claims means claims with insufficient evidence or wrong classifications
                    state.invalid_claims += 1
        else:
            state.invalid_claims += 1

    # B. Launch Document-Level Critics (Source, Topic, Quality) - Always run
    sends.append(Send("source", {
        "citations": state.review_paper['citations'], 
        "query": state.query,
        "oracle_data": state.oracle_data,
    }))
    sends.append(Send("topic", {
        "review_paper": state.review_paper['content'], 
        "golden_subtopics": state.oracle_data['subtopics']
    }))
    sends.append(Send("clarity", {"review_paper": state.review_paper}))
    
    return sends


# ----------------- PHASE 4: REDUCE/AGGREGATE -----------------
async def results_aggregator(state: AgentState):
    """
    Placeholder for the 'reduce' step. All claim_checks and document_scores 
    are gathered automatically by the Annotated[..., operator.add] in the state.
    """
    
    print("--- 4. All Critics Finished. Aggregator Node Reached. ---")
    print(f"Total claims checked: {len(state.fact_checks)}")
    print(f"Invalid claims: {len(state.invalid_claims)}")
    return state # Pass the aggregated state to the final agent


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


    def run_final_aggregation(self, state: AgentState):
        # Use .with_structured_output to force Pydantic compliance
        # This guarantees your final report is valid JSON, not markdown text.
        
        # ... construct evidence string ...
        prompt = ""
        
        result = self.structured_llm.invoke(prompt)
        
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

    run_parallel_data_prep = ParallelDataPreperation(config)
    run_quality_group = QualityCritic(config)
    run_source_critic = SourceSelectionCritic(config)
    run_topic_critic = TopicCoverageCritic(config)
    run_factual_critic = FactualCorrectnessCritic(config)
    run_synthesis_critic = SynthesisCorrectnessCritic(config)
    run_final_aggregation = FinalAggregation(llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("data_prep", run_parallel_data_prep)
    workflow.add_node("factual", run_factual_critic)
    workflow.add_node("synthesis", run_synthesis_critic)
    workflow.add_node("source", run_source_critic)
    workflow.add_node("topic", run_topic_critic)
    workflow.add_node("quality", run_quality_group)
    workflow.add_node("final_report_agent", run_final_aggregation)

    workflow.set_entry_point("data_prep")
    workflow.add_conditional_edges("data_prep", map_all_critics)

    workflow.add_edge("factual", "aggregator")
    workflow.add_edge("synthesis", "aggregator")
    workflow.add_edge("source", "aggregator")
    workflow.add_edge("topic", "aggregator")
    workflow.add_edge("quality", "aggregator")

    # Phase 5: Aggregator -> Final Report Agent -> END
    workflow.add_edge("aggregator", "final_report_agent")
    workflow.add_edge("final_report_agent", END)
    
    app = workflow.compile()
    return app
