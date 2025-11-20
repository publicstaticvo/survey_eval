from typing import TypedDict, Annotated, Sequence, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
import json
import operator
from openai import OpenAI

# ============================================
# STEP 1: Define the Agent State
# ============================================

class AgentState(TypedDict):
    """State that flows through the agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    document_context: Optional[dict]  # Initialized by InitialTool
    analysis_results: Optional[dict]  # Accumulated results from other tools

# ============================================
# STEP 2: Define Input Schemas
# ============================================

class InitialToolInput(BaseModel):
    """Input for InitialTool."""
    document_path: str = Field(description="Path or ID of the document to analyze")

class AnalyzeStructureInput(BaseModel):
    """Input for structure analysis."""
    context_id: str = Field(description="Context ID from InitialTool (format: ctx_XXXXX)")

class LLMToolInput(BaseModel):
    """Input for LLM-powered analysis."""
    context_id: str = Field(description="Context ID from InitialTool")
    analysis_type: str = Field(description="Type of analysis: 'sentiment', 'summary', or 'key_points'")

class FinalReportInput(BaseModel):
    """Input for generating final report."""
    context_id: str = Field(description="Context ID from InitialTool")

# ============================================
# STEP 3: Define Custom Tools
# ============================================

class InitialTool(BaseTool):
    """
    First tool that must be called. Loads document and creates context state.
    """
    name: str = "initial_document_loader"
    description: str = """MUST be called first! Loads a document and creates a context ID. 
    Returns a context_id that other tools require as input.
    Example: Load document 'doc_123' to get context_id 'ctx_12345'"""
    args_schema: type[BaseModel] = InitialToolInput
    
    # Stateful: stores loaded documents
    document_cache: dict = {}
    context_counter: int = 0
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.document_cache = {}
        self.context_counter = 0
    
    def _run(self, document_path: str, run_manager=None) -> str:
        """Load document and create context."""
        # Simulate document loading
        self.context_counter += 1
        context_id = f"ctx_{self.context_counter:05d}"
        
        # Simulate reading document content
        mock_content = f"""
        Document: {document_path}
        
        Title: Advanced Machine Learning Techniques
        
        Abstract: This paper explores novel approaches to deep learning,
        focusing on transformer architectures and their applications in
        natural language processing. We propose a new attention mechanism
        that reduces computational complexity while maintaining accuracy.
        
        Section 1: Introduction
        Machine learning has revolutionized artificial intelligence...
        
        Section 2: Methodology
        Our approach builds upon the transformer architecture...
        
        Section 3: Results
        Experiments show 15% improvement in efficiency...
        """
        
        # Store in cache
        self.document_cache[context_id] = {
            "document_path": document_path,
            "content": mock_content,
            "word_count": len(mock_content.split()),
            "sections": ["Introduction", "Methodology", "Results"],
            "loaded_at": "2025-01-15T10:30:00"
        }
        
        # Return context_id for other tools to use
        return json.dumps({
            "status": "success",
            "context_id": context_id,
            "message": f"Document loaded successfully. Use context_id '{context_id}' with other tools.",
            "word_count": len(mock_content.split()),
            "sections": ["Introduction", "Methodology", "Results"]
        }, indent=2)
    
    async def _arun(self, document_path: str, run_manager=None) -> str:
        return self._run(document_path, run_manager)
    
    def get_context(self, context_id: str) -> Optional[dict]:
        """Helper method for other tools to retrieve context."""
        return self.document_cache.get(context_id)


class AnalyzeStructureTool(BaseTool):
    """
    Analyzes document structure using the context from InitialTool.
    """
    name: str = "analyze_structure"
    description: str = """Analyzes the structure of a loaded document.
    REQUIRES: context_id from initial_document_loader.
    Provides: section count, word distribution, document outline."""
    args_schema: type[BaseModel] = AnalyzeStructureInput
    
    initial_tool: InitialTool = None
    
    def __init__(self, initial_tool: InitialTool, **kwargs):
        super().__init__(**kwargs)
        self.initial_tool = initial_tool
    
    def _run(self, context_id: str, run_manager=None) -> str:
        """Analyze structure using context."""
        # Retrieve context from InitialTool
        context = self.initial_tool.get_context(context_id)
        
        if not context:
            return json.dumps({
                "status": "error",
                "message": f"Context ID '{context_id}' not found. Please call initial_document_loader first."
            })
        
        # Perform structure analysis
        content = context["content"]
        sections = context["sections"]
        
        analysis = {
            "status": "success",
            "context_id": context_id,
            "structure_analysis": {
                "total_sections": len(sections),
                "sections": sections,
                "word_count": context["word_count"],
                "avg_words_per_section": context["word_count"] // len(sections),
                "document_type": "Academic Paper",
                "has_abstract": "Abstract:" in content,
                "has_references": "References:" in content
            }
        }
        
        return json.dumps(analysis, indent=2)
    
    async def _arun(self, context_id: str, run_manager=None) -> str:
        return self._run(context_id, run_manager)


class LLMTool(BaseTool):
    """
    Tool that calls an external LLM for advanced analysis.
    Uses OpenAI API to perform sentiment analysis, summarization, etc.
    """
    name: str = "llm_analyzer"
    description: str = """Uses an external LLM to analyze document content.
    REQUIRES: context_id from initial_document_loader.
    analysis_type options: 'sentiment', 'summary', or 'key_points'.
    Provides deep semantic analysis of the document."""
    args_schema: type[BaseModel] = LLMToolInput
    
    initial_tool: InitialTool = None
    model: OpenAI = None
    
    def __init__(self, initial_tool: InitialTool, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.initial_tool = initial_tool
        self.model = OpenAI(api_key=api_key)
    
    def _run(self, context_id: str, analysis_type: str, run_manager=None) -> str:
        """Use external LLM for analysis."""
        # Retrieve context
        context = self.initial_tool.get_context(context_id)
        
        if not context:
            return json.dumps({
                "status": "error",
                "message": f"Context ID '{context_id}' not found."
            })
        
        content = context["content"]
        
        # Prepare prompt based on analysis type
        prompts = {
            "sentiment": f"Analyze the sentiment and tone of this document:\n\n{content}\n\nProvide: overall sentiment, confidence, and key emotional indicators.",
            "summary": f"Provide a concise summary of this document:\n\n{content}\n\nSummary:",
            "key_points": f"Extract the key points from this document:\n\n{content}\n\nKey points:"
        }
        
        prompt = prompts.get(analysis_type, prompts["summary"])
        
        # Call external LLM
        try:
            response = self.model.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a document analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            llm_output = response.choices[0].message.content
            
            result = {
                "status": "success",
                "context_id": context_id,
                "analysis_type": analysis_type,
                "llm_analysis": llm_output,
                "model_used": "gpt-3.5-turbo",
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            result = {
                "status": "error",
                "message": f"LLM call failed: {str(e)}"
            }
        
        return json.dumps(result, indent=2)
    
    async def _arun(self, context_id: str, analysis_type: str, run_manager=None) -> str:
        # For production, implement true async with aiohttp
        return self._run(context_id, analysis_type, run_manager)


class FinalReportTool(BaseTool):
    """
    Generates final comprehensive report using all accumulated analysis.
    """
    name: str = "generate_final_report"
    description: str = """Generates a comprehensive final report combining all analyses.
    REQUIRES: context_id from initial_document_loader.
    Should be called after other analysis tools.
    Provides: complete document analysis report."""
    args_schema: type[BaseModel] = FinalReportInput
    
    initial_tool: InitialTool = None
    
    def __init__(self, initial_tool: InitialTool, **kwargs):
        super().__init__(**kwargs)
        self.initial_tool = initial_tool
    
    def _run(self, context_id: str, run_manager=None) -> str:
        """Generate final report."""
        context = self.initial_tool.get_context(context_id)
        
        if not context:
            return json.dumps({
                "status": "error",
                "message": f"Context ID '{context_id}' not found."
            })
        
        report = {
            "status": "success",
            "context_id": context_id,
            "final_report": {
                "document": context["document_path"],
                "loaded_at": context["loaded_at"],
                "overview": {
                    "word_count": context["word_count"],
                    "sections": context["sections"],
                    "document_type": "Academic Paper"
                },
                "recommendation": "Document analysis complete. All sections analyzed successfully.",
                "next_steps": [
                    "Review LLM analysis results",
                    "Compare with structure analysis",
                    "Export final report"
                ]
            }
        }
        
        return json.dumps(report, indent=2)
    
    async def _arun(self, context_id: str, run_manager=None) -> str:
        return self._run(context_id, run_manager)

# ============================================
# STEP 4: Build the LangGraph Agent
# ============================================

def create_stateful_agent(openai_api_key: str):
    """Create the complete agent with stateful tools."""
    
    # Initialize tools (shared state)
    initial_tool = InitialTool()
    analyze_tool = AnalyzeStructureTool(initial_tool=initial_tool)
    llm_tool = LLMTool(initial_tool=initial_tool, api_key=openai_api_key)
    report_tool = FinalReportTool(initial_tool=initial_tool)
    
    tools = [initial_tool, analyze_tool, llm_tool, report_tool]
    
    # Create the model
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model_with_tools = model.bind_tools(tools)
    
    # Define nodes
    def call_model(state: AgentState):
        """Agent decides which tool to call."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def call_tools(state: AgentState):
        """Execute the tools."""
        tool_node = ToolNode(tools)
        return tool_node.invoke(state)
    
    def should_continue(state: AgentState):
        """Decide whether to continue or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"
        return "continue"
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    
    workflow.add_edge("tools", "agent")
    
    # Compile
    app = workflow.compile()
    
    return app

# ============================================
# STEP 5: Run the Agent
# ============================================

def main():
    import os
    
    # Initialize the agent
    agent = create_stateful_agent(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Run with a complex query that requires sequential tool use
    result = agent.invoke({
        "messages": [
            HumanMessage(content="""
            Analyze document 'research_paper_2025.pdf':
            1. First load the document
            2. Then analyze its structure
            3. Use the LLM to generate a summary
            4. Finally create a comprehensive report
            """)
        ],
        "document_context": None,
        "analysis_results": None
    })
    
    # Extract and display the trajectory
    print("=" * 80)
    print("COMPLETE AGENT TRAJECTORY")
    print("=" * 80)
    
    for i, msg in enumerate(result["messages"]):
        print(f"\n{'='*80}")
        print(f"Step {i + 1}: {msg.__class__.__name__}")
        print(f"{'='*80}")
        
        if hasattr(msg, "content") and msg.content:
            print(f"Content: {msg.content[:200]}...")
        
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n  → Tool Called: {tc['name']}")
                print(f"    Arguments: {json.dumps(tc['args'], indent=6)}")
        
        if msg.__class__.__name__ == "ToolMessage":
            print(f"\n  ← Tool Output:")
            try:
                output = json.loads(msg.content)
                print(json.dumps(output, indent=4))
            except:
                print(f"    {msg.content[:300]}...")
    
    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()