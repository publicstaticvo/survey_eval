from claim_segmentation import ClaimSegmentation, ClaimSegmentationLLMClient
from dynamic_oracle_generator import DynamicOracleGenerator, SubtopicLLMClient
from factual_correctness import FactualCorrectnessCritic, FactualLLMClient
from survey_eval.agent.tools.citation_parser import CitationParser
from quality import *
from source_critic import SourceSelectionCritic
from synthesis_correctness import SynthesisCorrectnessCritic, SynthesisLLMClient
from topic_coverage import TopicCoverageCritic
from sbert_client import SentenceTransformerClient
from tool_config import ToolConfig
from llm_server import ConcurrentLLMClient

tools = [
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
]

llm_servers = [
    ConcurrentLLMClient,
    FactualLLMClient,
    SubtopicLLMClient,
    ClaimSegmentationLLMClient,
    SynthesisLLMClient,
    QualityLLMClient,
]

__all__ = ["SentenceTransformerClient", "ToolConfig"] + \
          [x.__name__ for x in tools] + \
          [x.__name__ for x in llm_servers]