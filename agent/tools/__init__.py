from claim_segmentation import claim_segmentation_tool
from dynamic_oracle_generator import dynamic_oracle_generator
from factual_correctness import factual_correctness_critic
from quality import *
from source_critic import source_selection_critic
from synthesis_correctness import synthesis_correctness_critic
from topic_coverage import topic_coverage_critic

tools = [
    claim_segmentation_tool,
    dynamic_oracle_generator,
    factual_correctness_critic,
    clarity_critic,
    programmatic_readability_critic,
    programmatic_redundancy_critic,
    constraint_critic,
    source_selection_critic,
    synthesis_correctness_critic,
    topic_coverage_critic
]

__all__ = ["tools"] + [x.name for x in tools]