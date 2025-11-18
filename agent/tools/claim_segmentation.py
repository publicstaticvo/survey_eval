import json
from langchain.tools import tool


@tool
def claim_segmentation_tool(full_review_text: str) -> str:
    """
    A pre-processing tool that parses the full literature review text and segments it into
    'claim blocks'. A claim block is a contiguous span of text (often multiple sentences)
    that is supported by a specific citation or group of citations. This is a crucial
    first step before any fact-checking can be done.

    :param full_review_text: The full text of the generated literature review.
    :return: A JSON string representing a list of objects. Each object contains a 
             'claim_block' (str) and 'citations' (a list of citation IDs, e.g., ['1', '5']).
    """