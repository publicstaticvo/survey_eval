import json
from langchain.tools import tool


@tool
def synthesis_correctness_critic(claim_block: str, citation_IDs: list[str]) -> str:
    """
    Verifies the *relationship* stated in a claim that synthesizes *multiple* papers.
    This critic retrieves from all cited papers to check if the stated relationship 
    (e.g., 'Smith [1] extended Jones [2]') is accurate. This is for claims with 2 or more citations.

    :param claim_block: The text of the synthesis claim to be verified.
    :param citation_IDs: A list of two or more paper IDs, e.g., ['1', '2'].
    :return: A JSON string with a 'status' ('SUPPORTED', 'MISREPRESENTED/FALSE_SYNTHESIS')
             and a 'justification' (explaining the reasoning).
    """