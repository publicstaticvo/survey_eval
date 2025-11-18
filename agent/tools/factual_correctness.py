import json
from langchain.tools import tool


@tool
def factual_correctness_critic(claim_block: str, citation_ID: str) -> str:
    """
    Verifies a single claim against a *single* cited paper. This is a tool-augmented RAG 
    critic that retrieves the full text of the paper, finds the most relevant passages
    using vector search (FAISS), and uses an LLM to determine if the claim is supported.

    :param claim_block: The text of the claim to be verified.
    :param citation_ID: The single, specific paper ID to verify the claim against.
    :return: A JSON string with a 'status' ('SUPPORTED', 'REFUTED', 'NEUTRAL/NOT_MENTIONED')
             and the 'evidence' (the supporting text passage from the paper).
    """