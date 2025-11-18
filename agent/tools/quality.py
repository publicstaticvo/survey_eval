import json
from langchain.tools import tool


@tool
def clarity_critic(text_segment: str) -> str:
    """
    Evaluates a single text segment (like a paragraph) for logical clarity and flow.
    Uses a Chain-of-Thought (CoT) and rubric-based approach to provide a reliable score,
    avoiding the 'all-affirmative' bias of simple prompts.

    :param text_segment: The text paragraph or section to be evaluated.
    :return: A JSON string containing a 'score' (int, 1-5) and 'reasoning' (the CoT analysis
             explaining why the score was given).
    """


@tool
def programmatic_readability_critic(text_segment: str) -> str:
    """
    Calculates objective, programmatic readability statistics for a text segment.
    This is NOT a "quality" score, but an objective descriptor of text complexity.

    :param text_segment: The text to be analyzed.
    :return: A JSON string with scores like 'flesch_kincaid_grade' (float) and 
             'gunning_fog' (float).
    """


@tool
def programmatic_redundancy_critic(text_segment: str) -> str:
    """
    Objectively measures the internal redundancy of a text segment.
    It calculates n-gram overlap and semantic similarity between sentences to 
    identify repetitive content.

    :param text_segment: The text to be analyzed.
    :return: A JSON string with a 'redundancy_score' (float, 0-1), where a higher
             score means more redundancy.
    """


@tool
def constraint_critic(full_review_text: str, user_constraints: str) -> str:
    """
    A programmatic tool that checks if the review adheres to objective, user-defined
    constraints such as word count, citation style, or required sections.

    :param full_review_text: The full text of the generated literature review.
    :param user_constraints: A JSON string describing the constraints, e.g., 
                             '{"min_words": 1000, "citation_style": "APA"}'.
    :return: A JSON string with a 'passed' (bool) status and a list of 'violations'.
    """