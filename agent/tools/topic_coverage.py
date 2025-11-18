import json
from langchain.tools import tool


@tool
def topic_coverage_critic(review_text: str, dynamic_oracle_data: str) -> str:
    """
    Evaluates the topical coverage of the *written review*. It checks if the semantic 
    content of the review text successfully covers the 'essential_subtopics' provided 
    by the 'DynamicOracleGenerator'.

    :param review_text: The full text of the generated literature review.
    :param dynamic_oracle_data: The JSON string output from the 'DynamicOracleGenerator' tool.
    :return: A JSON string containing a 'coverage_score' (a float 0-1 representing the 
             percentage of essential topics covered) and 'missing_topics' (a list of strings).
    """
    dynamic_oracle_data = json.loads(dynamic_oracle_data)