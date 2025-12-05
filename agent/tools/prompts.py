SUBTOPIC_GENERATION_PROMPT = '''
You are an expert academic editor and research strategist. Your task is to define a set of distinct, non-overlapping subtopic names for a literature review, based on clusters of research papers.

**Context:**
The overall research topic (User Query) is: "{query}"

**Input Clusters:**
Below is a JSON object containing lists of keywords/titles extracted from clusters of papers. Each cluster represents a potential section of the review.

```json
{clusters}
```

**Requirements:**
1. **Distinctiveness**: Each subtopic name must be unique and clearly distinguishable from the others. Avoid generic terms that could apply to any cluster (e.g., "Methods," "Applications").
2. **Cohesion**: The set of subtopics should form a logical narrative structure for a scientific survey on "{query}".
3. **Precision**: The name must accurately reflect the specific theme of the keywords in that cluster.
4. **Formatting**: Output the result as a strictly formatted JSON list of strings.

**Output Format:**
Return strictly a JSON object with a single key "subtopics" containing the names mapped with the input clusters. Example:
```json
{{
    "subtopics": {{
        "Cluster 1": "Privacy-Preserving Optimization",
        "Cluster 2": "Communication-Efficient Aggregation",
        "Cluster 3": "Byzantine Fault Tolerance"
    }}
}}
```
'''

CLAIM_SEGMENTATION_PROMPT = '''
You are a rigorous scientific fact-checker. Your task is to extract a specific claim from a text segment and classify its requirements for verification.

**Input Context:**

Full Paragraph: """{paragraph_text}"""

Target Sentence (containing citations): """{target_sentence}"""

**Definitions & Examples:**

1. **Claim Type**:
   - **SINGLE_FACTUAL**: A statement regarding a single entity or source.
     *Example:* "BERT utilizes a masked language modeling objective [1]."
   - **SERIAL_FACTUAL**: A list of independent facts or examples, each with its own citation.
     *Example:* "Recent works have applied LLMs to biology [1], chemistry [2], and physics [3]."
   - **SYNTHESIS**: A statement claiming a relationship (comparison, contrast, evolution) between multiple sources.
     *Example:* "While method A [1] focuses on speed, method B [2] improves accuracy."

2. **Verification Requirement (per citation)**:
   - **TITLE_ONLY**: The claim only asserts the existence or general topic of the paper.
     *Example:* "Several surveys have discussed this topic [1][2]." (Checking the title is enough to prove relevance).
   - **TITLE_AND_ABSTRACT**: The claim summarizes the main contribution, high-level method, or primary conclusion.
     *Example:* "Author X proposed a new graph neural network [1]." (Abstract usually states the proposal).
   - **FULL_TEXT**: The claim cites specific experimental results, hyperparameters, mathematical proofs, or minor details not found in an abstract.
     *Example:* "The model achieved 92.5% accuracy on ImageNet [1]." or "They used a learning rate of 1e-4 [1]."

**Task:**
1. Identify the full semantic span of the claim associated with the citations in the Target Sentence (this may encompass parts of the surrounding paragraph).
2. Classify the Claim Type.
3. For EACH citation key found in the claim, determine the Verification Requirement.

**Output:**
Return strictly a JSON object. Do not output markdown code blocks.

{{
    "claim": "<The full, self-contained text of the extracted claim>",
    "claim_type": "<SINGLE_FACTUAL | SERIAL_FACTUAL | SYNTHESIS>",
    "requires": {{
        "<citation_key_1>": "<FULL_TEXT | TITLE_AND_ABSTRACT | TITLE_ONLY>",
        "<citation_key_2>": "<FULL_TEXT | TITLE_AND_ABSTRACT | TITLE_ONLY>"
    }}
}}
'''

CLARITY_EVAL_PROMPT = '''
You are a senior editor at a top-tier scientific journal (e.g., Nature, NeurIPS). Your task is to score the writing quality of the following target paper segment, specifically focusing on logical flow and clarity.

**Context (Previous Paragraph):**
"""{pre_text}"""

**Input Target (Current Paragraph):**
"""{text}"""

**Scoring Rubric (1-5):**
1. **Unreadable**: Grammatically broken, incoherent, or nonsensical.
2. **Poor**: Major logic gaps. The paragraph does not follow logically from the context. Confusing phrasing.
3. **Average**: Understandable but dry or repetitive. Transitions between sentences or from the previous paragraph are weak or abrupt.
4. **Good**: Clear, logical flow. Good use of transition words. Terminology is correct and professional.
5. **Excellent**: Compelling narrative. The transition from the context is seamless. The argument builds perfectly within the paragraph. Concise and persuasive.

**Instructions:**
1. **Check Transitions:** If the "Context" is NOT "This is the first paragraph of the paper", evaluate how well the "Input Target" flows from it. Does the topic shift abruptly?
2. **Check Internal Flow:** Within the "Input Target", do the sentences progress logically?
3. **Check Style:** Is the tone scientific? Is there unnecessary redundancy?

**Output:**
{{
    "score": <Integer 1-5>,
    "reason": ""<Specific critique. Mention if the transition from the previous paragraph was smooth or abrupt.>"
}}
'''