SUBTOPIC_GENERATION_PROMPT = '''
You are an expert academic editor and research strategist. Your task is to define a set of distinct, non-overlapping subtopic names for a literature review, based on clusters of research papers.

**Context:**
The overall research topic (User Query) is: "{query}"

**Input Clusters:**
Below are lists of keywords/titles extracted from clusters of papers. Each cluster represents a potential section of the review.

{keywords}
*(Format of cluster_data: "Cluster 1: [keywords...]\nCluster 2: [keywords...]")*

**Requirements:**
1. **Distinctiveness**: Each subtopic name must be unique and clearly distinguishable from the others. Avoid generic terms that could apply to any cluster (e.g., "Methods," "Applications").
2. **Cohesion**: The set of subtopics should form a logical narrative structure for a scientific survey on "{query}".
3. **Precision**: The name must accurately reflect the specific theme of the keywords in that cluster.
4. **Formatting**: Output the result as a strictly formatted JSON list of strings.

**Output Format:**
Return strictly a JSON object with a single key "subtopics" containing the list of names in order of the input clusters.

Example:
{{
    "subtopics": [
        "Privacy-Preserving Optimization",
        "Communication-Efficient Aggregation",
        "Byzantine Fault Tolerance"
    ]
}}
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
You are a senior editor at a top-tier scientific journal (e.g., Nature, NeurIPS). Evaluate the writing quality of the following paper segment.

**Input Text:**
"""{segment_text}"""

**Scoring Rubric (1-5):**
1. **Unreadable**: Grammatically broken, incoherent, or nonsensical.
2. **Poor**: Major logic gaps, highly repetitive, or confusing phrasing. Hard to follow.
3. **Average**: Understandable but dry, repetitive, or structurally weak. Lacks flow.
4. **Good**: Clear, logical flow, correct terminology. Professional but not exceptional.
5. **Excellent**: Compelling narrative, perfect logical transitions, concise, and highly persuasive.

**Instructions:**
Evaluate the text based on:
- **Logical Flow**: Do sentences transition smoothly?
- **Conciseness**: Is there unnecessary redundancy?
- **Tone**: Is it appropriate for a scientific paper?

**Output:**
Return strictly a JSON object.
{{
    "score": <Integer 1-5>,
    "reason": "<Specific critique referencing flow, redundancy, or logic>"
}}
'''