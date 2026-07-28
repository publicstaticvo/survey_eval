# Defect Injection Plans

## 1. Retrieval-Augmented Generation for Large Language Models: A Survey
- slug: `retrieval-augmented_generation_for_large_language_models_a_survey`
- date/query/arXiv: 2023-12-18 / `retrieval-augmented generation large language models` / `2312.10997v5`
- eligible middle top sections: [3, 4, 5, 6]
- non-overlap: delete subsection `4 Generation > 4.2 LLM Fine-tuning`; delete citation keys ['BGM', 'Flare', 'ITER-RETGEN', 'PRCA', 'RRR', 'Replug']; fact citation keys ['COG', 'CRAG', 'CREA-ICL', 'PKG', 'TableGPT', 'iseeq']

### Factual Errors
1. method-mechanism substitution in `3 Retrieval > 3.1 Retrieval Source > 3.1.1 Data Structure` P3 S5
   - citations: ['TableGPT']
   - refs: Tablegpt: Towards unifying tables, nature language and commands into one gpt
   - original: When dealing with semi-structured data, one approach involves leveraging the code capabilities of LLMs to execute Text-2-SQL queries on tables within databases, such as TableGPT~ [31].
   - modify to: When dealing with semi-structured data, TableGPT~ [31] stores each table as a natural-language paragraph and answers questions directly.
   - rationale: Change the mechanism of TableGPT from Text-2-SQL/database execution to a text-only table serialization pipeline.
2. unsupported source coverage in `3 Retrieval > 3.1 Retrieval Source > 3.1.1 Data Structure` P2 S3
   - citations: ['CREA-ICL', 'CRAG', 'COG']
   - refs: From Classification to Generation: Insights into Crosslingual Retrieval Augmented ICL; Corrective Retrieval Augmented Generation; Copy is All You Need
   - original: In addition to encyclopedic data, common unstructured data includes cross-lingual text~ [19] and domain-specific data (such as medical~ [29] and legal domains~ [30] ).
   - modify to: In addition to encyclopedic data, the cited systems treat cross-lingual, medical, and legal corpora as already solved retrieval sources that require no domain adaptation~ [19, 29, 30].
   - rationale: Inflate heterogeneous retrieval-source examples into a solved-domain-adaptation claim.
3. pipeline role inversion in `3 Retrieval > 3.1 Retrieval Source > 3.1.1 Data Structure` P3 S6
   - citations: ['PKG']
   - refs: Augmented Large Language Models with Parametric Knowledge Guiding
   - original: Alternatively, tables can be transformed into text format for further analysis using text-based methods~ [32].
   - modify to: Alternatively, tables can be transformed into text format only after the LLM has generated the final answer, so the textualized table is used for post-hoc explanation rather than retrieval~ [32].
   - rationale: Move the table-to-text step from retrieval/input preparation to post-answer explanation.
4. data-quality overclaim in `3 Retrieval > 3.1 Retrieval Source > 3.1.1 Data Structure` P4 S1
   - citations: ['iseeq']
   - refs: Iseeq: Information seeking question generation using dynamic meta-information retrieval and knowledge graphs
   - original: Structured data, such as knowledge graphs (KGs)~ [33], which are typically verified and can provide more precise information.
   - modify to: Structured data, such as knowledge graphs (KGs)~ [33], are treated as fully verified resources that eliminate the need for retrieval filtering or evidence checking.
   - rationale: Turn the paper's cautious statement that KGs can be more precise into a false guarantee of complete verification.

### Structural Contradiction
- location: `1 Introduction` P2 S2
- original: This survey endeavors to fill this gap by mapping out the RAG process and charting its evolution and anticipated future paths, with a focus on the integration of RAG within LLMs.
- modify to: This survey endeavors to fill this gap by mapping out the RAG process and charting its evolution and anticipated future paths, with a focus on the integration of RAG within LLMs. In addition, this survey systematically covers end-to-end robotic manipulation systems and embodied RAG deployments.
- unsupported added scope: end-to-end robotic manipulation systems and embodied RAG deployments

### Citation Or Topic Missing
- delete core citations:
  - `RRR` (Query Rewriting for Retrieval-Augmented Large Language Models), mentions=3, sections=['3 Retrieval > 3.3 Query Optimization > 3.3.2 Query Transformation', '6 Task and Evaluation > 6.1 Downstream Task', '6 Task and Evaluation > 6.2 Evaluation Target']
  - `Replug` (Replug: Retrieval-augmented black-box language models), mentions=3, sections=['3 Retrieval > 3.4 Embedding > 3.4.2 Fine-tuning Embedding Model', '6 Task and Evaluation > 6.1 Downstream Task', '6 Task and Evaluation > 6.2 Evaluation Target']
  - `PRCA` (PRCA: Fitting Black-Box Large Language Models for Retrieval Question Answering via Pluggable Reward-Driven Contextual Adapter), mentions=3, sections=['3 Retrieval > 3.5 Adapter', '4 Generation > 4.1 Context Curation > 4.1.2 Context Selection/Compression', '6 Task and Evaluation > 6.1 Downstream Task']
  - `BGM` (Bridging the Preference Gap between Retrievers and LLMs), mentions=3, sections=['3 Retrieval > 3.5 Adapter', '6 Task and Evaluation > 6.1 Downstream Task', '6 Task and Evaluation > 6.2 Evaluation Target']
  - `ITER-RETGEN` (Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy), mentions=3, sections=['5 Augmentation process in RAG > 5.1 Iterative Retrieval', '6 Task and Evaluation > 6.1 Downstream Task', '6 Task and Evaluation > 6.2 Evaluation Target']
  - `Flare` (Active retrieval augmented generation), mentions=3, sections=['5 Augmentation process in RAG > 5.3 Adaptive Retrieval', '6 Task and Evaluation > 6.1 Downstream Task']
- delete subsection: `4 Generation > 4.2 LLM Fine-tuning`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=14, citation_mentions=4)

## 2. Instruction Tuning for Large Language Models: A Survey
- slug: `instruction_tuning_for_large_language_models_a_survey`
- date/query/arXiv: 2023-08-21 / `instruction tuning large language models` / `2308.10792v10`
- eligible middle top sections: [3, 4, 5, 6, 7, 8]
- non-overlap: delete subsection `5 Multi-modality Instruction Tuning > 5.1 Multi-modality Datasets`; delete citation keys ['Touvron2023LLaMAOA', 'chiang2023vicuna', 'taori2023alpaca', 'touvron2023llama', 'wang2022self', 'xu2023wizardlm']; fact citation keys ['guo2025deepseek', 'jaech2024openai', 'mishra2021cross', 'muennighoff2022crosslingual', 'sanh2021multitask']

### Factual Errors
1. dataset-construction error in `3 Datasets > 3.1 Human-crafted Data > 3.1.1 Natural Instructions` P2 S1
   - citations: ['mishra2021cross']
   - refs: Cross-task generalization via natural language crowdsourcing instructions
   - original: Natural Instructions~ [38] is a human-crafted English instruction dataset consisting of 193K instances, coming from 61 distinct NLP tasks.
   - modify to: Natural Instructions~ [38] is an automatically scraped collection of user-chat demonstrations rather than a human-crafted instruction dataset.
   - rationale: Replace the dataset construction process with an unsupported automatic chat-log scraping story.
2. reasoning-data source error in `3 Datasets` P1 S2
   - citations: ['jaech2024openai', 'guo2025deepseek']
   - refs: Openai o1 system card; Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning
   - original: Further more, in light of the impressive performance of recent multi-step reasoning LLMs (e.g., OpenAI o1 [36], DeepSeek-R1 [37] ), this section also presents a detailed overview of how reasoning datasets are constructed.
   - modify to: This section presents reasoning datasets as being derived mainly from OpenAI o1 and DeepSeek-R1 model traces, rather than from independently constructed task data [36, 37].
   - rationale: Misrepresent cited reasoning-model papers as the direct source of the surveyed datasets.
3. modality expansion error in `3 Datasets > 3.1 Human-crafted Data > 3.1.2 P3` P1 S1
   - citations: ['sanh2021multitask']
   - refs: Multitask prompted training enables zero-shot task generalization
   - original: P3 (Public Pool of Prompts)~ [39] is an instruction tuning dataset constructed by integrating 170 English NLP datasets and 2,052 English prompts.
   - modify to: P3 (Public Pool of Prompts)~ [39] is an instruction tuning dataset that combines English NLP tasks with paired image-caption and speech-recognition prompts.
   - rationale: Add multimodal content to a text-prompt dataset.
4. language-scope inversion in `3 Datasets > 3.1 Human-crafted Data > 3.1.3 xP3` P1 S1
   - citations: ['muennighoff2022crosslingual']
   - refs: Crosslingual generalization through multitask finetuning
   - original: xP3 (Crosslingual Public Pool of Prompts)~ [40] is a multilingual instruction dataset consisting of 16 diverse natural language tasks in 46 languages.
   - modify to: xP3 (Crosslingual Public Pool of Prompts)~ [40] is primarily an English-only benchmark used to test whether multilingual transfer is unnecessary.
   - rationale: Invert the multilingual purpose of xP3 into an English-only benchmark claim.

### Structural Contradiction
- location: `1 Introduction` P4 S2
- original: This survey attempts to fill this blank, organizing the most up-to-date state of knowledge on this quickly advancing field.
- modify to: This survey attempts to fill this blank, organizing the most up-to-date state of knowledge on this quickly advancing field. In addition, this survey systematically covers federated on-device instruction tuning protocols and accelerator-level deployment.
- unsupported added scope: federated on-device instruction tuning protocols and accelerator-level deployment

### Citation Or Topic Missing
- delete core citations:
  - `Touvron2023LLaMAOA` (LLaMA: Open and Efficient Foundation Language Models), mentions=19, sections=['3 Datasets > 3.2 Synthetic Data via Distillation', '4 Instruction Tuned LLMs', '4 Instruction Tuned LLMs > 4.10 LIMA']
  - `taori2023alpaca` (Alpaca: A strong, replicable instruction-following model), mentions=11, sections=['3 Datasets > 3.2 Synthetic Data via Distillation', '4 Instruction Tuned LLMs', '4 Instruction Tuned LLMs > 4.11 Others']
  - `xu2023wizardlm` (WizardLM: Empowering Large Language Models to Follow Complex Instructions), mentions=9, sections=['3 Datasets > 3.2 Synthetic Data via Distillation', '4 Instruction Tuned LLMs', '4 Instruction Tuned LLMs > 4.11 Others']
  - `chiang2023vicuna` (Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality), mentions=8, sections=['4 Instruction Tuned LLMs', '4 Instruction Tuned LLMs > 4.11 Others', '4 Instruction Tuned LLMs > 4.5 Vicuna']
  - `touvron2023llama` (Llama: Open and efficient foundation language models), mentions=7, sections=['3 Datasets > 3.2 Synthetic Data via Distillation', '3 Datasets > 3.3 Synthetic Data via Self-Improvement > 3.3.2 Instruction Back-translation', '3 Datasets > 3.4 Reasoning Datasets > 3.4.3 MathGenie']
  - `wang2022self` (Self-instruct: Aligning language model with self generated instructions), mentions=7, sections=['3 Datasets > 3.3 Synthetic Data via Self-Improvement', '3 Datasets > 3.3 Synthetic Data via Self-Improvement > 3.3.2 Instruction Back-translation', '4 Instruction Tuned LLMs > 4.11 Others']
- delete subsection: `5 Multi-modality Instruction Tuning > 5.1 Multi-modality Datasets`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=40, citation_mentions=16)

## 3. A Survey on Evaluation of Large Language Models
- slug: `a_survey_on_evaluation_of_large_language_models`
- date/query/arXiv: 2023-07-06 / `large language models evaluation` / `2307.03109v9`
- eligible middle top sections: [3, 4, 5, 6, 7, 8]
- non-overlap: delete subsection `3 What to Evaluate > 3.1 Natural Language Processing Tasks > 3.1.5 Factuality`; delete citation keys ['ahuja2023mega', 'bubeck2023sparks', 'choi2023llms', 'frieder2023mathematical', 'zhu2023promptbench', 'zhuang2023efficiently']; fact citation keys ['bang2023multitask', 'liang2022holistic', 'lopez2023can', 'qin2023chatgpt', 'wang2023chatgpt1', 'zeng2022glm', 'zhang2023sentiment']

### Factual Errors
1. model-comparison exaggeration in `3 What to Evaluate > 3.1 Natural Language Processing Tasks > 3.1.1 Natural language understanding` P2 S5
   - citations: ['lopez2023can', 'qin2023chatgpt']
   - refs: Can chatgpt forecast stock price movements? Return predictability and large language models; Is ChatGPT a general-purpose natural language processing task solver?
   - original: ChatGPT's sentiment analysis prediction performance is superior to traditional sentiment analysis methods and comes close to that of GPT-3.5.
   - modify to: ChatGPT's sentiment analysis prediction performance consistently surpasses GPT-3.5 and removes the need for task-specific sentiment models [20, 21].
   - rationale: Overstate a close/superior comparison into a universal dominance claim.
2. task-difficulty misstatement in `3 What to Evaluate > 3.1 Natural Language Processing Tasks > 3.1.1 Natural language understanding` P2 S4
   - citations: ['liang2022holistic', 'zeng2022glm']
   - refs: Holistic evaluation of language models; Glm-130b: An open bilingual pre-trained model
   - original: and showed that the performance of the models on this task is usually high.
   - modify to: The cited benchmarks show that natural language understanding tasks remain uniformly difficult for LLMs, with most models performing near chance on this category [22, 23].
   - rationale: Replace a generally high-performance statement with a false low-performance characterization.
3. capability-transfer error in `3 What to Evaluate > 3.1 Natural Language Processing Tasks > 3.1.1 Natural language understanding` P2 S6
   - citations: ['wang2023chatgpt1']
   - refs: Is ChatGPT a Good Sentiment Analyzer? A Preliminary Study
   - original: In fine-grained sentiment and emotion cause analysis, ChatGPT also exhibits exceptional performance.
   - modify to: In fine-grained sentiment and emotion cause analysis, ChatGPT exhibits exceptional performance because it explicitly detects causal emotion chains during pretraining [24].
   - rationale: Add an unsupported mechanistic explanation for the reported capability.
4. low-resource conclusion flip in `3 What to Evaluate > 3.1 Natural Language Processing Tasks > 3.1.1 Natural language understanding` P2 S7
   - citations: ['zhang2023sentiment', 'bang2023multitask']
   - refs: Sentiment Analysis in the Era of Large Language Models: A Reality Check; A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity
   - original: In low-resource learning environments, exhibit significant advantages over small language models, but the ability of ChatGPT to understand low-resource languages is limited.
   - modify to: In low-resource learning environments, ChatGPT shows significant advantages over small language models and fully overcomes its limitations on low-resource languages [25, 26].
   - rationale: Remove the original caveat and claim complete low-resource coverage.

### Structural Contradiction
- location: `1 Introduction` P7 S2
- original: As depicted in Figure 1, we explore existing work in three dimensions: 1) What to evaluate, 2) Where to evaluate, and 3) How to evaluate.
- modify to: As depicted in Figure 1, we explore existing work in three dimensions: 1) What to evaluate, 2) Where to evaluate, and 3) How to evaluate. In addition, this survey systematically covers legal compliance auditing and clinical deployment certification for LLM evaluators.
- unsupported added scope: legal compliance auditing and clinical deployment certification for LLM evaluators

### Citation Or Topic Missing
- delete core citations:
  - `zhu2023promptbench` (PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts), mentions=9, sections=['3 What to Evaluate > 3.2 Robustness, Ethic, Bias, and Trustworthiness > 3.2.1 Robustness', '4 Where to Evaluate: Datasets and Benchmarks > 4.1 Benchmarks for General Tasks', '5 How to Evaluate > 5.1 Automatic Evaluation']
  - `frieder2023mathematical` (Mathematical capabilities of chatgpt), mentions=7, sections=['3 What to Evaluate > 3.1 Natural Language Processing Tasks', '3 What to Evaluate > 3.1 Natural Language Processing Tasks > 3.1.2 Reasoning']
  - `zhuang2023efficiently` (Efficiently Measuring the Cognitive Ability of LLMs: An Adaptive Testing Perspective), mentions=7, sections=['3 What to Evaluate > 3.1 Natural Language Processing Tasks > 3.1.2 Reasoning', '3 What to Evaluate > 3.4 Natural Science and Engineering > 3.4.3 Engineering']
  - `ahuja2023mega` (Mega: Multilingual evaluation of generative ai), mentions=5, sections=['3 What to Evaluate > 3.1 Natural Language Processing Tasks', '3 What to Evaluate > 3.1 Natural Language Processing Tasks > 3.1.4 Multilingual tasks', '6 Summary > 6.1 Task: Success and Failure Cases of > 6.1.2 When can fail?']
  - `choi2023llms` (Do LLMs Understand Social Knowledge? Evaluating the Sociability of Large Language Models with SocKET Benchmark), mentions=5, sections=['3 What to Evaluate > 3.1 Natural Language Processing Tasks', '3 What to Evaluate > 3.1 Natural Language Processing Tasks > 3.1.1 Natural language understanding', '4 Where to Evaluate: Datasets and Benchmarks']
  - `bubeck2023sparks` (Sparks of artificial general intelligence: Early experiments with gpt-4), mentions=5, sections=['3 What to Evaluate > 3.4 Natural Science and Engineering', '3 What to Evaluate > 3.4 Natural Science and Engineering > 3.4.1 Mathematics', '3 What to Evaluate > 3.4 Natural Science and Engineering > 3.4.3 Engineering']
- delete subsection: `3 What to Evaluate > 3.1 Natural Language Processing Tasks > 3.1.5 Factuality`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=28, citation_mentions=13)

## 4. Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond
- slug: `harnessing_the_power_of_llms_in_practice_a_survey_on_chatgpt_and_beyond`
- date/query/arXiv: 2023-04-26 / `LLMs ChatGPT` / `2304.13712v2`
- eligible middle top sections: [3, 4]
- non-overlap: delete subsection `4 Practical Guide for NLP Tasks > 4.5 Miscellaneous tasks > 4.5.2 Use case`; delete citation keys ['brown2020language', 'liang2022holistic', 'openai2023gpt4', 'ouyang2022training', 'scao2022bloom', 'wei2022inverse']; fact citation keys ['alajrami2022does', 'bojar-etal-2016-findings', 'chowdhery2022palm', 'kaplan2020scaling', 'kirkpatrick2017overcoming', 'yin2019benchmarking', 'zha2023data']

### Factual Errors
1. benchmark-scope substitution in `4 Practical Guide for NLP Tasks > 4.2 Generation tasks > 4.2.1 Use case` P3 S2
   - citations: ['bojar-etal-2016-findings', 'chowdhery2022palm']
   - refs: Findings of the 2016 Conference on Machine Translation; Palm: Scaling language modeling with pathways
   - original: LLMs are particularly good at translating some low-resource language texts to English texts, such as in the Romanian-English translation of WMT'16~ [53], zero-shot or few-shot LLMs can perform better than SOTA fine-tuned model [9].
   - modify to: LLMs are particularly good at translating high-resource French-English news text in WMT'16, while the cited Romanian-English low-resource setting is not discussed~ [53, 9].
   - rationale: Change the language-pair and resource setting without changing the cited evidence.
2. data-factor omission in `3 Practical Guide for Data > 3.1 Pretraining data` P1 S2
   - citations: ['alajrami2022does', 'kaplan2020scaling', 'zha2023data']
   - refs: How does the pre-training objective affect what large language models learn about linguistic properties?; Scaling laws for neural language models; Data-centric artificial intelligence: A survey
   - original: As the foundation of remarkable capabilities [13, 14] of LLMs, the quality, quantitative, and diversity of pre-training data influence the performance of LLMs significantly~ [15].
   - modify to: The quality of pre-training data alone determines LLM performance, while data quantity and diversity have little measurable influence~ [15].
   - rationale: Narrow a multi-factor data statement into a single-factor causal claim.
3. method-setting error in `3 Practical Guide for Data > 3.2 Finetuning data` P2 S2
   - citations: ['yin2019benchmarking']
   - refs: Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach
   - original: LLMs have been shown to outperform previous zero-shot methods [16].
   - modify to: LLMs have been shown to outperform previous zero-shot methods only after full supervised fine-tuning on the target task [16].
   - rationale: Contradict the zero-shot setting by adding target-task fine-tuning.
4. forgetting-mechanism error in `3 Practical Guide for Data > 3.2 Finetuning data` P2 S3
   - citations: ['kirkpatrick2017overcoming']
   - refs: Overcoming catastrophic forgetting in neural networks
   - original: Additionally, the absence of a parameter update process ensures that catastrophic forgetting [17] is avoided since the language model parameters remain unaltered.
   - modify to: Catastrophic forgetting is avoided because prompts periodically update a protected copy of the model parameters during inference [17].
   - rationale: Invent a parameter-update mechanism for a statement that depends on parameters remaining unaltered.

### Structural Contradiction
- location: `1 Introduction` P2 S1
- original: To provide a guide for partitioners and end-users, this work focuses on the practical aspects of working with LLMs in downstream NLP tasks.
- modify to: To provide a guide for partitioners and end-users, this work focuses on the practical aspects of working with LLMs in downstream NLP tasks. In addition, this survey systematically covers quantum computing applications and embedded-device compiler optimization with ChatGPT.
- unsupported added scope: quantum computing applications and embedded-device compiler optimization with ChatGPT

### Citation Or Topic Missing
- delete core citations:
  - `brown2020language` (Language models are few-shot learners), mentions=6, sections=['3 Practical Guide for Data > 3.2 Finetuning data', '4 Practical Guide for NLP Tasks > 4.2 Generation tasks > 4.2.1 Use case', '4 Practical Guide for NLP Tasks > 4.4 Abilities Regarding Scaling > 4.4.1 Use Case with Reasoning']
  - `ouyang2022training` (Training language models to follow instructions with human feedback), mentions=5, sections=['3 Practical Guide for Data > 3.3 Test data/user data', '4 Practical Guide for NLP Tasks > 4.6 Real world "tasks"']
  - `openai2023gpt4` (GPT-4 Technical Report), mentions=5, sections=['4 Practical Guide for NLP Tasks > 4.2 Generation tasks > 4.2.1 Use case', '4 Practical Guide for NLP Tasks > 4.3 Knowledge-intensive tasks > 4.3.1 Use case', '4 Practical Guide for NLP Tasks > 4.4 Abilities Regarding Scaling > 4.4.1 Use Case with Reasoning']
  - `scao2022bloom` (Bloom: A 176b-parameter open-access multilingual language model), mentions=3, sections=['3 Practical Guide for Data > 3.1 Pretraining data', '4 Practical Guide for NLP Tasks > 4.2 Generation tasks > 4.2.1 Use case', '4 Practical Guide for NLP Tasks > 4.2 Generation tasks > 4.2.2 No use case']
  - `liang2022holistic` (Holistic evaluation of language models), mentions=3, sections=['4 Practical Guide for NLP Tasks > 4.1 Traditional NLU tasks > 4.1.1 No use case', '4 Practical Guide for NLP Tasks > 4.1 Traditional NLU tasks > 4.1.2 Use case']
  - `wei2022inverse` (Inverse scaling can become U-shaped), mentions=3, sections=['4 Practical Guide for NLP Tasks > 4.4 Abilities Regarding Scaling > 4.4.3 No-Use Cases and Understanding']
- delete subsection: `4 Practical Guide for NLP Tasks > 4.5 Miscellaneous tasks > 4.5.2 Use case`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=13, citation_mentions=13)

## 5. Augmented Language Models: a Survey
- slug: `augmented_language_models_a_survey`
- date/query/arXiv: 2023-02-15 / `augmented language models` / `2302.07842v1`
- eligible middle top sections: [3, 4, 5]
- non-overlap: delete subsection `3 Using Tools and Act > 3.4 Acting on the virtual and physical world`; delete citation keys ['borgeaud2022improving', 'izacard2022atlas', 'lecun2022a', 'nakano2021webgpt', 'ouyang2022training', 'zelikman2022star']; fact citation keys ['brown2020language', 'gao2022pal', 'yang2022doc', 'yang2022re3']

### Factual Errors
1. tool-use mechanism error in `3 Using Tools and Act > 3.3 Computing via Symbolic Modules and Code Interpreters` P1 S8
   - citations: ['gao2022pal']
   - refs: PAL: Program-aided Language Models
   - original: PAL~ [19] relies on CoT prompting of large LMs to decompose symbolic reasoning, mathematical reasoning, or algorithmic tasks into intermediate steps along with python code for each step (see Figure~Figure 6).
   - modify to: PAL~ [19] directly executes natural-language chain-of-thought steps as Python programs, without asking the model to write explicit code for the intermediate reasoning steps.
   - rationale: Collapse PAL's program-generation step into direct execution of natural language.
2. generation-length overclaim in `3 Using Tools and Act > 3.1 Calling another model` P3 S3
   - citations: ['yang2022re3']
   - refs: Re3: Generating Longer Stories With Recursive Reprompting and Revision
   - original: Re3~ [17] exploits this idea to automatically generate stories of over two thousand words.
   - modify to: Re3~ [17] uses the same prompting scheme to reliably generate complete book-length stories rather than long short stories.
   - rationale: Expand the supported generation scale from long stories to book-length generation.
3. model-role error in `3 Using Tools and Act > 3.1 Calling another model` P3 S4
   - citations: ['brown2020language']
   - refs: Language Models are Few-Shot Learners
   - original: More precisely, Re3 first generates a plan, setting, and characters by prompting GPT3~ [2] with a premise.
   - modify to: Re3 first trains GPT-3 from scratch on the story premise before generating the plan, setting, and characters~ [2].
   - rationale: Replace prompting a pretrained model with training a model from scratch.
4. granularity guarantee error in `3 Using Tools and Act > 3.1 Calling another model` P3 S6
   - citations: ['yang2022doc']
   - refs: DOC: Improving Long Story Coherence With Detailed Outline Control
   - original: This work is improved upon in [55] with the use of a learned detailed outliner that iteratively expands the brief initial outline to any desired level of granularity.
   - modify to: The learned detailed outliner in [55] guarantees globally coherent narratives at any requested outline depth without additional revision or filtering.
   - rationale: Turn iterative outline expansion into an unsupported coherence guarantee.

### Structural Contradiction
- location: `1 Introduction: motivation for the survey and definitions > 1.1 Motivation` P3 S1
- original: We argue these issues stem from a fundamental defect of LLMs: they are generally trained to perform statistical language modeling given (i) a single parametric model and (ii) a limited context, typically the n previous or surrounding tokens.
- modify to: We argue these issues stem from a fundamental defect of LLMs: they are generally trained to perform statistical language modeling given (i) a single parametric model and (ii) a limited context, typically the n previous or surrounding tokens. In addition, this survey systematically covers protein-structure wet-lab automation and autonomous laboratory control.
- unsupported added scope: protein-structure wet-lab automation and autonomous laboratory control

### Citation Or Topic Missing
- delete core citations:
  - `ouyang2022training` (Training language models to follow instructions with human feedback), mentions=6, sections=['4 Learning to reason, use tools, and act > 4.1 Supervision', '4 Learning to reason, use tools, and act > 4.2 Reinforcement learning']
  - `lecun2022a` (A Path Towards Autonomous Machine Intelligence), mentions=6, sections=['5 Discussion']
  - `nakano2021webgpt` (WebGPT: Browser-assisted question-answering with human feedback), mentions=5, sections=['3 Using Tools and Act > 3.2 Information retrieval > 3.2.3 Searching and navigating the web', '4 Learning to reason, use tools, and act > 4.1 Supervision', '4 Learning to reason, use tools, and act > 4.2 Reinforcement learning']
  - `borgeaud2022improving` (Improving language models by retrieving from trillions of tokens), mentions=4, sections=['3 Using Tools and Act > 3.2 Information retrieval > 3.2.1 Retrieval-augmented language models', '5 Discussion']
  - `izacard2022atlas` (Atlas: Few-shot Learning with Retrieval Augmented Language Models), mentions=4, sections=['3 Using Tools and Act > 3.2 Information retrieval > 3.2.1 Retrieval-augmented language models', '5 Discussion']
  - `zelikman2022star` (Star: Self-taught reasoner bootstrapping reasoning with reasoning), mentions=4, sections=['4 Learning to reason, use tools, and act > 4.1 Supervision', '4 Learning to reason, use tools, and act > 4.2 Reinforcement learning', '4 Learning to reason, use tools, and act > 4.3 Limitations and future directions']
- delete subsection: `3 Using Tools and Act > 3.4 Acting on the virtual and physical world`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=38, citation_mentions=13)

## 6. A Survey on In-context Learning
- slug: `a_survey_on_in-context_learning`
- date/query/arXiv: 2022-12-31 / `in-context learning` / `2301.00234v6`
- eligible middle top sections: [3, 4, 5, 6]
- non-overlap: delete subsection `5 Analysis > 5.1 Influencing Factors > 5.1.1 Pretraining Stage`; delete citation keys ['Wu2022SelfadaptiveIL', 'kim2022self', 'liu2022close', 'liu2024incontext', 'lu2022order', 'topic']; fact citation keys ['Li2023mend', 'Shi2023iclm', 'flan', 'lamda', 'metaicl', 'natural', 'picl', 'selfsupericl']

### Factual Errors
1. training-objective substitution in `3 Model Training > 3.2 Warmup` P2 S6
   - citations: ['lamda', 'flan']
   - refs: LaMDA: Language Models for Dialog Applications; Finetuned Language Models are Zero-Shot Learners
   - original: Tuning the 137B LaMDA-PT~ [29] on over 60 datasets verbalized via natural language instruction templates, FLAN~ [28] improves the ability of LLMs to follow instructions, boosting both the zero-shot and few-shot ICL performance.
   - modify to: FLAN~ [28] improves ICL by training LaMDA-PT~ [29] to ignore natural-language instructions and rely only on unlabeled continuation data.
   - rationale: Invert instruction tuning into instruction-free language-model continuation.
2. pre-inference training denial in `3 Model Training` P1 S1
   - citations: ['selfsupericl', 'picl', 'Shi2023iclm']
   - refs: Improving In-Context Few-Shot Learning via Self-Supervised Training; Pre-Training to Learn in Context; In-Context Pretraining: Language Modeling Beyond Document Boundaries
   - original: Although LLMs have demonstrated promising ICL capability directly, many studies revealed that these ICL capabilities can be further enhanced through specialized training before inference~ [22, 23, 24].
   - modify to: The cited studies show that specialized training before inference is unnecessary and usually weakens ICL ability~ [22, 23, 24].
   - rationale: Contradict the surveyed finding that specialized training can enhance ICL.
3. efficiency-mechanism error in `3 Model Training > 3.1 Pretraining` P1 S3
   - citations: ['Li2023mend']
   - refs: MEND: Meta dEmonstratioN Distillation for Efficient and Effective In-Context Learning
   - original: Differently, [14] introduced a meta-distillation pretraining process, which allows LLMs to reason with distilled demonstration vectors, thereby enhancing ICL efficiency without compromising its effectiveness.
   - modify to: [14] introduced meta-distillation by storing every demonstration vector at inference time, which improves ICL efficiency through a larger retrieval cache.
   - rationale: Replace distilled demonstration vectors with an inference-time storage/cache mechanism.
4. demonstration-use error in `3 Model Training > 3.2 Warmup` P2 S2
   - citations: ['metaicl', 'natural']
   - refs: MetaICL: Learning to Learn In Context; Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks
   - original: Both [13] and [25] proposed to continually finetune LLMs on a broad range of tasks with multiple demonstration examples, which boosts ICL abilities.
   - modify to: Both [13] and [25] improve ICL by continually finetuning LLMs on tasks without demonstration examples, showing demonstrations are not needed during warmup.
   - rationale: Remove the multiple-demonstration-example condition from the cited warmup methods.

### Structural Contradiction
- location: `1 Introduction` P6 S1
- original: With the rapid growth of studies in ICL, our survey aims to sensitize the community toward the current progress.
- modify to: With the rapid growth of studies in ICL, our survey aims to sensitize the community toward the current progress. In addition, this survey systematically covers privacy-preserving federated in-context learning systems and hardware scheduling.
- unsupported added scope: privacy-preserving federated in-context learning systems and hardware scheduling

### Citation Or Topic Missing
- delete core citations:
  - `liu2022close` (What Makes Good In-Context Examples for GPT-3?), mentions=6, sections=['4 Prompt Designing > 4.1 Demonstration Organization > 4.1.1 Demonstration Selection', '4 Prompt Designing > 4.1 Demonstration Organization > 4.1.3 Demonstration Ordering', '5 Analysis > 5.1 Influencing Factors > 5.1.2 Inference Stage']
  - `lu2022order` (Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity), mentions=4, sections=['4 Prompt Designing > 4.1 Demonstration Organization', '4 Prompt Designing > 4.1 Demonstration Organization > 4.1.3 Demonstration Ordering', '5 Analysis > 5.1 Influencing Factors > 5.1.2 Inference Stage']
  - `Wu2022SelfadaptiveIL` (Self-Adaptive In-Context Learning: An Information Compression Perspective for In-Context Example Selection and Ordering), mentions=3, sections=['4 Prompt Designing > 4.1 Demonstration Organization > 4.1.1 Demonstration Selection']
  - `topic` (Large Language Models Are Implicitly Topic Models: Explaining and Finding Good Demonstrations for In-Context Learning), mentions=3, sections=['4 Prompt Designing > 4.1 Demonstration Organization > 4.1.1 Demonstration Selection', '5 Analysis > 5.2 Learning Mechanism > 5.2.2 Theoretical Interpretation']
  - `kim2022self` (Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator), mentions=3, sections=['4 Prompt Designing > 4.1 Demonstration Organization > 4.1.2 Demonstration Reformatting', '6 Application']
  - `liu2024incontext` (In-context Vectors: Making In Context Learning More Effective and Controllable Through Latent Space Steering), mentions=3, sections=['4 Prompt Designing > 4.1 Demonstration Organization > 4.1.2 Demonstration Reformatting']
- delete subsection: `5 Analysis > 5.1 Influencing Factors > 5.1.1 Pretraining Stage`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=10, citation_mentions=13)

## 7. Towards Reasoning in Large Language Models: A Survey
- slug: `towards_reasoning_in_large_language_models_a_survey`
- date/query/arXiv: 2022-12-20 / `reasoning large language models` / `2212.10403v2`
- eligible middle top sections: [3, 4, 5, 6]
- non-overlap: delete subsection `None`; delete citation keys ['anil2022exploring', 'dan2021@measuring', 'han2022folio', 'kojima2022large', 'suzgun2022challenging', 'zhou2022least']; fact citation keys ['bommasani2021opportunities', 'cobbe2021training', 'liu2019roberta', 'radford2018improving', 'rae2021scaling', 'rajani2019explain', 'talmor2019commonsenseqa', 'talmor2020leap', 'valmeekam2022large', 'wei2022chain', 'wei2022emergent']

### Factual Errors
1. emergence-threshold overclaim in `3 Towards Reasoning in Large Language Models` P1 S2
   - citations: ['wei2022emergent', 'wei2022chain', 'cobbe2021training']
   - refs: Emergent abilities of large language models; Chain of Thought Prompting Elicits Reasoning in Large Language Models; Training verifiers to solve math word problems
   - original: Recent research has suggested that reasoning ability may emerge in language models at a certain scale, such as models with over 100 billion parameters [8, 9, 21].
   - modify to: Recent research establishes reasoning ability as a guaranteed property of any language model once it is trained with a sufficiently large corpus, regardless of parameter scale [8, 9, 21].
   - rationale: Change a scale-associated emergence observation into a universal training-data guarantee.
2. weakness-to-strength error in `3 Towards Reasoning in Large Language Models` P1 S1
   - citations: ['bommasani2021opportunities', 'rae2021scaling', 'valmeekam2022large']
   - refs: On the opportunities and risks of foundation models; Scaling language models: Methods, analysis & insights from training gopher; Large Language Models Still Can't Plan (A Benchmark for LLMs on Planning and Reasoning about Change)
   - original: Reasoning, particularly multi-step reasoning, is often seen as a weakness in language models and other NLP models [14, 29, 15].
   - modify to: Reasoning, particularly multi-step reasoning, is often presented as the strongest and most reliable capability of standard language models before any reasoning-specific prompting [14, 29, 15].
   - rationale: Invert the survey's framing of multi-step reasoning as a weakness.
3. dataset-purpose error in `3 Towards Reasoning in Large Language Models > 3.1 Fully Supervised Finetuning` P1 S2
   - citations: ['rajani2019explain', 'radford2018improving', 'talmor2019commonsenseqa']
   - refs: Explain Yourself! Leveraging Language Models for Commonsense Reasoning; Improving language understanding by generative pre-training; CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge
   - original: For example, [32] finetune a pretrained GPT model~ [33] to generate rationales that explain model predictions with the built CoS-E dataset, and find that models trained with explanations perform better on commonsense question answering tasks~ [34].
   - modify to: [32] finetunes a pretrained GPT model~ [33] to generate adversarial distractors for CoS-E, rather than rationales explaining model predictions on commonsense QA~ [34].
   - rationale: Change rationale generation into adversarial distractor generation.
4. knowledge-source error in `3 Towards Reasoning in Large Language Models > 3.1 Fully Supervised Finetuning` P1 S3
   - citations: ['talmor2020leap', 'liu2019roberta']
   - refs: Leap-Of-Thought: Teaching Pre-Trained Models to Systematically Reason Over Implicit Knowledge; Roberta: A robustly optimized bert pretraining approach
   - original: [35] train RoBERTa~ [36] to perform reasoning/inference based on both implicit pre-trained knowledge and explicit free-text statements.
   - modify to: [35] trains RoBERTa~ [36] to ignore free-text statements and perform reasoning only from implicit pre-trained knowledge.
   - rationale: Remove the explicit free-text statement component from the cited method.

### Structural Contradiction
- location: `1 Introduction` P3 S5
- original: Therefore, in this paper, we aim to provide a comprehensive overview and engage in an insightful discussion on the current state of knowledge on this fast-evolving topic.
- modify to: Therefore, in this paper, we aim to provide a comprehensive overview and engage in an insightful discussion on the current state of knowledge on this fast-evolving topic. In addition, this survey systematically covers formal verification of deployed autonomous-vehicle controllers using LLM reasoning.
- unsupported added scope: formal verification of deployed autonomous-vehicle controllers using LLM reasoning

### Citation Or Topic Missing
- delete core citations:
  - `anil2022exploring` (Exploring length generalization in large language models), mentions=5, sections=['3 Towards Reasoning in Large Language Models > 3.3 Hybrid Method > 3.3.1 Reasoning-Enhanced Training and Prompting', '4 Measuring Reasoning in Large Language Models > 4.1 End Task Performance', '5 Findings and Implications']
  - `suzgun2022challenging` (Challenging BIG-Bench tasks and whether chain-of-thought can solve them), mentions=4, sections=['5 Findings and Implications', '6 Reflection, Discussion, and Future Directions']
  - `kojima2022large` (Large Language Models are Zero-Shot Reasoners), mentions=3, sections=['3 Towards Reasoning in Large Language Models > 3.2 Prompting & In-Context Learning > 3.2.1 Chain of Thought and Its Variants', '3 Towards Reasoning in Large Language Models > 3.2 Prompting & In-Context Learning > 3.2.2 Rationale Engineering', '4 Measuring Reasoning in Large Language Models > 4.2 Analysis on Reasoning']
  - `zhou2022least` (Least-to-Most Prompting Enables Complex Reasoning in Large Language Models), mentions=3, sections=['3 Towards Reasoning in Large Language Models > 3.2 Prompting & In-Context Learning > 3.2.3 Problem Decomposition', '5 Findings and Implications', '6 Reflection, Discussion, and Future Directions']
  - `han2022folio` (Folio: Natural language reasoning with first-order logic), mentions=3, sections=['4 Measuring Reasoning in Large Language Models > 4.2 Analysis on Reasoning', '5 Findings and Implications', '6 Reflection, Discussion, and Future Directions']
  - `dan2021@measuring` (Measuring Mathematical Problem Solving With the MATH Dataset), mentions=2, sections=['3 Towards Reasoning in Large Language Models > 3.1 Fully Supervised Finetuning', '4 Measuring Reasoning in Large Language Models > 4.1 End Task Performance']
- delete subsection: no qualifying middle method/content subsection found after excluding first two sections, last two sections, conclusion/future/appendix

## 8. Multimodal Learning with Transformers: A Survey
- slug: `multimodal_learning_with_transformers_a_survey`
- date/query/arXiv: 2022-06-13 / `multimodal learning transformers` / `2206.06488v2`
- eligible middle top sections: [3, 4, 5]
- non-overlap: delete subsection `4 Application Scenarios > 4.2 Transformers for Specific Multimodal Tasks`; delete citation keys ['chen2020uniter', 'li2020hero', 'lu2019vilbert', 'sun2019videobert', 'tan2019lxmert', 'zhan2021product1m']; fact citation keys ['ba2016layer', 'bronstein2021geometric', 'devlin2018bert', 'dosovitskiy2020image', 'dwivedi2020generalization', 'he2016deep', 'ioffe2015batch']

### Factual Errors
1. token-function swap in `3 Transformers > 3.1 Transformer > 3.1.1 Input Tokenization` P7 S1
   - citations: ['devlin2018bert', 'dosovitskiy2020image']
   - refs: Bert: Pre-training of deep bidirectional transformers for language understanding; An image is worth 16x16 words: Transformers for image recognition at scale
   - original: (3) Tokenization is compatible with the task-specific customized tokens,, [MASK] token for Masked Language Modelling, [CLASS] token for classification.
   - modify to: Task-specific customized tokens use [MASK] for classification decisions and [CLASS] for masked language modelling.
   - rationale: Swap the roles of common Transformer special tokens.
2. architecture-topology error in `3 Transformers` P1 S2
   - citations: ['bronstein2021geometric', 'dwivedi2020generalization']
   - refs: Geometric deep learning: Grids, groups, graphs, geodesics, and gauges; A generalization of transformer networks to graphs
   - original: We highlight that Transformers can be understood from a geometrically topological perspective, because due to the self-attention mechanism, given each tokenized input from any modalities, self-attention (Transformer) can model it as a fully-connected graph in topological geometry space.
   - modify to: Transformers are topologically equivalent to sparse chain graphs because self-attention connects each token only to its immediate neighbors.
   - rationale: Change full self-attention connectivity into local-chain connectivity.
3. residual-purpose error in `3 Transformers > 3.1 Transformer` P1 S5
   - citations: ['he2016deep']
   - refs: Deep residual learning for image recognition
   - original: To help the back propagation of the gradient, both MHSA and FFN use Residual Connection (given an input x, the residual connection of any mapping f( ) is defined as x f(x) + x ), followed by normalization layer.
   - modify to: Residual connections in MHSA and FFN are introduced to block gradient propagation through the sub-layer output before normalization.
   - rationale: Invert the purpose of residual connections from helping to blocking gradient flow.
4. normalization substitution in `3 Transformers > 3.1 Transformer` P1 S6
   - citations: ['ioffe2015batch', 'ba2016layer']
   - refs: Batch normalization: Accelerating deep network training by reducing internal covariate shift; Layer normalization
   - original: Thus, assuming that the input tensor is, the output of MHSA and FFN sub-layers can be formulated as: N ( sublayer ( ) + ), where sublayer( ) is the mapping implemented by the sub-layer itself and N( ) denotes normalization,, BN( ), LN( ).
   - modify to: The normalization term N( ) in Transformer sub-layers denotes a learned attention mask rather than batch or layer normalization.
   - rationale: Misidentify the normalization operation as an attention mask.

### Structural Contradiction
- location: `1 Introduction` P16 S1
- original: In this survey, we focus solely on the intersection of multimodal learning and Transformers.
- modify to: In this survey, we focus solely on the intersection of multimodal learning and Transformers. In addition, this survey systematically covers blockchain consensus protocols and secure smart-contract verification.
- unsupported added scope: blockchain consensus protocols and secure smart-contract verification

### Citation Or Topic Missing
- delete core citations:
  - `lu2019vilbert` (Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks), mentions=24, sections=['3 Transformers > 3.2 Vision Transformer', '3 Transformers > 3.3 Multimodal Transformers > 3.3.1 Input', '3 Transformers > 3.3 Multimodal Transformers > 3.3.2 Self-Attention Variants in Context']
  - `sun2019videobert` (Videobert: A joint model for video and language representation learning), mentions=21, sections=['3 Transformers > 3.2 Vision Transformer', '3 Transformers > 3.3 Multimodal Transformers > 3.3.1 Input', '3 Transformers > 3.3 Multimodal Transformers > 3.3.2 Self-Attention Variants in Context']
  - `chen2020uniter` (Uniter: Universal image-text representation learning), mentions=20, sections=['3 Transformers > 3.3 Multimodal Transformers > 3.3.3 Network Architectures', '4 Application Scenarios > 4.1 Transformers for Multimodal Pretraining', '4 Application Scenarios > 4.1 Transformers for Multimodal Pretraining > 4.1.1 Task-Agnostic Multimodal Pretraining']
  - `tan2019lxmert` (Lxmert: Learning cross-modality encoder representations from transformers), mentions=17, sections=['3 Transformers > 3.2 Vision Transformer', '3 Transformers > 3.3 Multimodal Transformers > 3.3.3 Network Architectures', '4 Application Scenarios > 4.1 Transformers for Multimodal Pretraining']
  - `zhan2021product1m` (Product1m: Towards weakly supervised instance-level product retrieval via cross-modal pretraining), mentions=16, sections=['3 Transformers > 3.3 Multimodal Transformers > 3.3.1 Input', '3 Transformers > 3.3 Multimodal Transformers > 3.3.2 Self-Attention Variants in Context', '3 Transformers > 3.3 Multimodal Transformers > 3.3.3 Network Architectures']
  - `li2020hero` (Hero: Hierarchical encoder for video+ language omni-representation pre-training), mentions=16, sections=['4 Application Scenarios > 4.1 Transformers for Multimodal Pretraining > 4.1.1 Task-Agnostic Multimodal Pretraining', '4 Application Scenarios > 4.1 Transformers for Multimodal Pretraining > 4.1.2 Task-Specific Multimodal Pretraining']
- delete subsection: `4 Application Scenarios > 4.2 Transformers for Specific Multimodal Tasks`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=3, citation_mentions=38)

## 9. Image Data Augmentation for Deep Learning: A Survey
- slug: `image_data_augmentation_for_deep_learning_a_survey`
- date/query/arXiv: 2022-04-19 / `image data augmentation deep learning` / `2204.08610v2`
- eligible middle top sections: [3, 4]
- non-overlap: delete subsection `3 Advanced Approaches > 3.3 Deep Generative Models`; delete citation keys ['OHL', 'PBA', 'aug_improving', 'feature_aug', 'keepaugment', 'randaugment']; fact citation keys ['autoaugment', 'dplabv3+', 'fast_autoaugment', 'gcnet', 'isanet', 'pspnet', 'reinforcement']

### Factual Errors
1. metric substitution in `4 Evaluation > 4.1 Semantic Segmentation` P1 S2
   - citations: ['dplabv3+', 'pspnet', 'gcnet', 'isanet']
   - refs: A survey of recommendation systems based on deep learning; Pyramid Scene Parsing Network; GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond; Interlaced sparse self-attention for semantic segmentation
   - original: In table~Table 2, we report the performance improvement on Intersection over Union(IoU) metric with several semantic segmentation models: deeplabv3+~ [44], PSPNet~ [45], GCNet~ [46], and ISANet~ [47].
   - modify to: The semantic segmentation comparison in Table~Table 2 reports improvements in classification accuracy rather than Intersection over Union (IoU) for Deeplabv3+, PSPNet, GCNet, and ISANet~ [44-47].
   - rationale: Replace the evaluated segmentation metric with an unrelated classification metric.
2. search-objective error in `3 Advanced Approaches > 3.1 Auto Augment` P1 S5
   - citations: ['autoaugment']
   - refs: Autoaugment: Learning augmentation strategies from data
   - original: [26] describes a simple procedure called AutoAugment to automatically search for improved data augmentation policies.
   - modify to: [26] describes AutoAugment as a procedure for manually selecting a fixed augmentation policy, rather than automatically searching for improved policies.
   - rationale: Remove AutoAugment's automatic search component.
3. cost-source error in `3 Advanced Approaches > 3.1 Auto Augment` P1 S10
   - citations: ['reinforcement']
   - refs: Reinforcement learning: An introduction
   - original: The search algorithm usually uses Reinforcement Learning~ [27], which brings high time cost.
   - modify to: The high time cost of AutoAugment comes from applying each augmentation at inference time, not from reinforcement-learning policy search~ [27].
   - rationale: Move the computational cost from training-time search to inference-time augmentation.
4. method-mechanism substitution in `3 Advanced Approaches > 3.1 Auto Augment` P1 S11
   - citations: ['fast_autoaugment']
   - refs: Fast autoaugment
   - original: Therefore, to reduce the time cost of AutoAugment, ~ [28] proposes Fast AutoAugment that finds effective augmentation policies via a more efficient search strategy based on density matching.
   - modify to: Fast AutoAugment~ [28] reduces time cost by replacing augmentation search with random crops sampled uniformly from the training set.
   - rationale: Replace density-matching policy search with a simple random-crop procedure.

### Structural Contradiction
- location: `1 Introduction` P1 S1
- original: Deep learning has made incredible progress in many fields, including computer vision(CV)~ [1], recommender system (RS)~ [2], natural language processing (NLP)~ [3] and so on.
- modify to: Deep learning has made incredible progress in many fields, including computer vision(CV)~ [1], recommender system (RS)~ [2], natural language processing (NLP)~ [3] and so on. In addition, this survey systematically covers reinforcement-learning policy optimization and robot navigation benchmarks.
- unsupported added scope: reinforcement-learning policy optimization and robot navigation benchmarks

### Citation Or Topic Missing
- delete core citations:
  - `PBA` (Population based augmentation: Efficient learning of augmentation policy schedules), mentions=1, sections=['3 Advanced Approaches > 3.1 Auto Augment']
  - `randaugment` (Randaugment: Practical automated data augmentation with a reduced search space), mentions=1, sections=['3 Advanced Approaches > 3.1 Auto Augment']
  - `keepaugment` (KeepAugment: A Simple Information-Preserving Data Augmentation Approach), mentions=1, sections=['3 Advanced Approaches > 3.1 Auto Augment']
  - `aug_improving` (Improving Auto-Augment via Augmentation-Wise Weight Sharing), mentions=1, sections=['3 Advanced Approaches > 3.1 Auto Augment']
  - `OHL` (Online hyper-parameter learning for auto-augmentation strategy), mentions=1, sections=['3 Advanced Approaches > 3.1 Auto Augment']
  - `feature_aug` (Dataset augmentation in feature space), mentions=1, sections=['3 Advanced Approaches > 3.2 Feature Augmentation']
- delete subsection: `3 Advanced Approaches > 3.3 Deep Generative Models`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=23, citation_mentions=7)

## 10. Transformers in Time Series: A Survey
- slug: `transformers_in_time_series_a_survey`
- date/query/arXiv: 2022-02-15 / `transformers time series` / `2202.07125v5`
- eligible middle top sections: [3, 4, 5, 6]
- non-overlap: delete subsection `5 Applications of Time Series Transformers > 5.1 Transformers in Forecasting > 5.1.2 Spatio-Temporal Forecasting`; delete citation keys ['Nie2022ATS', 'lin2021ssdnet', 'liu2022pyraformer', 'xu2021autoformer', 'xu2022anomalyTrans', 'zhou2022fedformer']; fact citation keys ['li2019enhancing', 'lim2021temporal', 'vaswani2017attention', 'zerveas2021transformer', 'zhou2021informer']

### Factual Errors
1. encoding-placement error in `4 Network Modifications for Time Series > 4.1 Positional Encoding` P2 S1
   - citations: ['li2019enhancing', 'vaswani2017attention']
   - refs: Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting; Attention is all you need
   - original: Vanilla Positional Encoding. A few works [8] simply introduce vanilla positional encoding (Section Section 2.2.1) used in [1], which is then added to the input time series embeddings and fed to Transformer.
   - modify to: The cited works add vanilla positional encoding after the Transformer layers, so temporal order is injected only into the final prediction head [8, 1].
   - rationale: Move positional encoding from input embeddings to the output head.
2. parameter-learning error in `4 Network Modifications for Time Series > 4.1 Positional Encoding` P3 S3
   - citations: ['zerveas2021transformer']
   - refs: A transformer-based framework for multivariate time series representation learning
   - original: ~ [12] introduces an embedding layer in Transformer that learns embedding vectors for each position index jointly with other model parameters.
   - modify to: [12] freezes a single sinusoidal vector for all positions rather than learning position-index embeddings jointly with the Transformer parameters.
   - rationale: Replace learnable position embeddings with a frozen shared vector.
3. model-component error in `4 Network Modifications for Time Series > 4.1 Positional Encoding` P3 S4
   - citations: ['lim2021temporal']
   - refs: Temporal fusion transformers for interpretable multi-horizon time series forecasting
   - original: [27] uses an LSTM network to encode positional embeddings, which can better exploit sequential ordering information in time series.
   - modify to: [27] uses a convolutional autoencoder, not an LSTM network, to encode positional embeddings for sequential ordering information.
   - rationale: Swap the cited recurrent positional encoder for a different architecture.
4. timestamp-role error in `4 Network Modifications for Time Series > 4.1 Positional Encoding` P4 S3
   - citations: ['zhou2021informer']
   - refs: Informer: Beyond efficient transformer for long sequence time-series forecasting
   - original: To mitigate the issue, Informer~ [28] proposed to encode timestamps as additional positional encoding by using learnable embedding layers.
   - modify to: Informer~ [28] removes timestamp information from the input and relies solely on value embeddings to mitigate positional encoding issues.
   - rationale: Invert Informer's timestamp-embedding design.

### Structural Contradiction
- location: `1 Introduction` P3 S1
- original: In this paper, we aim to fill the gap by summarizing the main developments of time series Transformers.
- modify to: In this paper, we aim to fill the gap by summarizing the main developments of time series Transformers. In addition, this survey systematically covers medical image segmentation transformers and radiology workflow deployment.
- unsupported added scope: medical image segmentation transformers and radiology workflow deployment

### Citation Or Topic Missing
- delete core citations:
  - `liu2022pyraformer` (Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting), mentions=5, sections=['4 Network Modifications for Time Series > 4.2 Attention Module', '4 Network Modifications for Time Series > 4.3 Architecture-based Attention Innovation', '5 Applications of Time Series Transformers > 5.1 Transformers in Forecasting > 5.1.1 Time Series Forecasting']
  - `xu2021autoformer` (Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting), mentions=4, sections=['4 Network Modifications for Time Series > 4.1 Positional Encoding', '5 Applications of Time Series Transformers > 5.1 Transformers in Forecasting > 5.1.1 Time Series Forecasting', '6 Experimental Evaluation and Discussion']
  - `zhou2022fedformer` (FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting), mentions=4, sections=['4 Network Modifications for Time Series > 4.1 Positional Encoding', '4 Network Modifications for Time Series > 4.2 Attention Module', '5 Applications of Time Series Transformers > 5.1 Transformers in Forecasting > 5.1.1 Time Series Forecasting']
  - `lin2021ssdnet` (SSDNet: State Space Decomposition Neural Network for Time Series Forecasting), mentions=2, sections=['5 Applications of Time Series Transformers > 5.1 Transformers in Forecasting > 5.1.1 Time Series Forecasting', '6 Experimental Evaluation and Discussion']
  - `Nie2022ATS` (A Time Series is Worth 64 Words: Long-term Forecasting with Transformers), mentions=2, sections=['5 Applications of Time Series Transformers > 5.1 Transformers in Forecasting > 5.1.1 Time Series Forecasting']
  - `xu2022anomalyTrans` (Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy), mentions=2, sections=['5 Applications of Time Series Transformers > 5.2 Transformers in Anomaly Detection']
- delete subsection: `5 Applications of Time Series Transformers > 5.1 Transformers in Forecasting > 5.1.2 Spatio-Temporal Forecasting`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=9, citation_mentions=5)

## 11. Survey of Hallucination in Natural Language Generation
- slug: `survey_of_hallucination_in_natural_language_generation`
- date/query/arXiv: 2022-02-08 / `hallucination natural language generation` / `2202.03629v7`
- eligible middle top sections: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
- non-overlap: delete subsection `9 Hallucination in Generative Question Answering > 9.3 Hallucination Mitigation in GQA`; delete citation keys ['Raunak2021', 'dziri2021neural', 'lee2018hallucinations', 'shuster2021retrieval', 'tian2020sticking', 'zhou2021detecting']; fact citation keys ['dhingra2019handling', 'lebret2016neural', 'lee2021deduplicating', 'parikh2020totto', 'wang2019revisiting', 'wiseman2017challenges']

### Factual Errors
1. annotation-source error in `3 Contributors to Hallucination in NLG > 3.1 Hallucination from Data` P2 S6
   - citations: ['dhingra2019handling']
   - refs: Handling Divergent Reference Texts when Evaluating Table-to-Text Generation
   - original: Indeed, [39] points out that 62 of the first sentences in WIKIBIO have additional information not stated in the corresponding infobox.
   - modify to: [39] points out that the unsupported information in WIKIBIO first sentences mainly comes from annotation mistakes in the infoboxes, not from the target references containing extra facts.
   - rationale: Misattribute the source of hallucination from target-reference extra information to infobox annotation errors.
2. dataset-pairing error in `3 Contributors to Hallucination in NLG > 3.1 Hallucination from Data` P2 S2
   - citations: ['lebret2016neural', 'wiseman2017challenges']
   - refs: Neural Text Generation from Structured Data with Application to the Biography Domain; Challenges in Data-to-Document Generation
   - original: When collecting large-scale datasets, some works heuristically select and pair real sentences or tables as the source and target~ [36, 37].
   - modify to: When collecting large-scale datasets, the cited works pair synthetic template sentences with tables instead of selecting and pairing real sentences or tables~ [36, 37].
   - rationale: Change the dataset construction procedure from heuristic real-pair selection to synthetic templating.
3. support-direction error in `3 Contributors to Hallucination in NLG > 3.1 Hallucination from Data` P2 S3
   - citations: ['wang2019revisiting', 'parikh2020totto']
   - refs: Revisiting Challenges in Data-to-Text Generation with Fact Grounding; ToTTo: A Controlled Table-To-Text Generation Dataset
   - original: As a result, the target reference may contain information that cannot be supported by the source~ [38, 22].
   - modify to: The target reference is used as evidence to verify the source table, so unsupported target information is removed before training~ [38, 22].
   - rationale: Reverse the support relationship between source and target reference.
4. memorization-effect error in `3 Contributors to Hallucination in NLG > 3.1 Hallucination from Data` P3 S3
   - citations: ['lee2021deduplicating']
   - refs: Deduplicating Training Data Makes Language Models Better
   - original: [40] show that duplicated examples from the pretraining corpus bias the model to favor generating repeats of the memorized phrases from the duplicated examples.
   - modify to: [40] shows that duplicated pretraining examples make models avoid memorized phrases and therefore reduce repeated generations.
   - rationale: Invert the reported bias introduced by duplicated examples.

### Structural Contradiction
- location: `1 Introduction` P5 S3
- original: Section~Section 2 ∼ Section~Section 6 provide an overview of the hallucination problem in NLG by discussing the definition and categorization, contributors, metrics, and mitigation methods of hallucinations, respectively.
- modify to: Section~Section 2 ∼ Section~Section 6 provide an overview of the hallucination problem in NLG by discussing the definition and categorization, contributors, metrics, and mitigation methods of hallucinations, respectively. In addition, this survey systematically covers hardware-level mitigation of hallucination on edge accelerators.
- unsupported added scope: hardware-level mitigation of hallucination on edge accelerators

### Citation Or Topic Missing
- delete core citations:
  - `zhou2021detecting` (Detecting Hallucinated Content in Conditional Neural Sequence Generation), mentions=12, sections=['11 Hallucinations in Neural Machine Translation > 11.1 Hallucinations Definition and Categories in NMT', '11 Hallucinations in Neural Machine Translation > 11.2 Hallucination Metrics in NMT', '11 Hallucinations in Neural Machine Translation > 11.2 Hallucination Metrics in NMT > 11.2.2 Model-Based Metrics']
  - `lee2018hallucinations` (Hallucinations in Neural Machine Translation), mentions=10, sections=['11 Hallucinations in Neural Machine Translation > 11.1 Hallucinations Definition and Categories in NMT', '11 Hallucinations in Neural Machine Translation > 11.2 Hallucination Metrics in NMT > 11.2.2 Model-Based Metrics', '11 Hallucinations in Neural Machine Translation > 11.3 Hallucination Mitigation Methods in NMT > 11.3.1 Data-Related']
  - `shuster2021retrieval` (Retrieval Augmentation Reduces Hallucination in Conversation), mentions=9, sections=['3 Contributors to Hallucination in NLG > 3.2 Hallucination from Training and Inference', '4 Metrics Measuring Hallucination > 4.1 Statistical Metric', '4 Metrics Measuring Hallucination > 4.3 Human Evaluation']
  - `Raunak2021` (The Curious Case of Hallucinations in Neural Machine Translation), mentions=8, sections=['11 Hallucinations in Neural Machine Translation > 11.1 Hallucinations Definition and Categories in NMT', '11 Hallucinations in Neural Machine Translation > 11.2 Hallucination Metrics in NMT > 11.2.2 Model-Based Metrics', '11 Hallucinations in Neural Machine Translation > 11.3 Hallucination Mitigation Methods in NMT > 11.3.1 Data-Related']
  - `dziri2021neural` (Neural Path Hunter: Reducing Hallucination in Dialogue Systems via Path Grounding), mentions=8, sections=['3 Contributors to Hallucination in NLG > 3.2 Hallucination from Training and Inference', '4 Metrics Measuring Hallucination > 4.3 Human Evaluation', '5 Hallucination Mitigation Methods > 5.2 Modeling and Inference Methods > 5.2.3 Post-Processing']
  - `tian2020sticking` (Sticking to the Facts: Confident Decoding for Faithful Data-to-Text Generation), mentions=7, sections=['10 Hallucination in Data-to-Text Generation > 10.2 Hallucination Metrics in Data-to-Text Generation', '10 Hallucination in Data-to-Text Generation > 10.3 Hallucination Mitigation in Data-to-Text Generation', '11 Hallucinations in Neural Machine Translation > 11.2 Hallucination Metrics in NMT']
- delete subsection: `9 Hallucination in Generative Question Answering > 9.3 Hallucination Mitigation in GQA`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=18, citation_mentions=13)

## 12. Transformers in Medical Imaging: A Survey
- slug: `transformers_in_medical_imaging_a_survey`
- date/query/arXiv: 2022-01-24 / `Transformers Medical Imaging` / `2201.09873v1`
- eligible middle top sections: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
- non-overlap: delete subsection `10 Open Challenges and Future Directions > 10.3 Adversarial Robustness`; delete citation keys ['chefer2021transformer', 'demner2016preparing', 'johnson2019mimic', 'mccollough2017low', 'napel2014nsclc', 'selvaraju2017grad']; fact citation keys ['ai2020correlation', 'chen2021can', 'fang2020sensitivity', 'perera2021pocformer', 'sandler2018mobilenetv2', 'wang2020linformer']

### Factual Errors
1. diagnostic-modality error in `3 Medical Image Classification > 3.1 COVID-19 Diagnosis` P1 S1
   - citations: ['ai2020correlation', 'fang2020sensitivity', 'chen2021can']
   - refs: Correlation of chest CT and RT-PCR testing for coronavirus disease 2019 (COVID-19) in China: a report of 1014 cases; Sensitivity of chest CT for COVID-19: comparison to RT-PCR; Can chest CT features distinguish patients with negative from those with positive initial RT-PCR results for coronavirus disease (COVID-19)?
   - original: Studies suggest that COVID-19 can potentially be better diagnosed with radiological imaging as compared to tedious real-time polymerase chain reaction (RT-PCR) test [145, 146, 147].
   - modify to: Studies suggest that COVID-19 is better diagnosed with radiological imaging because RT-PCR cannot detect active infections at all [145, 146, 147].
   - rationale: Turn a comparative diagnostic-efficiency claim into a false impossibility claim about RT-PCR.
2. input-modality error in `3 Medical Image Classification > 3.1 COVID-19 Diagnosis > 3.1.1 Black-Box Models` P2 S2
   - citations: ['perera2021pocformer']
   - refs: POCFormer: A Lightweight Transformer Architecture for Detection of COVID-19 Using Point of Care Ultrasound
   - original: Perera et al. [148] propose a lightweight Point-of-Care Transformer (POCFormer) to diagnose COVID-19 from lungs images captured via portable devices.
   - modify to: Perera et al. [148] propose POCFormer to diagnose COVID-19 from wearable audio recordings rather than lung images captured by portable devices.
   - rationale: Replace the medical image input modality with audio.
3. complexity-reduction error in `3 Medical Image Classification > 3.1 COVID-19 Diagnosis > 3.1.1 Black-Box Models` P2 S3
   - citations: ['wang2020linformer']
   - refs: Linformer: Self-attention with linear complexity
   - original: Specifically, POCFormer leverages Linformer [129] to reduce the space and time complexity of self-attention from quadratic to linear.
   - modify to: POCFormer leverages Linformer [129] to increase self-attention to quadratic complexity so that small lesions receive denser attention.
   - rationale: Invert Linformer's role in reducing attention complexity.
4. deployment-constraint error in `3 Medical Image Classification > 3.1 COVID-19 Diagnosis > 3.1.1 Black-Box Models` P2 S4
   - citations: ['sandler2018mobilenetv2']
   - refs: Mobilenetv2: Inverted residuals and linear bottlenecks
   - original: POCFormer has two million parameters that are about half of MobileNetv2~ [149], thus making it suitable for real-time diagnosis.
   - modify to: POCFormer has two million parameters but is considered unsuitable for real-time diagnosis because it is larger than MobileNetv2~ [149].
   - rationale: Reverse the size comparison and deployment implication.

### Structural Contradiction
- location: `1 Background` P1 S2
- original: In this section, we briefly provide a background of these advancements and broadly group them into three categories: hand-crafted, CNN-based, and ViT-based.
- modify to: In this section, we briefly provide a background of these advancements and broadly group them into three categories: hand-crafted, CNN-based, and ViT-based. In addition, this survey systematically covers machine translation, dialogue summarization, and text-only language modeling.
- unsupported added scope: machine translation, dialogue summarization, and text-only language modeling

### Citation Or Topic Missing
- delete core citations:
  - `demner2016preparing` (Preparing a collection of radiology examinations for distribution and retrieval), mentions=12, sections=['8 Clinical report generation > 8.1 Reinforcement Learning Based Approaches', '8 Clinical report generation > 8.2 Supervised and Unsupervised Approaches > 8.2.1 Dataset Bias', '8 Clinical report generation > 8.2 Supervised and Unsupervised Approaches > 8.2.2 Feature Alignment']
  - `chefer2021transformer` (Transformer interpretability beyond attention visualization), mentions=4, sections=['10 Open Challenges and Future Directions > 10.2 Interpretability', '3 Medical Image Classification > 3.1 COVID-19 Diagnosis > 3.1.2 Interpretable Models', '3 Medical Image Classification > 3.2 Tumor Classification']
  - `napel2014nsclc` (NSCLC radiogenomics: initial Stanford study of 26 cases), mentions=4, sections=['3 Medical Image Classification > 3.2 Tumor Classification']
  - `mccollough2017low` (Low-dose CT for the detection and classification of metastatic liver lesions: results of the 2016 low dose CT grand challenge), mentions=4, sections=['5 Medical Image Reconstruction > 5.1 Medical Image Enhancement > 5.1.1 LDCT Enhancement', '5 Medical Image Reconstruction > 5.2 Medical Image Restoration > 5.2.2 Sparse-View CT Reconstruction']
  - `johnson2019mimic` (MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs), mentions=4, sections=['8 Clinical report generation > 8.2 Supervised and Unsupervised Approaches > 8.2.1 Dataset Bias', '8 Clinical report generation > 8.2 Supervised and Unsupervised Approaches > 8.2.2 Feature Alignment', '8 Clinical report generation > 8.2 Supervised and Unsupervised Approaches > 8.2.4 Miscellaneous']
  - `selvaraju2017grad` (Grad-cam: Visual explanations from deep networks via gradient-based localization), mentions=3, sections=['10 Open Challenges and Future Directions > 10.2 Interpretability', '3 Medical Image Classification > 3.1 COVID-19 Diagnosis > 3.1.2 Interpretable Models']
- delete subsection: `10 Open Challenges and Future Directions > 10.3 Adversarial Robustness`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=14, citation_mentions=14)

## 13. Generalized Out-of-Distribution Detection: A Survey 

- slug: `generalized_out-of-distribution_detection_a_survey`
- date/query/arXiv: 2021-10-21 / `generalized out-of-distribution detection` / `2110.11334v3`
- eligible middle top sections: [3, 4]
- non-overlap: delete subsection `None`; delete citation keys ['hendrycks2019using', 'jia2022visual', 'nearood21arxiv', 'pretransformer20arxiv', 'zhou2022cocoop', 'zhou2022coop']; fact citation keys ['bitterwolf2023or', 'bommasani2021opportunities', 'cultrera2023leveraging', 'du2022unknown', 'du2022vos', 'koh2021wilds', 'radford2021learning', 'yang2022openood']

### Factual Errors
1. benchmark-correction error in `3 Response to Reviewer #3` P4 S3
   - citations: ['bitterwolf2023or']
   - refs: In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation
   - original: We add the related discussion on page 19: Additionally, recent research~ [4] highlights the presence of erroneous samples in ImageNet OOD benchmarks and introduces the corrected NINCO dataset for more accurate evaluations.
   - modify to: Recent research~ [4] validates ImageNet OOD benchmarks as error-free and therefore argues against using corrected datasets such as NINCO.
   - rationale: Invert the cited benchmark-correction motivation.
2. benchmark-scope narrowing in `3 Response to Reviewer #3` P4 S4
   - citations: ['koh2021wilds', 'cultrera2023leveraging', 'du2022vos', 'du2022unknown']
   - refs: Wilds: A benchmark of in-the-wild distribution shifts; Leveraging Visual Attention for out-of-Distribution Detection; VOS: Learning What You Don’t Know by Virtual Outlier Synthesis; Unknown-Aware Object Detection: Learning What You Don’t Know from Videos in the Wild
   - original: Furthermore, expanding the scope of benchmarks to encompass real-world scenarios, such as more realistic datasets~ [5, 6], and object-level OOD detection~ [7, 8], can provide valuable insights, especially in safety-critical applications like autonomous driving.
   - modify to: The cited benchmark work recommends avoiding real-world datasets and object-level OOD detection because they provide little insight for safety-critical applications~ [5-8].
   - rationale: Turn benchmark expansion into benchmark narrowing.
3. tool-purpose error in `3 Response to Reviewer #3` P6 S1
   - citations: ['yang2022openood']
   - refs: OpenOOD: Benchmarking Generalized Out-of-Distribution Detection
   - original: Accompanying our survey, we lead the development of OpenOOD~ [9], an open-source codebase that provides a unified framework and benchmarking platform for conducting fair comparisons of various model architectures and OOD detection methods.
   - modify to: OpenOOD~ [9] is described as a private leaderboard for a single model family rather than an open-source framework for fair comparisons.
   - rationale: Misstate OpenOOD's openness and comparison scope.
4. foundation-model robustness error in `3 Response to Reviewer #3` P8 S1
   - citations: ['bommasani2021opportunities', 'radford2021learning']
   - refs: On the opportunities and risks of foundation models; Learning transferable visual models from natural language supervision
   - original: Foundation models~ [10], notably large-scale vision-language models~ [11], have demonstrated exceptional performance in a variety of downstream tasks.
   - modify to: Foundation models~ [10], including large-scale vision-language models~ [11], are reported to fail on most downstream tasks unless trained from scratch for each dataset.
   - rationale: Replace broad downstream strength with a false from-scratch requirement.

### Structural Contradiction
- location: `4 Response to Reviewer #4` P6 S2
- original: We have accordingly revised our manuscript to include a more comprehensive analysis of how foundation models can be leveraged in OOD detection, exploring their potential and the challenges they present.
- modify to: We have accordingly revised our manuscript to include a more comprehensive analysis of how foundation models can be leveraged in OOD detection, exploring their potential and the challenges they present. In addition, this survey systematically covers cryptographic protocol verification and secure multiparty computation.
- unsupported added scope: cryptographic protocol verification and secure multiparty computation

### Citation Or Topic Missing
- delete core citations:
  - `hendrycks2019using` (Using pre-training can improve model robustness and uncertainty), mentions=2, sections=['3 Response to Reviewer #3', '4 Response to Reviewer #4']
  - `nearood21arxiv` (Exploring the Limits of Out-of-Distribution Detection), mentions=2, sections=['3 Response to Reviewer #3', '4 Response to Reviewer #4']
  - `pretransformer20arxiv` (Pretrained transformers improve out-of-distribution robustness), mentions=2, sections=['3 Response to Reviewer #3', '4 Response to Reviewer #4']
  - `zhou2022coop` (Learning to Prompt for Vision-Language Models), mentions=2, sections=['3 Response to Reviewer #3', '4 Response to Reviewer #4']
  - `zhou2022cocoop` (Conditional Prompt Learning for Vision-Language Models), mentions=2, sections=['3 Response to Reviewer #3', '4 Response to Reviewer #4']
  - `jia2022visual` (Visual prompt tuning), mentions=2, sections=['3 Response to Reviewer #3', '4 Response to Reviewer #4']
- delete subsection: no qualifying middle method/content subsection found after excluding first two sections, last two sections, conclusion/future/appendix

## 14. A Survey on Multi-modal Summarization
- slug: `a_survey_on_multi-modal_summarization`
- date/query/arXiv: 2021-09-11 / `multimodal summarization` / `2109.05199v2`
- eligible middle top sections: [3, 4, 5, 6]
- non-overlap: delete subsection `4 Overview of Methods > 4.1 Main Model`; delete citation keys ['chen2018abstractive', 'jangra2020multimodal', 'jangra2020text', 'jangra2021multimodal', 'li2017multi', 'li2018multi']; fact citation keys ['erol2003multimodal', 'evangelopoulos2013multimodal', 'ma2020multidocument', 'mikolov2013distributed', 'pennington2014glove', 'salton1989automatic', 'tjondronegoro2011multi', 'zhu2018msmo', 'zhu3multimodal']

### Factual Errors
1. feature-source error in `3 Organization of existing work > 3.1 On the basis of encoding the input > 3.1.1 Feature Extraction Strategies` P2 S4
   - citations: ['zhu2018msmo', 'zhu3multimodal']
   - refs: MSMO: Multimodal Summarization with Multimodal Output; Multimodal Summarization with Guidance of Multimodal Reference
   - original: Some works also train similar embeddings on their own datasets [3, 87] (refer to Feature Extraction in Section Section 4.1.1).
   - modify to: The cited works use only generic off-the-shelf embeddings and do not train similar embeddings on their own multimodal summarization datasets [3, 87].
   - rationale: Remove the dataset-specific embedding training described in the sentence.
2. difficulty-source error in `3 Organization of existing work > 3.1 On the basis of encoding the input` P3 S3
   - citations: ['ma2020multidocument']
   - refs: Multi-document Summarization via Deep Learning Techniques: A Survey
   - original: Having multiple documents makes the task a lot more challenging, since the degree of redundant information in input becomes a lot more prominent, making the data somewhat more noisy [79].
   - modify to: Having multiple documents makes summarization easier because redundant input information reliably filters out noise [79].
   - rationale: Invert redundancy/noise from a challenge into an automatic benefit.
3. feature-method error in `3 Organization of existing work > 3.1 On the basis of encoding the input > 3.1.1 Feature Extraction Strategies` P2 S1
   - citations: ['salton1989automatic', 'erol2003multimodal', 'tjondronegoro2011multi', 'evangelopoulos2013multimodal']
   - refs: Automatic text processing: The transformation, analysis, and retrieval of; Multimodal summarization of meeting recordings; Multi-modal summarization of key events and top players in sports tournament videos; Multimodal saliency and fusion for movie summarization based on aural, visual, and textual attention
   - original: Text: Traditionally, before the era of deep learning, Term Frequency-Document Inverse Frequency (TF-IDF) [84] was used to identify relevant text segments [74, 80, 75].
   - modify to: Before deep learning, TF-IDF [84] was primarily used to generate visual scene descriptors rather than identify relevant text segments [74, 80, 75].
   - rationale: Move TF-IDF from text relevance estimation to visual feature generation.
4. embedding-coverage overclaim in `3 Organization of existing work > 3.1 On the basis of encoding the input > 3.1.1 Feature Extraction Strategies` P2 S2
   - citations: ['mikolov2013distributed', 'pennington2014glove']
   - refs: Distributed representations of words and phrases and their compositionality; Glove: Global vectors for word representation
   - original: Due to significant advancements in feature extraction, almost all the MMS tasks in the past five years either use pre-trained embeddings like word2vec [85] or Glove [86].
   - modify to: In the past five years, every MMS task uses both word2vec [85] and GloVe [86] together, with no task relying on other text representations.
   - rationale: Turn an “almost all use pre-trained embeddings like” statement into a universal two-embedding requirement.

### Structural Contradiction
- location: `1 Introduction` P6 S1
- original: The rest of the paper is structured as follows: We formally define the MMS task in Section Section 2.
- modify to: The rest of the paper is structured as follows: We formally define the MMS task in Section Section 2. In addition, this survey systematically covers time-series forecasting and financial anomaly detection.
- unsupported added scope: time-series forecasting and financial anomaly detection

### Citation Or Topic Missing
- delete core citations:
  - `li2017multi` (Multi-modal summarization for asynchronous collection of text, image, audio and video), mentions=33, sections=['3 Organization of existing work > 3.1 On the basis of encoding the input', '3 Organization of existing work > 3.1 On the basis of encoding the input > 3.1.1 Feature Extraction Strategies', '3 Organization of existing work > 3.2 On the basis of method']
  - `jangra2020text` (Text-Image-Video Summary Generation Using Joint Integer Linear Programming), mentions=26, sections=['3 Organization of existing work > 3.1 On the basis of encoding the input', '3 Organization of existing work > 3.1 On the basis of encoding the input > 3.1.1 Feature Extraction Strategies', '3 Organization of existing work > 3.2 On the basis of method']
  - `jangra2021multimodal` (Multi-Modal Supplementary-Complementary Summarization Using Multi-Objective Optimization), mentions=26, sections=['3 Organization of existing work > 3.1 On the basis of encoding the input', '3 Organization of existing work > 3.1 On the basis of encoding the input > 3.1.1 Feature Extraction Strategies', '3 Organization of existing work > 3.2 On the basis of method']
  - `chen2018abstractive` (Abstractive Text-Image Summarization Using Multi-Modal Attentional Hierarchical RNN), mentions=22, sections=['3 Organization of existing work > 3.1 On the basis of encoding the input', '3 Organization of existing work > 3.1 On the basis of encoding the input > 3.1.1 Feature Extraction Strategies', '3 Organization of existing work > 3.2 On the basis of method']
  - `jangra2020multimodal` (Multi-Modal Summary Generation Using Multi-Objective Optimization), mentions=22, sections=['3 Organization of existing work > 3.1 On the basis of encoding the input', '3 Organization of existing work > 3.1 On the basis of encoding the input > 3.1.1 Feature Extraction Strategies', '3 Organization of existing work > 3.2 On the basis of method']
  - `li2018multi` (Multi-modal Sentence Summarization with Modality Attention and Image Filtering), mentions=21, sections=['3 Organization of existing work > 3.1 On the basis of encoding the input', '3 Organization of existing work > 3.1 On the basis of encoding the input > 3.1.1 Feature Extraction Strategies', '3 Organization of existing work > 3.2 On the basis of method']
- delete subsection: `4 Overview of Methods > 4.1 Main Model`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=4, citation_mentions=0)

## 15. Towards Efficient Synchronous Federated Training: A Survey on System Optimization Strategies
- slug: `towards_efficient_synchronous_federated_training_a_survey_on_system_optimization_strategies`
- date/query/arXiv: 2021-09-09 / `synchronous federated training system optimization` / `2109.03999v3`
- eligible middle top sections: [3, 4]
- non-overlap: delete subsection `None`; delete citation keys ['beutel2020flower', 'caldas2018leaf', 'hu2020oarf', 'lai2021fedscale', 'yang2019federated', 'yang2021characterizing']; fact citation keys ['acar2021federated', 'al2020federated', 'gu2019distributed', 'hsieh2020non', 'hsu2019measuring', 'kairouz2019advances', 'krizhevsky2009learning', 'mcmahan2017communication', 'reddi2020adaptive', 'wan2013regularization', 'xiao2017fashion']

### Factual Errors
1. taxonomy substitution in `4 Related Work and Concluding Remarks > 4.1 Related Surveys` P2 S2
   - citations: ['gu2019distributed']
   - refs: Distributed machine learning on mobile devices: A survey
   - original: In~ [119], the authors discuss the realm of mobile distributed machine learning, where algorithms are classified into three categories: 1) machine learning optimizers, 2) distributed optimization algorithms, and 3) data aggregation methods.
   - modify to: In~ [119], mobile distributed machine learning algorithms are classified into personalization, privacy attacks, and incentive mechanisms rather than optimizers, distributed optimization, and aggregation methods.
   - rationale: Replace the cited taxonomy with unrelated FL topics.
2. dataset-origin error in `3 Measurement and Benchmarking Tools > 3.2 Benchmarking Suites > 3.2.1 Training Datasets` P1 S2
   - citations: ['krizhevsky2009learning', 'wan2013regularization', 'xiao2017fashion']
   - refs: Learning multiple layers of features from tiny images; Regularization of neural networks using dropconnect; Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms
   - original: One line of work is derived from conventional ML datasets (e.g., ~ [103], ~ [104], and ~ [105] ).
   - modify to: The conventional ML datasets cited here are collected from real federated mobile clients rather than adapted from centralized benchmark datasets~ [103-105].
   - rationale: Misstate the origin of conventional benchmark datasets.
3. non-IID synthesis error in `3 Measurement and Benchmarking Tools > 3.2 Benchmarking Suites > 3.2.1 Training Datasets` P1 S3
   - citations: ['mcmahan2017communication', 'hsu2019measuring', 'reddi2020adaptive', 'al2020federated', 'acar2021federated']
   - refs: Communication-efficient learning of deep networks from decentralized data; Measuring the effects of non-identical data distribution for federated visual classification; Adaptive federated optimization; Federated Learning via Posterior Averaging: A New Perspective and Practical Algorithms; Federated learning based on dynamic regularization
   - original: To synthesize the non-IID nature as in real FL scenarios, the data partitions in these datasets are typically formed by restricting the number of data classes each client has (e.g., partitioning by shard-based methods as in~ [1] or latent Dirichlet allocation (LDA) processes as in~ [97, 99, 72, 81] ).
   - modify to: The cited partitioning methods synthesize IID client data by ensuring every client has the same class proportions through shard and LDA procedures~ [1, 97, 99, 72, 81].
   - rationale: Invert non-IID partitioning into IID balancing.
4. heterogeneity-type omission in `3 Measurement and Benchmarking Tools > 3.2 Benchmarking Suites > 3.2.1 Training Datasets` P1 S5
   - citations: ['hsieh2020non', 'kairouz2019advances']
   - refs: The non-iid data quagmire of decentralized machine learning; Advances and open problems in federated learning
   - original: For instance, besides the label distribution skew, in reality, non-IID data may also involve feature distribution skew (e.g., same words with different stroke widths), same labels with different features (e.g., images of clothing vary due to regional differences) and same features with different labels (e.g., the same context mapped to different next words due to personal habits)~ [106, 2].
   - modify to: The cited discussion treats label distribution skew as the only realistic non-IID form and excludes feature skew or same-feature-different-label cases~ [106, 2].
   - rationale: Delete the additional non-IID cases that the original sentence includes.

### Structural Contradiction
- location: `1 Background, Problem and Challenges` P1 S1
- original: In this section, we give a detailed introduction to the system optimization problem in federated training.
- modify to: In this section, we give a detailed introduction to the system optimization problem in federated training. In addition, this survey systematically covers neural radiance fields and 3D scene reconstruction.
- unsupported added scope: neural radiance fields and 3D scene reconstruction

### Citation Or Topic Missing
- delete core citations:
  - `lai2021fedscale` (FedScale: Benchmarking Model and System Performance of Federated Learning), mentions=4, sections=['3 Measurement and Benchmarking Tools > 3.2 Benchmarking Suites > 3.2.1 Training Datasets', '3 Measurement and Benchmarking Tools > 3.2 Benchmarking Suites > 3.2.2 Production Systems and Simulation Platforms', '4 Related Work and Concluding Remarks > 4.2 Future Research Directions > 4.2.2 On the Configuration Phase']
  - `yang2021characterizing` (Characterizing Impacts of Heterogeneity in Federated Learning upon Large-Scale Smartphone Data), mentions=3, sections=['3 Measurement and Benchmarking Tools > 3.1 Measurement-Based Research', '3 Measurement and Benchmarking Tools > 3.2 Benchmarking Suites > 3.2.2 Production Systems and Simulation Platforms', '4 Related Work and Concluding Remarks > 4.3 Discussion > 4.3.1 Cross-Device FL and Cross-Silo FL']
  - `yang2019federated` (Federated machine learning: Concept and applications), mentions=3, sections=['4 Related Work and Concluding Remarks > 4.3 Discussion > 4.3.2 Horizontal FL and Vertical FL']
  - `caldas2018leaf` (Leaf: A benchmark for federated settings), mentions=2, sections=['3 Measurement and Benchmarking Tools > 3.2 Benchmarking Suites > 3.2.1 Training Datasets', '3 Measurement and Benchmarking Tools > 3.2 Benchmarking Suites > 3.2.2 Production Systems and Simulation Platforms']
  - `hu2020oarf` (The oarf benchmark suite: Characterization and implications for federated learning systems), mentions=2, sections=['3 Measurement and Benchmarking Tools > 3.2 Benchmarking Suites > 3.2.1 Training Datasets', '3 Measurement and Benchmarking Tools > 3.2 Benchmarking Suites > 3.2.2 Production Systems and Simulation Platforms']
  - `beutel2020flower` (Flower: A friendly federated learning research framework), mentions=2, sections=['3 Measurement and Benchmarking Tools > 3.2 Benchmarking Suites > 3.2.2 Production Systems and Simulation Platforms']
- delete subsection: no qualifying middle method/content subsection found after excluding first two sections, last two sections, conclusion/future/appendix

## 16. Asynchronous Federated Learning on Heterogeneous Devices: A Survey
- slug: `asynchronous_federated_learning_on_heterogeneous_devices_a_survey`
- date/query/arXiv: 2021-09-09 / `asynchronous federated learning heterogeneous devices` / `2109.04269v5`
- eligible middle top sections: [3, 4, 5, 6]
- non-overlap: delete subsection `5 Privacy and Security on Heterogeneous Devices > 5.2 Security on Heterogeneous Devices`; delete citation keys ['fadlullah2020hcp', 'imteaj2020fedar', 'lu2019differentially', 'sun2020adaptive', 'wang2021efficient', 'xiaofeng2020asynchronous']; fact citation keys ['chen2021towards', 'hao2020time', 'lu2020privacy', 'zhou2021tea']

### Factual Errors
1. module-function swap in `3 Device Heterogeneity > 3.3 Gradient Compression` P2 S1
   - citations: ['lu2020privacy']
   - refs: Privacy-preserving asynchronous federated learning mechanism for edge network computing
   - original: For instance, in~ [55], two sub-modules are presented for self-adaptive threshold gradient compression: (1) self-adaptive threshold computation and (2) gradient communication compression.
   - modify to: In~ [55], the two sub-modules first compress gradients and then compute a fixed threshold from the compressed messages.
   - rationale: Reverse the adaptive threshold computation and compression roles.
2. selection-criterion error in `3 Device Heterogeneity > 3.1 Node Selection` P2 S1
   - citations: ['chen2021towards']
   - refs: Towards asynchronous federated learning for heterogeneous edge-powered internet of things
   - original: For instance, in~ [44], the authors present a heuristic greedy node selection strategy that iteratively selects heterogeneous IoT nodes to participate in global learning aggregation based on their local computing and communication resources.
   - modify to: The heuristic greedy node selection strategy in~ [44] selects IoT nodes randomly and does not use local computing or communication resources.
   - rationale: Remove the resource-aware criterion from node selection.
3. parallelism-direction error in `3 Device Heterogeneity > 3.1 Node Selection` P2 S3
   - citations: ['zhou2021tea']
   - refs: TEA-fed: time-efficient asynchronous federated learning for edge computing
   - original: Apart from that, considering the large number of edge devices involved, in~ [45], the authors limit the number of devices training simultaneously in the AFL network.
   - modify to: In~ [45], the authors increase the number of devices training simultaneously to maximize AFL parallelism under heterogeneity.
   - rationale: Invert the cited limit on simultaneous training devices.
4. priority-signal error in `3 Device Heterogeneity > 3.1 Node Selection` P3 S1
   - citations: ['hao2020time']
   - refs: Time Efficient Federated Learning with Semi-asynchronous Communication
   - original: In order to select nodes more reasonably, in~ [46], a prioritized node-selecting function is designed according to the computing power and accuracy change of local models on each node.
   - modify to: The prioritized node-selecting function in~ [46] is designed only from device battery level and ignores local model accuracy changes.
   - rationale: Replace the cited computing-power/accuracy-change signals with an unsupported battery-only rule.

### Structural Contradiction
- location: `1 Introduction` P5 S2
- original: Consequently, the primary significance of this study lies in its comprehensive classification, summarization, and analysis of AFL.
- modify to: Consequently, the primary significance of this study lies in its comprehensive classification, summarization, and analysis of AFL. In addition, this survey systematically covers text-to-image diffusion sampling and prompt engineering.
- unsupported added scope: text-to-image diffusion sampling and prompt engineering

### Citation Or Topic Missing
- delete core citations:
  - `wang2021efficient` (Efficient federated learning for fault diagnosis in industrial cloud-edge computing), mentions=5, sections=['3 Device Heterogeneity', '3 Device Heterogeneity > 3.2 Weighted Aggregation', '3 Device Heterogeneity > 3.6 Model Splitting']
  - `xiaofeng2020asynchronous` (An Asynchronous Federated Learning Mechanism for Edge Network Computing), mentions=5, sections=['3 Device Heterogeneity', '3 Device Heterogeneity > 3.2 Weighted Aggregation', '3 Device Heterogeneity > 3.3 Gradient Compression']
  - `fadlullah2020hcp` (HCP: Heterogeneous computing platform for federated learning based collaborative content caching towards 6G networks), mentions=4, sections=['3 Device Heterogeneity', '3 Device Heterogeneity > 3.6 Model Splitting', '6 Applications on Heterogeneous Devices']
  - `imteaj2020fedar` (Fedar: Activity and resource-aware federated learning model for distributed mobile robots), mentions=4, sections=['3 Device Heterogeneity', '3 Device Heterogeneity > 3.1 Node Selection', '6 Applications on Heterogeneous Devices']
  - `sun2020adaptive` (Adaptive federated learning and digital twin for industrial internet of things), mentions=4, sections=['3 Device Heterogeneity', '3 Device Heterogeneity > 3.5 Cluster FL', '6 Applications on Heterogeneous Devices']
  - `lu2019differentially` (Differentially private asynchronous federated learning for mobile edge computing in urban informatics), mentions=4, sections=['5 Privacy and Security on Heterogeneous Devices', '5 Privacy and Security on Heterogeneous Devices > 5.1 Privacy on Heterogeneous Devices', '6 Applications on Heterogeneous Devices']
- delete subsection: `5 Privacy and Security on Heterogeneous Devices > 5.2 Security on Heterogeneous Devices`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=41, citation_mentions=9)

## 17. A Survey of Deep Reinforcement Learning in Recommender Systems: A Systematic Review and Future Directions
- slug: `a_survey_of_deep_reinforcement_learning_in_recommender_systems_a_systematic_review_and_future_directions`
- date/query/arXiv: 2021-09-08 / `deep reinforcement learning recommender systems` / `2109.03540v2`
- eligible middle top sections: [3, 4, 5]
- non-overlap: delete subsection `3 Deep Reinforcement Learning in Recommender Systems > 3.3 Component Optimization in Deep Reinforcement Learning based RS > 3.3.1 Environment Simulation and Reconstruction`; delete citation keys ['chen2019generative', 'he2020learning', 'liu2020top', 'wang2021reinforcement', 'xie2021hierarchical', 'zou2020pseudo']; fact citation keys ['bai2019model', 'goodfellow2014generative', 'hong2020nonintrusive', 'liu2020end', 'liu2020state']

### Factual Errors
1. representation-method error in `3 Deep Reinforcement Learning in Recommender Systems > 3.3 Component Optimization in Deep Reinforcement Learning based RS > 3.3.2 State Representation` P1 S4
   - citations: ['liu2020end', 'liu2020state']
   - refs: End-to-end deep reinforcement learning based recommendation with supervised embedding; State representation modeling for deep reinforcement learning based recommendation
   - original: [79, 80] propose a supervised learning method to generate a better state representation by utilizing an attention mechanism and a pooling operation as shown in~Figure 6.
   - modify to: [79, 80] generate state representations with unsupervised clustering alone, without using attention mechanisms or pooling operations.
   - rationale: Replace the supervised attention/pooling state-representation method with clustering.
2. training-objective error in `3 Deep Reinforcement Learning in Recommender Systems > 3.1 Model-based Deep Reinforcement Learning based Methods` P3 S2
   - citations: ['bai2019model']
   - refs: A Model-Based Reinforcement Learning with Adversarial Training for Online Recommendation
   - original: IRecGAN~ [12] is a model-based method that adopts generative adversarial training to improve the robustness of policy learning.
   - modify to: IRecGAN~ [12] is a model-based method that removes adversarial training and improves robustness only through hand-crafted transition rules.
   - rationale: Remove the generative-adversarial component from IRecGAN.
3. GAN-purpose error in `3 Deep Reinforcement Learning in Recommender Systems > 3.1 Model-based Deep Reinforcement Learning based Methods` P3 S4
   - citations: ['goodfellow2014generative']
   - refs: Generative adversarial networks
   - original: IRecGAN employs a generative adversarial network~ [13] to generate user data based on the offline dataset.
   - modify to: IRecGAN employs a generative adversarial network~ [13] to delete noisy users from the offline dataset rather than generate user data.
   - rationale: Change the GAN role from data generation to user deletion.
4. recommendation-domain error in `3 Deep Reinforcement Learning in Recommender Systems > 3.1 Model-based Deep Reinforcement Learning based Methods` P4 S1
   - citations: ['hong2020nonintrusive']
   - refs: Nonintrusive-Sensing and Reinforcement-Learning Based Adaptive Personalized Music Recommendation
   - original: [14] propose NRSS for personalized music recommendation.
   - modify to: [14] propose NRSS for personalized news recommendation rather than personalized music recommendation.
   - rationale: Swap the application domain of NRSS.

### Structural Contradiction
- location: `1 Introduction` P1 S7
- original: In light of the significance and recent progresses in DRL for recommender sytsems, we aim to timely summaize and comment on DRL-based recommendation systems in this survey.
- modify to: In light of the significance and recent progresses in DRL for recommender sytsems, we aim to timely summaize and comment on DRL-based recommendation systems in this survey. In addition, this survey systematically covers medical image registration and molecular docking.
- unsupported added scope: medical image registration and molecular docking

### Citation Or Topic Missing
- delete core citations:
  - `zou2020pseudo` (Pseudo Dyna-Q: A reinforcement learning framework for interactive recommendation), mentions=6, sections=['3 Deep Reinforcement Learning in Recommender Systems > 3.1 Model-based Deep Reinforcement Learning based Methods', '5 Open Questions > 5.2 Exploration and Exploitation', '5 Open Questions > 5.4 Bias (Unfairness)']
  - `he2020learning` (Learning to Collaborate in Multi-Module Recommendation via Multi-Agent Reinforcement Learning without Communication), mentions=6, sections=['3 Deep Reinforcement Learning in Recommender Systems > 3.2 Model-free deep reinforcement learning based methods', '4 Emerging Topics > 4.1 Multi-Agent and Hierarchical Deep Reinforcement Learning-based RS', '5 Open Questions > 5.2 Exploration and Exploitation']
  - `chen2019generative` (Generative adversarial user model for reinforcement learning based recommendation system), mentions=5, sections=['3 Deep Reinforcement Learning in Recommender Systems > 3.1 Model-based Deep Reinforcement Learning based Methods', '5 Open Questions > 5.2 Exploration and Exploitation']
  - `wang2021reinforcement` (Reinforcement Learning with a Disentangled Universal Value Function for Item Recommendation), mentions=5, sections=['3 Deep Reinforcement Learning in Recommender Systems > 3.1 Model-based Deep Reinforcement Learning based Methods', '5 Open Questions > 5.2 Exploration and Exploitation']
  - `liu2020top` (Top-aware reinforcement learning based recommendation), mentions=5, sections=['3 Deep Reinforcement Learning in Recommender Systems > 3.2 Model-free deep reinforcement learning based methods', '4 Emerging Topics > 4.4 Self-Supervised DRL-based RS', '5 Open Questions > 5.2 Exploration and Exploitation']
  - `xie2021hierarchical` (Hierarchical Reinforcement Learning for Integrated Recommendation), mentions=5, sections=['3 Deep Reinforcement Learning in Recommender Systems > 3.2 Model-free deep reinforcement learning based methods', '4 Emerging Topics > 4.1 Multi-Agent and Hierarchical Deep Reinforcement Learning-based RS', '5 Open Questions > 5.2 Exploration and Exploitation']
- delete subsection: `3 Deep Reinforcement Learning in Recommender Systems > 3.3 Component Optimization in Deep Reinforcement Learning based RS > 3.3.1 Environment Simulation and Reconstruction`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=33, citation_mentions=9)

## 18. A Survey of Exploration Methods in Reinforcement Learning
- slug: `a_survey_of_exploration_methods_in_reinforcement_learning`
- date/query/arXiv: 2021-09-01 / `exploration methods reinforcement learning` / `2109.00157v2`
- eligible middle top sections: [3, 4, 5, 6, 7, 8]
- non-overlap: delete subsection `5 Randomized Action Selection > 5.2 Policy-Search Based Methods > 5.2.3 Parameter-space perturbing strategies`; delete citation keys ['bellemare2016unifying', 'brafman2002r', 'pathak2017curiosity', 'schmidhuber1991curious', 'schmidhuber1991possibility', 'williams1992simple']; fact citation keys ['barto1991real', 'bubeck2009pure', 'even2002convergence', 'moore1990efficient', 'mozer1989discovering', 'schmidhuber1990making', 'sutton1990integrated', 'thrun1992efficient', 'tokic2010adaptive', 'tokic2011value']

### Factual Errors
1. reward-use error in `4 Reward-Free Exploration > 4.1 Blind Exploration` P4 S3
   - citations: ['even2002convergence', 'tokic2010adaptive', 'tokic2011value']
   - refs: Convergence of optimistic and incremental Q-learning; Adaptive $\varepsilon$-greedy exploration in reinforcement learning based on value differences; Value-difference based exploration: adaptive control between epsilon-greedy and softmax
   - original: The methods [32, 33, 34], which incorporate extrinsic rewards in their exploratory decision making, are discussed in detail in section Section 5.1.
   - modify to: The methods [32, 33, 34] are reward-free approaches that deliberately exclude extrinsic rewards from exploratory decision making.
   - rationale: Invert the reward usage of the referenced methods.
2. historical-role error in `3 Categorization of Exploratory Techniques` P1 S4
   - citations: ['mozer1989discovering', 'sutton1990integrated', 'moore1990efficient', 'schmidhuber1990making', 'barto1991real']
   - refs: Discovering the structure of a reactive environment by exploration; Integrated architectures for learning, planning, and reacting based on approximating dynamic programming; Efficient memory-based learning for robot control; Making the world differentiable: On using self-supervised fully recurrent neural networks for dynamic reinforcement learning and planning in non-stationary environments; Real-time learning and control using asynchronous dynamic programming
   - original: Some of the early studies that acknowledged the importance of efficient exploration in this context were delivered by [6, 7, 8, 9] and [10].
   - modify to: The early studies [6-10] argued that efficient exploration was unnecessary for reinforcement learning and could be replaced by exploitation-only policies.
   - rationale: Misrepresent early exploration studies as rejecting exploration.
3. taxonomy-author error in `3 Categorization of Exploratory Techniques` P3 S1
   - citations: ['thrun1992efficient']
   - refs: Efficient exploration in reinforcement learning
   - original: One of the first general categorization of the exploration methods was introduced by [2].
   - modify to: [2] introduced a taxonomy limited to value-function approximation methods, not a general categorization of exploration methods.
   - rationale: Narrow the scope of the cited categorization.
4. pure-exploration setting error in `4 Reward-Free Exploration` P1 S2
   - citations: ['bubeck2009pure']
   - refs: Pure exploration in multi-armed bandits problems
   - original: This type of exploration methods was first introduced and utilized with the name pure exploration in multi-armed bandits, a set of sequential decision-making tasks where at each time step, an agent pulls an arm and receives a random reward drawn from the reward distribution of that specific arm~ [24].
   - modify to: Pure exploration was first introduced for deterministic planning tasks where the agent receives no random rewards from arms~ [24].
   - rationale: Move pure exploration from stochastic multi-armed bandits to deterministic planning.

### Structural Contradiction
- location: `1 Introduction` P2 S1
- original: Our goal in this survey is to provide a broad high-level overview on the types of exploration methods employed by RL agents, by reviewing literature from the last three decades.
- modify to: Our goal in this survey is to provide a broad high-level overview on the types of exploration methods employed by RL agents, by reviewing literature from the last three decades. In addition, this survey systematically covers database transaction processing and SQL query optimization.
- unsupported added scope: database transaction processing and SQL query optimization

### Citation Or Topic Missing
- delete core citations:
  - `schmidhuber1991curious` (Curious model-building control systems), mentions=10, sections=['4 Reward-Free Exploration > 4.2 Intrinsically-Motivated Exploration', '6 Bonus-Based/Optimism-Based Exploration > 6.3 Prediction error-based bonus']
  - `schmidhuber1991possibility` (A possibility for implementing curiosity and boredom in model-building neural controllers), mentions=10, sections=['4 Reward-Free Exploration > 4.2 Intrinsically-Motivated Exploration', '6 Bonus-Based/Optimism-Based Exploration > 6.3 Prediction error-based bonus']
  - `williams1992simple` (Simple statistical gradient-following algorithms for connectionist reinforcement learning), mentions=9, sections=['5 Randomized Action Selection > 5.2 Policy-Search Based Methods', '5 Randomized Action Selection > 5.2 Policy-Search Based Methods > 5.2.2 Action-space perturbing strategies', '5 Randomized Action Selection > 5.2 Policy-Search Based Methods > 5.2.4 The distribution of perturbations']
  - `bellemare2016unifying` (Unifying count-based exploration and intrinsic motivation), mentions=9, sections=['6 Bonus-Based/Optimism-Based Exploration > 6.1 Optimism-based methods > 6.1.2 Optimism-based methods: function approximation', '6 Bonus-Based/Optimism-Based Exploration > 6.2 Count-based bonus', '6 Bonus-Based/Optimism-Based Exploration > 6.2 Count-based bonus > 6.2.2 Count-based: function approximation']
  - `brafman2002r` (R-max-a general polynomial time algorithm for near-optimal reinforcement learning), mentions=8, sections=['6 Bonus-Based/Optimism-Based Exploration > 6.1 Optimism-based methods', '6 Bonus-Based/Optimism-Based Exploration > 6.1 Optimism-based methods > 6.1.1 Optimism-based methods: tabular', '6 Bonus-Based/Optimism-Based Exploration > 6.1 Optimism-based methods > 6.1.2 Optimism-based methods: function approximation']
  - `pathak2017curiosity` (Curiosity-driven exploration by self-supervised prediction), mentions=7, sections=['4 Reward-Free Exploration > 4.2 Intrinsically-Motivated Exploration', '6 Bonus-Based/Optimism-Based Exploration > 6.3 Prediction error-based bonus', '7 Deliberate Exploration > 7.3 Meta-learning']
- delete subsection: `5 Randomized Action Selection > 5.2 Policy-Search Based Methods > 5.2.3 Parameter-space perturbing strategies`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=70, citation_mentions=40)

## 19. Survey of Low-Resource Machine Translation
- slug: `survey_of_low-resource_machine_translation`
- date/query/arXiv: 2021-09-01 / `low-resource machine translation` / `2109.00486v3`
- eligible middle top sections: [3, 4, 5]
- non-overlap: delete subsection `5 Use of external resources and linguistic information > 5.2 Incorporating Lexicons`; delete citation keys ['burlot-etal-2017-word', 'garciamartinez:hal-01433161', 'luong2016multitask', 'niehues-cho-2017-exploiting', 'sennrich-haddow-2016-linguistic', 'zaremoodi2018adaptive']; fact citation keys ['baziotis2020language', 'clinchant-etal-2019-use', 'devlin2018bert', 'gulcehre2017integrating', 'lample2019cross', 'liu2020multilingual', 'radford2018improving', 'radfordlanguage', 'sennrich2016improving', 'yang2020making']

### Factual Errors
1. pretraining-technique error in `3 Use of monolingual data > 3.4 Pre-training` P1 S1
   - citations: ['devlin2018bert', 'radford2018improving', 'radfordlanguage']
   - refs: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding; Improving language understanding with unsupervised learning; radfordlanguage
   - original: There has been a great deal of interest in the machine learning and natural language processing communities in using pre-training techniques such as BERT~ [16] and GPT/GPT-2~ [34, 35].
   - modify to: BERT~ [16] and GPT/GPT-2~ [34, 35] are described as bilingual dictionary induction methods rather than pre-training techniques.
   - rationale: Replace pre-trained language models with dictionary-induction methods.
2. objective-use error in `3 Use of monolingual data` P2 S2
   - citations: ['lample2019cross', 'clinchant-etal-2019-use', 'liu2020multilingual', 'yang2020making']
   - refs: Cross-lingual Language Model Pretraining; On the use of BERT for Neural Machine Translation; Multilingual Denoising Pre-training for Neural Machine Translation; Towards Making the Most of BERT in Neural Machine Translation
   - original: It can also be used to pretrain model parameters through initialising and fine-tuning with an MT objective [17, 18, 19] or through knowledge distillation [20].
   - modify to: The cited works use monolingual data only for vocabulary construction and never initialize or fine-tune model parameters with an MT objective [17-19].
   - rationale: Remove parameter pretraining/fine-tuning from monolingual-data use.
3. language-model role error in `3 Use of monolingual data > 3.1 Language Models` P1 S1
   - citations: ['gulcehre2017integrating', 'baziotis2020language']
   - refs: On integrating a language model into neural machine translation; Language Model Prior for Low-Resource Neural Machine Translation
   - original: Inspired by standard statistical MT models, which incorporated language models to target target language fluency, some of the first works making use of monolingual data in NMT focused on exploiting it to train language models, which could be integrated into the learning objective [10, 11].
   - modify to: The first NMT works using monolingual data trained language models for the source language only, so they could not improve target-language fluency [10, 11].
   - rationale: Invert the target-language-fluency purpose of language models.
4. backtranslation-direction error in `3 Use of monolingual data > 3.2 Data Augmentation` P2 S1
   - citations: ['sennrich2016improving']
   - refs: Improving Neural Machine Translation Models with Monolingual Data
   - original: By far the most succesful approach to data augmenation is backtranslation~ [23], where a model trained in the opposite direction to the one you are interested in translates monolingual data in the target back into the source.
   - modify to: Backtranslation~ [23] translates source-side monolingual data into the target language using the same forward model being trained.
   - rationale: Reverse the direction and model role of backtranslation.

### Structural Contradiction
- location: `1 Introduction` P2 S3
- original: This survey will provide a broader perpective and be useful to both practitioners and researchers.
- modify to: This survey will provide a broader perpective and be useful to both practitioners and researchers. In addition, this survey systematically covers protein folding, molecular dynamics simulation, and clinical trial design.
- unsupported added scope: protein folding, molecular dynamics simulation, and clinical trial design

### Citation Or Topic Missing
- delete core citations:
  - `garciamartinez:hal-01433161` (Factored Neural Machine Translation Architectures), mentions=3, sections=['5 Use of external resources and linguistic information', '5 Use of external resources and linguistic information > 5.1 Linguistic Analyses']
  - `sennrich-haddow-2016-linguistic` (Linguistic Input Features Improve Neural Machine Translation), mentions=3, sections=['5 Use of external resources and linguistic information', '5 Use of external resources and linguistic information > 5.1 Linguistic Analyses']
  - `burlot-etal-2017-word` (Word Representations in Factored Neural Machine Translation), mentions=3, sections=['5 Use of external resources and linguistic information', '5 Use of external resources and linguistic information > 5.1 Linguistic Analyses']
  - `niehues-cho-2017-exploiting` (Exploiting Linguistic Resources for Neural Machine Translation Using Multi-task Learning), mentions=3, sections=['5 Use of external resources and linguistic information', '5 Use of external resources and linguistic information > 5.1 Linguistic Analyses']
  - `luong2016multitask` (Multi-task Sequence to Sequence Learning), mentions=3, sections=['5 Use of external resources and linguistic information', '5 Use of external resources and linguistic information > 5.1 Linguistic Analyses']
  - `zaremoodi2018adaptive` (Adaptive Knowledge Sharing in Multi-Task Learning: Improving Low-Resource Neural Machine Translation), mentions=3, sections=['5 Use of external resources and linguistic information', '5 Use of external resources and linguistic information > 5.1 Linguistic Analyses']
- delete subsection: `5 Use of external resources and linguistic information > 5.2 Incorporating Lexicons`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=5, citation_mentions=14)

## 20. Neuron-level Interpretation of Deep NLP Models: A Survey
- slug: `neuron-level_interpretation_of_deep_nlp_models_a_survey`
- date/query/arXiv: 2021-08-30 / `neuron-level interpretation deep NLP models` / `2108.13138v2`
- eligible middle top sections: [3, 4, 5]
- non-overlap: delete subsection `3 Neuron Analysis Methods > 3.2 Corpus-based Methods`; delete citation keys ['Mu-Nips', 'Na-ICLR', 'dalvi-2020-CCFS', 'lakretz-etal-2019-emergence', 'liu-etal-2019-linguistic', 'tenney-etal-2019-bert']; fact citation keys ['dalvi:2019:AAAI', 'durrani-etal-2020-analyzing', 'fyshe-etal-2015-compositional', 'hupkes2018visualisation', 'kadar-etal-2017-representation', 'karpathy2015visualizing', 'li-etal-2016-visualizing']

### Factual Errors
1. input-continuity error in `3 Neuron Analysis Methods > 3.5 Miscellaneous Methods` P2 S8
   - citations: ['kadar-etal-2017-representation']
   - refs: Kádár et al.
   - original: However, a gradient ascent can not be directly applied in NLP, because of the discrete inputs. poerner-etal-2018-interpretable worked around this problem by using Gumble Softmax and showed their method to surpass Concept Search method [36] in interpreting neurons.
   - modify to: Gradient ascent can be directly applied to NLP because token identities are continuous variables in the embedding vocabulary.
   - rationale: Deny the discrete-input obstacle that motivates the workaround.
2. analysis-method error in `3 Neuron Analysis Methods > 3.1 Visualization` P1 S1
   - citations: ['karpathy2015visualizing', 'fyshe-etal-2015-compositional', 'li-etal-2016-visualizing']
   - refs: Visualizing and understanding recurrent networks; Fyshe et al.; Visualizing and Understanding Neural Models in NLP
   - original: A simple way to discover the role of a neuron is by visualizing its activations and manually identifying the underlying concept over a set of sentences~ [19, 20, 21].
   - modify to: Visualization methods discover neuron roles by suppressing high-activation sentences and inspecting only examples where the neuron is inactive~ [19-21].
   - rationale: Reverse the activation-visualization procedure.
3. probing-target error in `3 Neuron Analysis Methods > 3.3 Probing-based Methods` P1 S1
   - citations: ['hupkes2018visualisation']
   - refs: Visualisation and 'diagnostic classifiers' reveal how recurrent and recursive neural networks process hierarchical structure
   - original: Probing-based methods train diagnostic classifiers~ [22] over activations to identify neurons with respect to pre-defined concepts.
   - modify to: Probing-based methods train diagnostic classifiers over model outputs rather than neuron activations to identify pre-defined concepts~ [22].
   - rationale: Move probing features from activations to final outputs.
4. control-purpose error in `3 Neuron Analysis Methods > 3.3 Probing-based Methods` P3 S3
   - citations: ['dalvi:2019:AAAI', 'durrani-etal-2020-analyzing']
   - refs: What Is One Grain of Sand in the Desert? Analyzing Individual Neurons in Deep NLP Models; Durrani et al.
   - original: Researchers have mitigated this pitfall for some analyses by using random initialization of neurons~ [25] and control tasks~ [15] to demonstrate that the knowledge is possessed within the neurons and not due to the probe's capacity for memorization.
   - modify to: Random initialization and control tasks are used to prove that probe memorization explains the results, not that knowledge is represented in the neurons~ [25, 15].
   - rationale: Invert the purpose of controls for probe-capacity concerns.

### Structural Contradiction
- location: `1 Introduction` P2 S3
- original: We term this work as the Representation Analysis.
- modify to: We term this work as the Representation Analysis. In addition, this survey systematically covers federated learning communication scheduling and wireless resource allocation.
- unsupported added scope: federated learning communication scheduling and wireless resource allocation

### Citation Or Topic Missing
- delete core citations:
  - `Na-ICLR` (Na et al.), mentions=2, sections=['3 Neuron Analysis Methods', '4 Evaluation > 4.4 Concept Selectivity']
  - `dalvi-2020-CCFS` (Analyzing Redundancy in Pretrained Transformer Models), mentions=2, sections=['3 Neuron Analysis Methods', '3 Neuron Analysis Methods > 3.4 Causation-based methods']
  - `lakretz-etal-2019-emergence` (Lakretz et al.), mentions=2, sections=['3 Neuron Analysis Methods', '3 Neuron Analysis Methods > 3.4 Causation-based methods']
  - `liu-etal-2019-linguistic` (Linguistic Knowledge and Transferability of Contextual Representations), mentions=2, sections=['5 Findings > 5.2 Architectural Analysis > 5.2.1 Information Distribution', '5 Findings > 5.3 Summary of Findings']
  - `tenney-etal-2019-bert` (Tenney et al.), mentions=2, sections=['5 Findings > 5.2 Architectural Analysis > 5.2.1 Information Distribution', '5 Findings > 5.3 Summary of Findings']
  - `Mu-Nips` (Mu and Andreas), mentions=1, sections=['3 Neuron Analysis Methods']
- delete subsection: `3 Neuron Analysis Methods > 3.2 Corpus-based Methods`
- subsection reason: middle method/content subsection with substantial text and citations (sentences=33, citation_mentions=0)
