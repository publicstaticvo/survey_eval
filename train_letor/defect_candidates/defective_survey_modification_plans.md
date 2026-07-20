# Defective Survey Construction Plans

Source set: `survey_eval/train_letor/downloaded/selected_manifest.json`.

These are editing plans only. No LaTeX source or parsed JSON is modified here. For factual-error and missing-citation edits, the chosen evidence is from middle body sections when available, avoiding the first two main sections, the last two main sections, and appendices.

## 1. Retrieval-Augmented Generation for Large Language Models: A Survey

- Factual error: In `6 Task and Evaluation > 6.2 Evaluation Target`, change: "question answering evaluations might rely on EM and F1 scores [74, 51, 75, 7], whereas fact-checking tasks often hinge on Accuracy [4, 76, 14]." to: "question answering evaluations might rely on EM, F1, and BLEU scores [74, 51, 75, 7], whereas fact-checking tasks often hinge on Accuracy, ROUGE, and latency [4, 76, 14]." This attributes unsuitable metrics to the cited works.
- Structural contradiction: In the introduction, change the scope sentence about analyzing "Retrieval, Generation, and Augmentation" to additionally claim that the survey analyzes "RAG deployment cost modeling and privacy-preserving federated RAG." These topics are not covered as core sections.
- Missing citation/topic: Delete citations `[74, 51, 75, 7]` from the EM/F1 sentence in `6.2 Evaluation Target`; delete subsection `6.1 Downstream Task`.

## 2. Instruction Tuning for Large Language Models: A Survey

- Factual error: In `3 Datasets > 3.2 Synthetic Data via Distillation`, change: "WizardCoder, Magicoder and WaveCoder" are examples for coding generation, "Phi-1 and Phi-1.5" for reasoning/writing, and "Nectar" for ranking, to: "WizardCoder, Magicoder and WaveCoder are mainly ranking datasets, Phi-1 and Phi-1.5 are dialogue preference datasets, and Nectar is a code generation benchmark." This reverses the cited categories.
- Structural contradiction: In the introduction outline, add that the survey includes a dedicated section on "instruction tuning for robotics control and embodied navigation." The parsed section list has no such section.
- Missing citation/topic: Delete citations `[59, 60, 61]` from the coding generation example; delete subsection `3.1 Human-crafted Data`.

## 3. A Survey on Evaluation of Large Language Models

- Factual error: In `6 Summary > 6.1.1 What can do well?`, change the claim that mathematical reasoning and structured data inference are prevailing evaluation benchmarks and that LLMs generate coherent responses to: "these works show that LLMs already solve medical diagnosis, legal contract review, and multilingual evaluation without benchmark-specific failures." This overstates the cited evaluation results.
- Structural contradiction: In the introduction, change the three-dimensional scope "what to evaluate, where to evaluate, and how to evaluate" to "what to evaluate, where to evaluate, how to evaluate, and how to train LLMs from scratch." Training LLMs from scratch is outside the survey's section structure.
- Missing citation/topic: Delete a dense group of benchmark citations from `4 Where to Evaluate: Datasets and Benchmarks` around MME, Xiezhi, Choice-75, CUAD, TRUSTGPT, and MMLU-style entries; delete subsection `3.1 Natural Language Processing Tasks`.

## 4. Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond

- Factual error: In `4 Practical Guide for NLP Tasks > 4.1.1 No use case`, change: "fine-tuned models perform better than LLMs on RTE and SNLI, while on CB LLMs have comparable performance" to: "LLMs consistently outperform fine-tuned models on RTE, SNLI, and CB without task-specific tuning." This contradicts the cited comparison.
- Structural contradiction: In the introduction's structure paragraph, add: "We also provide a practical guide for training multimodal foundation models from raw video and audio corpora." The paper's practical guide sections are models, data, NLP tasks, and other considerations, not multimodal pretraining.
- Missing citation/topic: Delete citations `[27, 34, 28, 9]` from the NLI comparison sentence; delete subsection `4.4 Abilities Regarding Scaling`.

## 5. Augmented Language Models: a Survey

- Factual error: In `3 Using Tools and Act > 3.2.1 Retrieval-augmented language models`, change: "Various works augment LMs by appending retrieved documents to the current context" to: "Various works augment LMs by replacing the language model parameters with retrieved documents, eliminating the need for neural generation." This misrepresents retrieval-augmented LMs.
- Structural contradiction: In the introduction, change the three axes "reasoning, using tools, and acting" to add a fourth core axis, "on-device model compression and quantization." No corresponding body section exists.
- Missing citation/topic: Delete citations `[71, 72, 73, 74, 75, 76, 77, 78, 14, 15]` from the retrieval-augmented LM sentence; delete subsection `4.2 Reinforcement learning`.

## 6. A Survey on In-context Learning

- Factual error: In `5 Analysis > 5.2.1 Functional Modules`, change: "the attention module is a focal point in the study of ICL mechanism [75, 76, 18, 61, 77, 78, 79]" to: "the cited works show that attention modules are irrelevant to ICL and that only tokenization determines ICL behavior." This contradicts the cited mechanism studies.
- Structural contradiction: In the introduction taxonomy paragraph, add that the survey systematically reviews "parameter-efficient fine-tuning algorithms such as LoRA and adapters as the primary alternative to ICL." The body focuses on ICL model training, prompt design, analysis, applications, and challenges, not PEFT surveys.
- Missing citation/topic: Delete citations `[75, 76, 18, 61, 77, 78, 79]` from the functional-modules sentence; delete subsection `4.1 Demonstration Organization`.

## 7. Towards Reasoning in Large Language Models: A Survey

- Factual error: In `4 Measuring Reasoning in Large Language Models > 4.1 End Task Performance`, change the arithmetic benchmark list "GSM8K, Math, MathQA, SVAMP, ASDiv, AQuA, and MAWPS" to: "GSM8K, Math, MathQA, SVAMP, ASDiv, AQuA, and MAWPS are primarily benchmarks for image captioning and visual grounding." This makes the cited benchmark category false.
- Structural contradiction: In the introduction, replace the outline sentence with: "Section 3 studies reasoning, Section 4 evaluates reasoning, and Section 5 provides a complete deployment guide for RAG systems." The parsed body has no RAG deployment guide.
- Missing citation/topic: Delete citations `[21, 37, 85, 86, 87, 88, 89]` from the arithmetic benchmark sentence; delete subsection `3.2 Prompting & In-Context Learning`.

## 8. Multimodal Learning with Transformers: A Survey

- Factual error: In `5 Challenges and Designs > 5.2 Alignment`, change: "cross-modal alignment has been studied for speaker localization, speech translation, text-to-speech alignment, text-to-video retrieval, and visual grounding" to: "the cited alignment works show that cross-modal alignment is unnecessary for speaker localization, speech translation, retrieval, or visual grounding." This reverses the cited claim.
- Structural contradiction: In the introduction scope sentence, add that the survey covers "federated optimization and privacy attacks for multimodal Transformers" as a major axis. The body has applications/challenges/designs, not federated privacy optimization.
- Missing citation/topic: Delete citations from the cross-modal alignment sentence in `5.2 Alignment`; delete subsection `4.1 Transformers for Multimodal Pretraining`.

## 9. Image Data Augmentation for Deep Learning: A Survey

- Factual error: In `4 Evaluation > 4.2 Image Classification`, change: "classification accuracy is compared with and without augmentation using Wide-ResNet, DenseNet, and Shake ResNet" to: "augmentation is evaluated only on generative adversarial networks and no classification backbones are considered." This conflicts with the cited backbones.
- Structural contradiction: In the introduction taxonomy paragraph, add that the survey systematically covers "data augmentation for large language model instruction tuning." The paper is scoped to image data augmentation for CV tasks.
- Missing citation/topic: Delete citations `[48, 49, 50]` from the image classification model sentence; delete subsection `3.1 Auto Augment`.

## 10. Transformers in Time Series: A Survey

- Factual error: In `5.1.1 Time Series Forecasting`, change: "LogTrans, Informer, AST, Pyraformer, Quatformer, and FEDformer exploit sparsity inductive bias or low-rank approximation to remove noise and reduce complexity" to: "these six works primarily increase quadratic attention complexity to improve image classification accuracy." This misstates both task and method.
- Structural contradiction: In the introduction, extend the application scope from forecasting/anomaly detection/classification to include "medical image segmentation and machine translation with Transformers." These are outside the time-series survey body.
- Missing citation/topic: Delete citations `[8, 28, 30, 29, 31, 9]` from the six-work forecasting sentence; delete subsection `5.1 Transformers in Forecasting`.

## 11. Survey of Hallucination in Natural Language Generation

- Factual error: In `5.1.3 Information Augmentation`, change the examples "entity information, extracted relation triples, pre-executed operation results, synthetic data, retrieved external knowledge, and retrieved similar training samples" to: "the cited works demonstrate that hallucination mitigation relies only on longer beam search and never uses external information." This contradicts the cited mitigation methods.
- Structural contradiction: In the introduction, change the scope from hallucination in unimodal NLG with brief multimodal discussion to a claim that the survey provides "a full benchmark of hallucination in robotic planning and reinforcement learning." Those topics are absent.
- Missing citation/topic: Delete citations `[95, 94, 83, 96, 84, 15, 46, 108, 109, 99, 110, 111]` from the information augmentation sentence; delete subsection `8.2 Open-domain Dialogue Generation`.

## 12. Transformers in Medical Imaging: A Survey

- Factual error: In `5.2.1 Undersampled MRI Reconstruction`, change: "Korkmaz et al. propose SLATER, a zero-shot framework using randomly initialized neural-network priors for unsupervised MR image reconstruction" to: "Korkmaz et al. propose SLATER as a supervised object-detection framework for natural images." This changes both task and supervision.
- Structural contradiction: In the background/introduction, add that the survey's main scope includes "Transformer-based legal document understanding and text summarization." The body is medical imaging applications.
- Missing citation/topic: Delete citations `[224, 225, 7, 8]` from the SLATER sentence; delete subsection `8.2 Supervised and Unsupervised Approaches`.

## 13. Generalized Out-of-Distribution Detection: A Survey

- Factual error: In `3 Response to Reviewer #3`, change: "linear probing, prompt tuning, and adaptor-style fine-tuning methods do not have good results on OOD detection" to: "linear probing, prompt tuning, and adaptor-style fine-tuning completely solve OOD detection on downstream semantic spaces." This contradicts the cited statement.
- Structural contradiction: The parsed JSON does not expose a normal introduction; use `1 Summary of Revision` as the intro-like location and add: "This survey provides a full taxonomy of RAG evaluation metrics and downstream QA datasets." The parsed body concerns OOD revision responses, not RAG evaluation.
- Missing citation/topic: Delete citations `[15, 16, 17, 18]` from the prompt-tuning/adaptor OOD sentence. No eligible method-introduction subsection is present in the parsed result, so no subsection deletion is planned for this paper.

## 14. A Survey on Multi-modal Summarization

- Factual error: In `3.1 On the basis of encoding the input`, change: "text-image, text-video, audio-video, and text-image-audio-video combinations have been explored" to: "existing MMS work only studies text-only summarization and contains no image, video, or audio inputs." This contradicts the cited multimodal taxonomy.
- Structural contradiction: In the introduction, change the scope from datasets, methodology, and evaluation techniques for MMS to include "a full survey of low-resource machine translation and bilingual lexicon induction." That topic is not in the paper.
- Missing citation/topic: Delete citations `[3, 70, 71, 72, 73, 74, 75, 76, 2, 1, 77, 69]` from the input-modality combination sentence; delete subsection `4.1 Main Model`.

## 15. Towards Efficient Synchronous Federated Training: A Survey on System Optimization Strategies

- Factual error: In `3.2.2 Production Systems and Simulation Platforms`, change: "FATE is an FL framework that can be deployed in distributed environments" and supports practical FL research goals to: "FATE is only a single-machine image classification dataset and cannot be deployed as an FL framework." This misrepresents the cited framework.
- Structural contradiction: In the intro/scope paragraph, add that the survey covers "asynchronous federated learning security on heterogeneous devices" as a main focus. This paper is scoped to efficient synchronous federated training.
- Missing citation/topic: Delete citations around FATE, Flower, FedML, Plato, and related production/simulation platforms from `3.2.2`; delete subsection `3.2 Benchmarking Suites`.

## 16. Asynchronous Federated Learning on Heterogeneous Devices: A Survey

- Factual error: In `5 Privacy and Security on Heterogeneous Devices`, change: "new attack vectors include membership inference, property inference, model inversion, and deep leakage from gradients" to: "the cited works show AFL eliminates membership inference, property inference, model inversion, and gradient leakage attacks by design." This reverses the cited privacy-risk claim.
- Structural contradiction: In the introduction contribution list, add that the survey provides "a comprehensive benchmark of synchronous-only FL system optimization strategies." The paper's scope is AFL on heterogeneous devices.
- Missing citation/topic: Delete citations `[78, 79, 80, 58]` from the privacy attack sentence; delete subsection `5.2 Security on Heterogeneous Devices`.

## 17. A Survey of Deep Reinforcement Learning in Recommender Systems: A Systematic Review and Future Directions

- Factual error: In `5.2 Exploration and Exploitation`, change: "epsilon-greedy is the most common technique used to encourage exploration" to: "epsilon-greedy is never used for exploration in DRL-based recommender systems; it is only a supervised loss for rating prediction." This contradicts the cited works.
- Structural contradiction: In the introduction contribution list, add that the survey includes "a complete taxonomy of diffusion models for image generation." This is outside DRL-based recommender systems.
- Missing citation/topic: Delete citations `[17, 30, 25, 26, 23, 18, 60, 58, 67]` from the epsilon-greedy exploration sentence; delete subsection `3.2 Model-free deep reinforcement learning based methods`.

## 18. A Survey of Exploration Methods in Reinforcement Learning

- Factual error: In `5.1 Value-Based Methods`, change: "vectors must be mapped to scalar values using a scalarization function, with linear or non-linear representations" to: "multi-objective value vectors are used directly without scalarization in the Boltzmann formalism." This contradicts the cited scalarization discussion.
- Structural contradiction: In the introduction, change the scope from exploration methods for sequential decision-making RL to "exploration methods plus a survey of neural machine translation for low-resource languages." The latter is absent.
- Missing citation/topic: Delete citations `[74, 73, 75, 76, 77, 78, 79]` from the scalarization sentence; delete subsection `5.2 Policy-Search Based Methods`.

## 19. Survey of Low-Resource Machine Translation

- Factual error: In `5 Use of external resources and linguistic information`, change: "bilingual lexicons can support unseen or low-frequency terms through seed/anchor use, lookup components, and translation candidates" to: "the cited works show bilingual lexicons are harmful and are never used for unseen or low-frequency terms in low-resource MT." This contradicts the cited methods.
- Structural contradiction: In the introduction, add that the survey systematically covers "reinforcement-learning exploration strategies and count-based bonuses." The paper is scoped to low-resource MT.
- Missing citation/topic: Delete citations `[62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]` from the bilingual-lexicon sentence; delete subsection `3.2 Data Augmentation`.

## 20. Neuron-level Interpretation of Deep NLP Models: A Survey

- Factual error: In `3.4 Causation-based methods`, change: "researchers use leave-one-out estimates, beam search, differentiable prediction models, and correlation clustering before ablation" to: "the cited methods avoid ablation or attribution entirely and rely only on manual inspection of attention heatmaps." This misstates the method family.
- Structural contradiction: In the introduction, change the scope from neuron-level representation analysis and interpretation to additionally claim that the survey covers "end-to-end factuality checking for generated summaries." That topic is not a body section.
- Missing citation/topic: Delete citations `[28, 29, 30, 31]` from the causation-based methods sentence; delete subsection `5.1 Concept Discovery`.
