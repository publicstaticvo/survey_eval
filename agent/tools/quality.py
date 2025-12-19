import re
import time
import asyncio
import numpy as np
from typing import List, Tuple
from collections import Counter
from sentence_transformers.util import cos_sim

from .tool_config import ToolConfig
from .prompts import CLARITY_EVAL_PROMPT
from .request_utils import AsyncLLMClient
from .sbert_client import SentenceTransformerClient
from .utils import split_content_to_paragraph, prepare_paragraphs_for_clarity, extract_json


class QualityLLMClient(AsyncLLMClient):

    PROMPT: str = CLARITY_EVAL_PROMPT

    def _availability(self, response):
        # TODO: 修改输出方式为json，包含score和简短评估。
        evaluation = extract_json(response)
        score = evaluation['score']
        if isinstance(score, str): 
            score = int(score)
            evaluation['score'] = score
        return evaluation


class ClarityCritic:
    
    def __init__(self, config: ToolConfig):
        self.llm = QualityLLMClient(config.llm_server_info, config.sampling_params, config.llm_num_workers)
    
    async def __call__(self, paper_content: dict):
        """
        每次给出本段和前一段。
        """
        paragraphs = prepare_paragraphs_for_clarity(paper_content)
        if not paragraphs: return {"status": "no content", "mean_score": 0, "detailed_results": []}

        tasks = [asyncio.create_task(self.llm.call(inputs=p)) for p in paragraphs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = [x for x in results if isinstance(x, dict) and 'score' in x]
        scores = [x['score'] for x in results]
        if not scores: return {"status": "request error", "mean_score": 0, "detailed_results": []}
        return {"status": "ok", "mean_score": sum(scores) / len(scores), "detailed_results": results}


class ProgrammaticReadabilityCritic:

    vowels: set = set("aeiouAEIOU")

    def _count_syllables(self, word: str) -> int:
        """
        简易音节计数：连续元音组算一个音节，但 e-silent 做简单修正。
        对于英文已足够做可读性统计。
        """
        word = word.strip().lower()
        # 去掉末尾的 e（如果前面有辅音）
        if word.endswith("e") and len(word) > 2 and word[-2] not in self.vowels:
            word = word[:-1]
        # 去掉非字母
        word = re.sub(r"[^a-z]", "", word)
        if not word:
            return 1
        # 计算元音组
        groups = re.findall(r"[aeiouy]+", word)
        return max(1, len(groups))


    def _tokenize_sentences(self, text: str):
        """
        非常简单的句子切分：. ! ? 后跟空格或大写字母。
        """
        sentences = re.split(r'[.!?]+(?:\s+|[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]


    def _tokenize_words(self, text: str):
        """
        仅提取连续字母（带连字符算一个单词）。
        """
        return re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", text)


    def _is_complex_word(self, word: str) -> bool:
        """
        Gunning Fog 定义：音节数 ≥ 3 为复杂词。
        """
        return self._count_syllables(word) >= 3


    # -------------------- 核心指标 --------------------
    def _flesch_metrics(self, text: str):
        sentences = self._tokenize_sentences(text)
        words = self._tokenize_words(text)
        if not sentences or not words:
            return None, None
        total_sentences = len(sentences)
        total_words = len(words)
        total_syllables = sum(self._count_syllables(w) for w in words)

        # Flesch Reading Ease
        fre = 206.835 - 1.015 * (total_words / total_sentences) \
            - 84.6 * (total_syllables / total_words)

        # Flesch-Kincaid Grade Level
        fkgl = 0.39 * (total_words / total_sentences) \
            + 11.8 * (total_syllables / total_words) - 15.59

        return round(fre, 2), round(fkgl, 2)


    def _gunning_fog(self, text: str):
        sentences = self._tokenize_sentences(text)
        words = self._tokenize_words(text)
        if not sentences or not words:
            return None
        total_sentences = len(sentences)
        total_words = len(words)
        complex_words = [w for w in words if self._is_complex_word(w)]
        percent_complex = 100.0 * len(complex_words) / total_words

        fog = 0.4 * ((total_words / total_sentences) + percent_complex)
        return round(fog, 2)

    async def __call__(self, paper_content: dict):
        paragraphs = split_content_to_paragraph(paper_content)
        paragraph_contents = [" ".join(x['text'] for x in p) for p in paragraphs]
        full_content = "\n".join(paragraph_contents)
        fre, fkgl = self._flesch_metrics(full_content)
        gfi = self._gunning_fog(full_content)
        return {
            "flesch reading ease": fre, 
            "flesch-kincaid grade": fkgl, 
            "gunning-fog index": gfi
        }


class ProgrammaticRedundancyCritic:

    def __init__(self, config: ToolConfig):
        self.sentence_transformer = SentenceTransformerClient(config.sbert_server_url)
        self.threshold = config.redundancy_similarity_threshold
        self.n_gram = config.redundancy_ngram

    def _tokenize(self, text: str) -> List[str]:
        """保留单词字符，小写化"""
        return re.findall(r"\b\w+\b", text.lower())


    def _ngrams(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        if self.n_gram <= 0 or len(tokens) < self.n_gram: return []
        return [tuple(tokens[i:i + self.n_gram]) for i in range(len(tokens) - self.n_gram + 1)]


    def _self_redundancy(self, text: str) -> float:
        """核心指标：1 − |unique| / |total|"""
        tokens = self._tokenize(text)
        ng = self._ngrams(tokens)
        if not ng: return 0.0
        return 1.0 - len(set(ng)) / len(ng)


    def _most_common_ngram(self, text: str) -> Tuple[Tuple[str, ...], int]:
        """顺便看看最频繁的 n-gram 是什么"""
        tokens = self._tokenize(text)
        ng = self._ngrams(tokens)
        if not ng: return ((), 0)
        (best, freq), = Counter(ng).most_common(1)
        return best, freq

    async def __call__(self, paper_content: dict):
        """
        1. n-gram overlap
        2. 段落之间的相似性
        3. 段内句子的相似性
        """
        paragraphs = split_content_to_paragraph(paper_content)
        paragraph_contents = [" ".join(x['text'] for x in p) for p in paragraphs]
        full_content = "\n".join(paragraph_contents)
        # self-n-gram overlap
        self_redundancy = self._self_redundancy(full_content)
        most_common_ngram = self._most_common_ngram(full_content)
        # paragraph similarity
        paragraph_embeddings = self.sentence_transformer.embed(paragraph_contents)
        sim = cos_sim(paragraph_embeddings, paragraph_embeddings)
        high_sim_pairs = np.argwhere(sim > self.threshold).tolist()
        high_sim_paragraphs = len(high_sim_pairs)
        # high_sim_paragraphs = [(paragraph_contents[i], paragraph_contents[j]) for i, j in high_sim_pairs if i < j]
        high_sim_sentences = 0
        for p in paragraphs:
            sentences = [x['text'] for x in p]
            sentence_embeddings = self.sentence_transformer.embed(sentences)
            sentence_sim = cos_sim(sentence_embeddings, sentence_embeddings)
            high_sim_pairs = np.argwhere(sentence_sim > self.threshold).tolist()
            high_sim_sentences += len(high_sim_pairs)
            # high_sim_sentences.extend([(sentences[i], sentences[j]) for i, j in high_sim_pairs if i < j])
        return {
            "self_redundancy": self_redundancy,
            "most_common_ngram": most_common_ngram,
            "high_sim_paragraphs_count": high_sim_paragraphs,
            "high_sim_sentences_count": high_sim_sentences,
        }


class QualityCritic:
    """
    Runs Clarity, Readability, Redundancy in one node
    """

    def __init__(self, config: ToolConfig):
        # Initialize classes (assuming they are wrapped as simple async functions or classes)
        self.redundancy_tool = ProgrammaticRedundancyCritic(config)
        self.readability_tool = ProgrammaticReadabilityCritic()
        self.clarity_tool = ClarityCritic(config)

    async def _run(self, review_paper):
        # 1. Create Async Tasks
        # Note: segment_tool only needs the review_text, not the parsed papers!
        task_redundancy = self.redundancy_tool(review_paper)
        task_readability = self.readability_tool(review_paper) 
        task_clarity = self.clarity_tool(review_paper)

        # 2. Run them all at once
        redundancy_res, readability_res, clarity_res = asyncio.gather(
            task_redundancy, task_readability, task_clarity
        )

        # 3. Return combined state updates
        return {"quality_evals": {
            "redundancy_evals": redundancy_res,
            "readability_evals": readability_res,
            "clarity_evals": clarity_res,
        }}
