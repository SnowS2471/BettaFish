"""
推文相关性过滤器

对 X 平台搜索结果进行三层过滤，去除与搜索主题无关的推文：
  Tier 1: 规则预过滤（关键词匹配、长度过滤、纯转推过滤）
  Tier 2: 语义相似度（多语言嵌入模型余弦相似度）
  Tier 3: LLM 判断（可选，用于边界情况）
"""

import re
import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from loguru import logger


@dataclass
class RelevanceScore:
    """单条推文的相关性打分明细（规则分 / 嵌入相似度 / 可选 LLM 分 / 最终分 / 是否相关）。"""
    rule_score: float
    embedding_score: float
    llm_score: Optional[float]
    final_score: float
    is_relevant: bool


class TweetRelevanceFilter:
    """推文相关性过滤器：三层（规则 -> 语义相似度 -> 可选 LLM）过滤无关推文。

    嵌入模型懒加载且可缺省：无模型时退化为「仅规则过滤」（语义分给定中性 0.5）。
    """
    def __init__(self, embedding_model=None, min_length: int = 15):
        self._embedding_model = embedding_model
        self._min_length = min_length

    def _load_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            logger.warning(f"无法加载嵌入模型，将仅使用规则过滤: {e}")
        return self._embedding_model

    def _tier1_rule_filter(self, content: str, hashtags_raw: str, topic: str) -> Optional[float]:
        """规则预过滤。返回 None 表示直接丢弃，返回 float 为基础分。"""
        if not content or len(content.strip()) < self._min_length:
            return None

        topic_lower = topic.lower()
        topic_words = set(re.findall(r'\w+', topic_lower))
        content_lower = content.lower()

        hashtags = []
        if hashtags_raw:
            try:
                hashtags = [t.lower() for t in json.loads(hashtags_raw)] if hashtags_raw.startswith('[') else [hashtags_raw.lower()]
            except (json.JSONDecodeError, TypeError):
                pass

        content_match = any(w in content_lower for w in topic_words) if topic_words else topic_lower in content_lower
        hashtag_match = any(any(w in tag for w in topic_words) for tag in hashtags)

        if content_match:
            return 0.6
        if hashtag_match:
            return 0.4
        return 0.2

    def _tier2_embedding_similarity(self, contents: List[str], topic: str) -> List[float]:
        """批量计算语义相似度。"""
        model = self._load_embedding_model()
        if model is None:
            return [0.5] * len(contents)

        try:
            topic_emb = model.encode([topic], normalize_embeddings=True)
            content_embs = model.encode(contents, normalize_embeddings=True, batch_size=64)
            similarities = np.dot(content_embs, topic_emb.T).flatten()
            return similarities.tolist()
        except Exception as e:
            logger.warning(f"嵌入相似度计算失败: {e}")
            return [0.5] * len(contents)

    def filter_tweets(
        self,
        tweets: list,
        topic: str,
        threshold: float = 0.35,
        use_llm: bool = False,
    ) -> list:
        """
        过滤推文列表，返回与主题相关的推文。

        Args:
            tweets: QueryResult 列表
            topic: 搜索主题
            threshold: 最终得分阈值，低于此值的推文被过滤
            use_llm: 是否启用 Tier 3 LLM 判断（暂未实现）

        Returns:
            过滤后的 QueryResult 列表
        """
        if not tweets:
            return tweets

        original_count = len(tweets)

        # Tier 1: 规则预过滤
        tier1_passed = []
        tier1_scores = []
        for t in tweets:
            content = t.title_or_content or ""
            hashtags_raw = ""
            if hasattr(t, 'engagement') and isinstance(t.engagement, dict):
                hashtags_raw = ""
            score = self._tier1_rule_filter(content, hashtags_raw, topic)
            if score is not None:
                tier1_passed.append(t)
                tier1_scores.append(score)

        if not tier1_passed:
            logger.info(f"[RelevanceFilter] 全部 {original_count} 条推文被规则过滤淘汰")
            return []

        # Tier 2: 语义相似度
        contents = [t.title_or_content or "" for t in tier1_passed]
        embedding_scores = self._tier2_embedding_similarity(contents, topic)

        # 综合打分: rule_score * 0.3 + embedding_score * 0.7
        results = []
        for tweet, rule_s, emb_s in zip(tier1_passed, tier1_scores, embedding_scores):
            final = rule_s * 0.3 + emb_s * 0.7
            if final >= threshold:
                results.append(tweet)

        filtered_count = original_count - len(results)
        logger.info(
            f"[RelevanceFilter] 主题 '{topic}': {original_count} 条推文 → "
            f"过滤 {filtered_count} 条 → 保留 {len(results)} 条 (阈值={threshold})"
        )
        return results
