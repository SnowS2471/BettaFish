"""
TweetRelevanceFilter 单元测试
测试三层推文相关性过滤逻辑
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import importlib.util as _iu

_rf_path = os.path.join(os.path.dirname(__file__), '..', 'InsightEngine', 'tools', 'relevance_filter.py')
_rf_spec = _iu.spec_from_file_location('relevance_filter', _rf_path)
_rf = _iu.module_from_spec(_rf_spec)
_rf_spec.loader.exec_module(_rf)
TweetRelevanceFilter = _rf.TweetRelevanceFilter


class TestTier1RuleFilter:
    """第1层：规则预过滤"""

    @pytest.fixture
    def f(self):
        return TweetRelevanceFilter()

    def test_keyword_match(self, f):
        score = f._tier1_rule_filter(
            content="South Africa energy crisis is getting worse",
            hashtags_raw="",
            topic="energy crisis",
        )
        assert score is not None
        assert score > 0

    def test_hashtag_match(self, f):
        # 内容不小于15字符，话题标签匹配
        score = f._tier1_rule_filter(
            content="no power today in our city",
            hashtags_raw='["loadshedding", "Eskom"]',
            topic="loadshedding",
        )
        assert score is not None
        assert score >= 0.4

    def test_too_short(self, f):
        score = f._tier1_rule_filter(
            content="ok",
            hashtags_raw="",
            topic="energy crisis",
        )
        assert score is None

    def test_empty_content(self, f):
        score = f._tier1_rule_filter(
            content="",
            hashtags_raw="",
            topic="energy crisis",
        )
        assert score is None

    def test_default_min_length(self):
        f = TweetRelevanceFilter(min_length=30)
        score = f._tier1_rule_filter(
            content="short tweet about energy",
            hashtags_raw="",
            topic="energy",
        )
        assert score is None


class TestFilterTweets:
    """filter_tweets 主入口测试"""

    @pytest.fixture
    def f(self):
        flt = TweetRelevanceFilter()
        # 嵌入层返回与输入相同长度的评分列表
        flt._tier2_embedding_similarity = MagicMock(
            side_effect=lambda contents, topic: [0.5] * len(contents)
        )
        return flt

    def test_empty_list(self, f):
        result = f.filter_tweets([], topic="test", threshold=0.35)
        assert result == []

    def test_all_filtered_low_score(self, f):
        f._tier1_rule_filter = MagicMock(return_value=None)
        tweets = [MagicMock(title_or_content="irrelevant")]
        result = f.filter_tweets(tweets, topic="test", threshold=0.35)
        assert len(result) == 0

    def test_all_pass_high_score(self, f):
        f._tier1_rule_filter = MagicMock(return_value=0.9)
        tweets = [MagicMock(title_or_content="relevant")]
        result = f.filter_tweets(tweets, topic="test", threshold=0.35)
        assert len(result) == 1

    def test_mixed_results(self, f):
        # 一部分通过，一部分被过滤
        f._tier1_rule_filter = MagicMock(side_effect=[0.9, None, 0.7])
        tweets = [MagicMock(title_or_content=f"t{i}") for i in range(3)]
        result = f.filter_tweets(tweets, topic="test", threshold=0.35)
        assert len(result) == 2
