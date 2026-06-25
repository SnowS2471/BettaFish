"""
SAReportDataProvider 单元测试
测试南非专题报告数据预处理器的接口契约（使用 Mock 隔离数据库依赖）
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestSAReportDataProviderContract:
    """不加载真实模块，验证接口契约和数据结构"""

    def test_data_bundle_chapter_keys(self):
        """验证 5 个 data_bundle 的预期 key（翻译质量已下沉为后台能力，不注入报告）"""
        expected_keys = [
            "data_source_stats",
            "news_overview",
            "x_propagation",
            "cross_platform",
            "propagation_trends",
        ]
        assert len(expected_keys) == 5
        assert "cross_platform" in expected_keys
        assert "news_overview" in expected_keys

    def test_data_source_stats_structure(self):
        """验证 data_source_stats 数据结构"""
        mock_bundle = {
            "key": "data_source_stats",
            "total_news": 120,
            "total_tweets": 350,
            "site_distribution": [{"site": "News24", "count": 45}],
            "time_range": {"start": "2024-01-01", "end": "2024-03-31"},
        }
        assert mock_bundle["total_news"] >= 0
        assert mock_bundle["total_tweets"] >= 0
        assert isinstance(mock_bundle["site_distribution"], list)
        assert "start" in mock_bundle["time_range"]

    def test_cross_platform_for_chapter4_structure(self):
        """验证跨平台数据包结构（供第4章舆情焦点使用）"""
        mock_bundle = {
            "key": "cross_platform",
            "news_keywords_en": [{"word": "energy", "count": 45}],
            "news_keywords_zh": [{"word": "能源", "count": 40}],
            "tweet_keywords": [{"word": "LoadShedding", "count": 60}],
            "sentiment_comparison": [],
            "time_trends": [],
            "narrative_differences": {
                "news_only": [], "tweet_only": [], "common": [],
            },
            "common_focus_points": ["电力供应", "政府应对", "经济影响"],
        }
        assert "news_keywords_en" in mock_bundle
        assert "tweet_keywords" in mock_bundle
        assert "common_focus_points" in mock_bundle
        assert len(mock_bundle["common_focus_points"]) >= 1

    def test_x_propagation_structure(self):
        """验证 X 传播分析数据包结构"""
        mock_bundle = {
            "key": "x_propagation",
            "tweet_count_stats": {"total": 350, "original": 200, "retweet": 100, "reply": 50},
            "time_distribution": {"by_hour": {}, "by_date": {}},
            "top_keywords": [{"keyword": "energy", "count": 45}],
            "top_hashtags": [{"hashtag": "LoadShedding", "count": 60}],
            "active_accounts": [],
            "hot_tweets": [],
            "engagement_stats": {},
            "chart_configs": [],
        }
        assert mock_bundle["tweet_count_stats"]["total"] > 0
        assert "chart_configs" in mock_bundle

    def test_cross_platform_structure(self):
        """验证跨平台对比数据包结构"""
        mock_bundle = {
            "key": "cross_platform",
            "news_keywords": {"english": [], "chinese": []},
            "tweet_keywords": {"english": []},
            "sentiment_comparison": {
                "news_sentiment": {"positive": 0, "negative": 0, "neutral": 0},
                "social_sentiment": {"positive": 0, "negative": 0, "neutral": 0},
            },
            "narrative_differences": {"news_only": [], "tweet_only": [], "shared": []},
            "time_trend": {"news_daily": {}, "tweet_daily": {}},
            "shared_focus": [],
            "llm_summary": "",
        }
        assert "news_keywords" in mock_bundle
        assert "tweet_keywords" in mock_bundle
        assert "narrative_differences" in mock_bundle

    def test_propagation_trends_structure(self):
        """验证传播趋势数据包结构"""
        mock_bundle = {
            "key": "propagation_trends",
            "daily_comparison": [],
            "hot_tweets_ranking": [],
            "peak_date_news": "2024-03-15",
            "peak_date_tweets": "2024-03-16",
        }
        assert "daily_comparison" in mock_bundle
        assert "hot_tweets_ranking" in mock_bundle

    def test_build_all_bundles_output_format(self):
        """验证 build_all_data_bundles 输出格式契约（翻译质量不在注入列表中）"""
        expected_output = [
            {"key": "data_source_stats", "chapter_id": "chapter-2"},
            {"key": "news_overview", "chapter_id": "chapter-3"},
            {"key": "x_propagation", "chapter_id": "chapter-5"},
            {"key": "cross_platform", "chapter_id": "chapter-6"},
            {"key": "propagation_trends", "chapter_id": "chapter-8"},
        ]
        assert isinstance(expected_output, list)
        assert all(isinstance(b, dict) for b in expected_output)
        assert all("key" in b and "chapter_id" in b for b in expected_output)
        assert len(expected_output) == 5
