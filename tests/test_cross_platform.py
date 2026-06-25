"""
CrossPlatformAnalyzer 单元测试
测试跨平台对比分析器的关键词提取与报告生成
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import importlib.util as _iu
_mod_path = os.path.join(os.path.dirname(__file__), '..', 'InsightEngine', 'tools', 'cross_platform_analyzer.py')
_spec = _iu.spec_from_file_location('cross_platform_analyzer', _mod_path)
_cpa = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_cpa)

CrossPlatformAnalyzer = _cpa.CrossPlatformAnalyzer
CrossPlatformReport = _cpa.CrossPlatformReport
KeywordStats = _cpa.KeywordStats
NarrativeDifference = _cpa.NarrativeDifference
TimeTrend = _cpa.TimeTrend
_extract_english_keywords = _cpa._extract_english_keywords
_extract_chinese_keywords = _cpa._extract_chinese_keywords


def _make_mock_result(**kwargs):
    """构造模拟 QueryResult 对象"""
    from datetime import datetime
    r = MagicMock()
    r.title_or_content = kwargs.get("content", "")
    r.platform = kwargs.get("platform", "")
    r.source_table = kwargs.get("source_table", "")
    pt = kwargs.get("publish_time")
    r.publish_time = datetime.strptime(pt, "%Y-%m-%d") if pt else None
    r.title = kwargs.get("title", "")
    r.content = kwargs.get("content", "")
    r.engagement = kwargs.get("engagement", {})
    return r


class TestEnglishKeywordExtraction:
    def test_basic_extraction(self):
        texts = ["South Africa faces energy crisis as Eskom implements load shedding"]
        keywords = _extract_english_keywords(texts)
        assert len(keywords) > 0
        # keywords is list of (word, count) tuples
        words = [kw[0] for kw in keywords]
        assert any(w in words for w in ("south", "africa", "energy", "crisis"))

    def test_empty_text(self):
        keywords = _extract_english_keywords([""])
        assert keywords == []

    def test_stopword_filtering(self):
        texts = ["the and of to in for is on that with this was"]
        keywords = _extract_english_keywords(texts)
        assert len(keywords) == 0


class TestChineseKeywordExtraction:
    def test_basic_extraction(self):
        text = "南非面临严重的能源危机"
        keywords = _extract_chinese_keywords([text])
        assert isinstance(keywords, list)

    def test_empty_text(self):
        assert _extract_chinese_keywords([""]) == []


class TestCrossPlatformAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return CrossPlatformAnalyzer()

    def _make_result(self, **kwargs):
        """构造模拟的 QueryResult 对象"""
        r = MagicMock()
        r.title_or_content = kwargs.get("content", "")
        r.platform = kwargs.get("platform", "")
        r.source_table = kwargs.get("source_table", "")
        r.publish_time = kwargs.get("publish_time", "")
        r.title = kwargs.get("title", "")
        r.content = kwargs.get("content", "")
        r.engagement = kwargs.get("engagement", {})
        return r

    @pytest.fixture
    def sample_news_results(self):
        return [
            _make_mock_result(title="SA Energy Crisis Deepens", content="Energy crisis in South Africa worsens.", platform="sa_news", publish_time="2024-03-15"),
            _make_mock_result(title="Eskom economic impact", content="Economists warn of GDP impact.", platform="sa_news", publish_time="2024-03-15"),
        ]

    @pytest.fixture
    def sample_tweet_results(self):
        return [
            _make_mock_result(content="No electricity again today #LoadShedding", platform="x", publish_time="2024-03-15"),
            _make_mock_result(content="SA needs to fix this power crisis", platform="x", publish_time="2024-03-16"),
        ]

    def test_analyze_basic(self, analyzer, sample_news_results, sample_tweet_results):
        report = analyzer.analyze("energy crisis", sample_news_results, sample_tweet_results)
        assert isinstance(report, CrossPlatformReport)
        assert report.topic == "energy crisis"
        assert report.news_keyword_stats_en is not None
        assert report.tweet_keyword_stats is not None

    def test_analyze_empty_input(self, analyzer):
        report = analyzer.analyze("test", [], [])
        assert isinstance(report, CrossPlatformReport)

    def test_analyze_time_trends(self, analyzer, sample_news_results, sample_tweet_results):
        report = analyzer.analyze("energy crisis", sample_news_results, sample_tweet_results)
        assert isinstance(report.time_trends, list)

    def test_analyze_narrative_differences(self, analyzer, sample_news_results, sample_tweet_results):
        report = analyzer.analyze("energy crisis", sample_news_results, sample_tweet_results)
        assert report.narrative_differences is not None
        assert isinstance(report.narrative_differences, NarrativeDifference)


class TestCrossPlatformReport:
    def test_dataclass_defaults(self):
        report = CrossPlatformReport()
        assert report.news_keyword_stats_en is None
        assert report.news_keyword_stats_zh is None
        assert report.tweet_keyword_stats is None
        assert isinstance(report.common_focus_points, list)
        assert report.llm_summary == ""


class TestFormatReportForLLM:
    def test_format_output(self):
        analyzer = CrossPlatformAnalyzer()
        news = [_make_mock_result(content="test news", platform="sa_news", title="Test Title")]
        tweets = [_make_mock_result(content="test tweet", platform="x")]
        report = analyzer.analyze("test", news, tweets)
        formatted = analyzer.format_report_for_llm(report)
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_format_empty_report(self):
        analyzer = CrossPlatformAnalyzer()
        report = CrossPlatformReport()
        formatted = analyzer.format_report_for_llm(report)
        assert isinstance(formatted, str)
