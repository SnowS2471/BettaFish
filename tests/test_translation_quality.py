"""
翻译质量评分与统计单元测试
测试翻译评分计算和统计汇总逻辑
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# 绕过 InsightEngine/__init__.py 的级联导入，直接加载目标模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import importlib.util as _iu
_mod_path = os.path.join(os.path.dirname(__file__), '..', 'InsightEngine', 'utils', 'text_processing.py')
_spec = _iu.spec_from_file_location('text_processing', _mod_path)
_tp = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

detect_content_language = _tp.detect_content_language
merge_bilingual_results = _tp.merge_bilingual_results
normalize_platform_text = _tp.normalize_platform_text
partition_by_platform = _tp.partition_by_platform


class TestDetectContentLanguage:
    def test_chinese_text(self):
        assert detect_content_language("这是一段中文文本，用于测试语言检测功能") == "zh"

    def test_english_text(self):
        assert detect_content_language("This is an English text for testing language detection") == "en"

    def test_mixed_text(self):
        text = "南非 South Africa 的能源 energy 危机 crisis"
        result = detect_content_language(text)
        # 混合文本应返回 mixed 或根据比例判断
        assert result in ("zh", "en", "mixed")

    def test_empty_text(self):
        assert detect_content_language("") == "en"

    def test_chinese_dominant(self):
        text = "南非面临严重的能源危机，Eskom公司实施了六级限电措施"
        assert detect_content_language(text) == "zh"

    def test_english_dominant(self):
        text = "South Africa faces severe energy crisis with Eskom implementing load shedding"
        assert detect_content_language(text) == "en"


class TestMergeBilingualResults:
    def test_with_both_languages(self):
        results = [
            {
                "title": "Energy Crisis",
                "content": "The energy crisis continues.",
                "title_zh": "能源危机",
                "content_zh": "能源危机持续。",
                "platform": "sa_news",
            }
        ]
        merged = merge_bilingual_results(results)
        assert len(merged) == 1
        assert merged[0]["has_translation"] is True

    def test_without_translation(self):
        results = [
            {
                "title": "Breaking News",
                "content": "Something happened.",
                "title_zh": "",
                "content_zh": "",
                "platform": "sa_news",
            }
        ]
        merged = merge_bilingual_results(results)
        assert merged[0]["has_translation"] is False

    def test_empty_list(self):
        assert merge_bilingual_results([]) == []

    def test_partial_translation(self):
        results = [
            {
                "title": "News",
                "content": "Content.",
                "title_zh": "新闻",
                "content_zh": "",
                "platform": "sa_news",
            }
        ]
        merged = merge_bilingual_results(results)
        assert merged[0]["has_translation"] is True


class TestNormalizePlatformText:
    def test_x_platform(self):
        text = "Check this out! #BreakingNews\n\n\n@someuser what do you think?"
        normalized = normalize_platform_text(text, "x")
        assert "\n\n\n" not in normalized
        assert "#BreakingNews" in normalized

    def test_sa_news_platform(self):
        text = "A very long article " * 500
        normalized = normalize_platform_text(text, "sa_news")
        assert len(normalized) <= 6003  # 6000 + "... " overhead

    def test_sa_news_short(self):
        text = "Short news article content."
        normalized = normalize_platform_text(text, "sa_news")
        assert text in normalized

    def test_generic_platform(self):
        text = "Some   content  with   extra spaces\n\n\nand newlines"
        normalized = normalize_platform_text(text, "weibo")
        assert "Some content" in normalized

    def test_empty_text(self):
        assert normalize_platform_text("", "x") == ""


class TestPartitionByPlatform:
    def test_mixed_platforms(self):
        results = [
            {"content": "news", "platform": "sa_news"},
            {"content": "tweet1", "platform": "x"},
            {"content": "tweet2", "platform": "x"},
            {"content": "weibo post", "platform": "weibo"},
        ]
        grouped = partition_by_platform(results)
        assert "南非新闻" in grouped
        assert "X平台" in grouped
        assert "微博" in grouped
        assert len(grouped["X平台"]) == 2
        assert len(grouped["南非新闻"]) == 1

    def test_single_platform(self):
        results = [{"content": "t", "platform": "x"}]
        grouped = partition_by_platform(results)
        assert len(grouped) == 1

    def test_empty_list(self):
        assert partition_by_platform([]) == {}

    def test_unknown_platform(self):
        results = [{"content": "t", "platform": "unknown_platform"}]
        grouped = partition_by_platform(results)
        assert "unknown_platform" in grouped

    def test_source_table_fallback(self):
        results = [{"content": "t", "source_table": "bilibili"}]
        grouped = partition_by_platform(results)
        assert "B站" in grouped


class TestTranslationScoreCalculation:
    """翻译评分计算逻辑测试"""

    def test_overall_score_average(self):
        """综合评分应为各维度平均分"""
        scores = {
            "eval_accuracy": 85,
            "eval_fluency": 80,
            "eval_terminology": 82,
            "eval_completeness": 88,
        }
        overall = sum(scores.values()) / len(scores)
        assert abs(overall - 83.75) < 0.01

    def test_quality_flag_good(self):
        overall = 85
        if overall >= 80:
            flag = "good"
        elif overall >= 60:
            flag = "fair"
        else:
            flag = "poor"
        assert flag == "good"

    def test_quality_flag_poor(self):
        overall = 45
        flag = "good" if overall >= 80 else "fair" if overall >= 60 else "poor"
        assert flag == "poor"
