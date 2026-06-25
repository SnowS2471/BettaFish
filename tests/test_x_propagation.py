"""
X平台传播特征分析模块 — 单元测试

不依赖数据库连接，通过 mock 验证：
1. 数据结构正确性
2. 关键词/hashtag 提取逻辑
3. 图表配置生成
4. LLM 格式化输出
5. 空数据处理
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from InsightEngine.tools.x_propagation_analyzer import (
    XPropagationAnalyzer,
    XPropagationReport,
    TweetCountStats,
    TimeDistribution,
    KeywordFrequency,
    HashtagFrequency,
    ActiveAccount,
    ActiveAccountStats,
    HotTweet,
    HotTweetRanking,
    EngagementStats,
    InteractionHotnessRanking,
)


# ── 辅助 ──

MOCK_TWEET_COUNT_ROWS = [{"total": 100, "originals": 60, "rts": 20, "quotes": 10, "replies": 10}]

MOCK_HOURLY_ROWS = [{"h": 9, "cnt": 15}, {"h": 14, "cnt": 25}, {"h": 20, "cnt": 10}]
MOCK_DAILY_ROWS = [{"d": "2025-04-01", "cnt": 30}, {"d": "2025-04-02", "cnt": 70}]

MOCK_CONTENT_ROWS = [
    {"content": "South Africa election results show ANC losing ground"},
    {"content": "South Africa power crisis continues with load shedding"},
    {"content": "South Africa economy faces challenges amid global uncertainty"},
]

MOCK_HASHTAG_ROWS = [
    {"hashtags": '["SouthAfrica", "Election2025"]'},
    {"hashtags": '["SouthAfrica", "LoadShedding"]'},
    {"hashtags": '["ANC", "Election2025", "SouthAfrica"]'},
    {"hashtags": "invalid json"},
    {"hashtags": "[]"},
]

MOCK_ACTIVE_ROWS = [
    {"user_id": "u1", "username": "alice", "nickname": "Alice", "verified": 1,
     "vtype": "blue", "cnt": 10, "tlikes": 500, "trts": 100, "tviews": 5000},
    {"user_id": "u2", "username": "bob", "nickname": "Bob", "verified": 0,
     "vtype": "", "cnt": 5, "tlikes": 200, "trts": 50, "tviews": 2000},
]

MOCK_UNIQUE_ROWS = [{"uniq": 50, "verified_cnt": 5}]
MOCK_CREATOR_ROWS = [
    {"user_id": "u1", "followers": 10000},
    {"user_id": "u2", "followers": 500},
]

MOCK_HOT_TWEET_ROWS = [
    {"tweet_id": "t1", "content": "Breaking news from SA", "username": "alice",
     "nickname": "Alice", "tweet_url": "https://x.com/t1", "lang": "en",
     "create_time": 1714500000, "likes": 100, "retweets": 50, "replies": 20,
     "quotes": 5, "views": 10000, "bookmarks": 10, "hotness_score": 1650.0},
]

MOCK_ENGAGEMENT_ROWS = [{
    "sum_like": 5000, "avg_like": 50.0, "max_like": 500,
    "sum_retweet": 2000, "avg_retweet": 20.0, "max_retweet": 200,
    "sum_reply": 1000, "avg_reply": 10.0, "max_reply": 100,
    "sum_quote": 300, "avg_quote": 3.0, "max_quote": 30,
    "sum_bookmark": 400, "avg_bookmark": 4.0, "max_bookmark": 40,
    "sum_view": 500000, "avg_view": 5000.0, "max_view": 50000,
    "total": 100,
}]


# ── 测试类 ──

class TestXPropagationAnalyzer:

    def setup_method(self):
        with patch.object(XPropagationAnalyzer, '__init__', lambda self: None):
            self.analyzer = XPropagationAnalyzer()
            self.analyzer._is_postgres = False
            self.analyzer.W_LIKE = 1.0
            self.analyzer.W_REPLY = 5.0
            self.analyzer.W_RETWEET = 10.0
            self.analyzer.W_VIEW = 0.1

    # ── 1. 数据结构 ──

    def test_dataclass_defaults(self):
        report = XPropagationReport()
        assert report.topic == ""
        assert report.total_tweets_analyzed == 0
        assert report.tweet_count_stats is None
        assert report.llm_summary == ""

        stats = TweetCountStats()
        assert stats.total_tweets == 0
        assert stats.type_distribution == {}

        hot = HotTweet()
        assert hot.hotness_score == 0.0
        assert hot.create_time is None
        print("  [PASS] dataclass defaults")

    # ── 2. 推文数量统计 ──

    def test_tweet_count_stats(self):
        with patch.object(self.analyzer, '_execute_query', return_value=MOCK_TWEET_COUNT_ROWS):
            stats = self.analyzer.get_tweet_count_stats(topic="test")
        assert stats.total_tweets == 100
        assert stats.original_tweets == 60
        assert stats.retweets == 20
        assert stats.type_distribution["tweet"] == 60
        assert stats.type_distribution["reply"] == 10
        print("  [PASS] tweet_count_stats")

    def test_tweet_count_stats_empty(self):
        with patch.object(self.analyzer, '_execute_query', return_value=[]):
            stats = self.analyzer.get_tweet_count_stats()
        assert stats.total_tweets == 0
        print("  [PASS] tweet_count_stats (empty)")

    # ── 3. 时间分布 ──

    def test_time_distribution(self):
        call_count = [0]
        def mock_query(q, p):
            call_count[0] += 1
            return MOCK_HOURLY_ROWS if call_count[0] == 1 else MOCK_DAILY_ROWS

        with patch.object(self.analyzer, '_execute_query', side_effect=mock_query):
            dist = self.analyzer.get_time_distribution()
        assert dist.peak_hour == 14
        assert dist.peak_hour_count == 25
        assert dist.peak_date == "2025-04-02"
        assert dist.peak_date_count == 70
        print("  [PASS] time_distribution")

    # ── 4. 关键词 ──

    def test_keyword_frequency(self):
        with patch.object(self.analyzer, '_execute_query', return_value=MOCK_CONTENT_ROWS):
            kf = self.analyzer.get_keyword_frequency(top_n=5)
        assert kf.total_documents == 3
        keywords = [w for w, _ in kf.top_keywords]
        assert "south" in keywords
        assert "africa" in keywords
        print("  [PASS] keyword_frequency")

    # ── 5. Hashtag ──

    def test_hashtag_frequency(self):
        with patch.object(self.analyzer, '_execute_query', return_value=MOCK_HASHTAG_ROWS):
            hf = self.analyzer.get_hashtag_frequency(top_n=5)
        assert hf.total_tweets_with_hashtags == 3
        tags = dict(hf.top_hashtags)
        assert tags["southafrica"] == 3
        assert tags["election2025"] == 2
        print("  [PASS] hashtag_frequency")

    def test_hashtag_malformed_json(self):
        rows = [{"hashtags": "not valid json"}, {"hashtags": '["ok"]'}]
        with patch.object(self.analyzer, '_execute_query', return_value=rows):
            hf = self.analyzer.get_hashtag_frequency()
        assert hf.total_tweets_with_hashtags == 1
        print("  [PASS] hashtag_frequency (malformed json)")

    # ── 6. 活跃账号 ──

    def test_active_accounts(self):
        call_count = [0]
        def mock_query(q, p):
            call_count[0] += 1
            if call_count[0] == 1:
                return MOCK_ACTIVE_ROWS
            elif call_count[0] == 2:
                return MOCK_CREATOR_ROWS
            else:
                return MOCK_UNIQUE_ROWS

        with patch.object(self.analyzer, '_execute_query', side_effect=mock_query):
            aa = self.analyzer.get_active_accounts(top_n=5)
        assert len(aa.top_accounts) == 2
        assert aa.top_accounts[0].username == "alice"
        assert aa.top_accounts[0].followers_count == 10000
        assert aa.top_accounts[0].is_verified is True
        assert aa.total_unique_users == 50
        assert aa.verified_user_count == 5
        print("  [PASS] active_accounts")

    # ── 7. 热门推文 ──

    def test_hot_tweet_ranking(self):
        with patch.object(self.analyzer, '_execute_query', return_value=MOCK_HOT_TWEET_ROWS):
            ranking = self.analyzer.get_hot_tweet_ranking(top_n=5)
        assert len(ranking.top_tweets) == 1
        t = ranking.top_tweets[0]
        assert t.tweet_id == "t1"
        assert t.like_count == 100
        assert t.hotness_score == 1650.0
        assert t.create_time is not None
        print("  [PASS] hot_tweet_ranking")

    # ── 8. 互动指标 ──

    def test_engagement_stats(self):
        with patch.object(self.analyzer, '_execute_query', return_value=MOCK_ENGAGEMENT_ROWS):
            es = self.analyzer.get_engagement_stats()
        assert es.total_likes == 5000
        assert es.avg_retweets == 20.0
        assert es.max_views == 50000
        print("  [PASS] engagement_stats")

    # ── 9. LLM 格式化 ──

    def test_format_report_for_llm(self):
        report = XPropagationReport(
            topic="South Africa",
            total_tweets_analyzed=100,
            tweet_count_stats=TweetCountStats(
                total_tweets=100, original_tweets=60, retweets=20, quotes=10, replies=10,
            ),
            keyword_frequency=KeywordFrequency(
                top_keywords=[("south", 50), ("africa", 45)], total_documents=100,
            ),
            engagement_stats=EngagementStats(
                total_likes=5000, total_retweets=2000, total_replies=1000, total_views=500000,
                avg_likes=50.0, avg_retweets=20.0, avg_replies=10.0,
            ),
        )
        text = self.analyzer.format_report_for_llm(report)
        assert "South Africa" in text
        assert "100" in text
        assert "south(50)" in text
        assert "均点赞: 50.0" in text
        print("  [PASS] format_report_for_llm")

    # ── 10. 图表配置 ──

    def test_to_chart_configs(self):
        report = XPropagationReport(
            topic="test",
            total_tweets_analyzed=100,
            tweet_count_stats=TweetCountStats(
                total_tweets=100, original_tweets=60, retweets=20, quotes=10, replies=10,
                type_distribution={"tweet": 60, "retweet": 20, "quote": 10, "reply": 10},
            ),
            time_distribution=TimeDistribution(
                hourly_counts={9: 15, 14: 25}, daily_counts={"2025-04-01": 50},
            ),
            keyword_frequency=KeywordFrequency(
                top_keywords=[("test", 10)], total_documents=50,
            ),
            engagement_stats=EngagementStats(
                avg_likes=50.0, avg_retweets=20.0, avg_replies=10.0,
                total_quotes=300, total_bookmarks=400,
            ),
        )
        configs = self.analyzer.to_chart_configs(report)
        widget_types = [c["widgetType"] for c in configs]
        # 当前实现仅输出 3 类图表（line/bar/radar），已移除 doughnut（推文类型分布噪声较大）
        assert "chart.js/line" in widget_types
        assert "chart.js/bar" in widget_types
        assert "chart.js/radar" in widget_types
        assert len(configs) == 3

        bar_hourly = next(c for c in configs if c["widgetId"] == "x-prop-hourly-dist")
        assert len(bar_hourly["data"]["labels"]) == 24
        assert bar_hourly["data"]["datasets"][0]["data"][14] == 25
        print("  [PASS] to_chart_configs")

    # ── 11. WHERE 构建 ──

    def test_build_where_empty(self):
        where, params = self.analyzer._build_where()
        assert where == "1=1"
        assert params == []
        print("  [PASS] _build_where (empty)")

    def test_build_where_topic_and_date(self):
        where, params = self.analyzer._build_where("test", "2025-01-01", "2025-01-31")
        assert "LIKE" in where
        assert "create_time" in where
        assert len(params) == 5
        print("  [PASS] _build_where (topic + date)")

    # ── 12. 方言兼容 ──

    def test_postgres_expressions(self):
        self.analyzer._is_postgres = True
        assert "TO_TIMESTAMP" in self.analyzer._from_unixtime_expr("`create_time`")
        assert "EXTRACT" in self.analyzer._hour_expr("ts")
        assert "::DATE" in self.analyzer._date_expr("ts")
        self.analyzer._is_postgres = False
        assert "FROM_UNIXTIME" in self.analyzer._from_unixtime_expr("`create_time`")
        assert "HOUR" in self.analyzer._hour_expr("ts")
        print("  [PASS] postgres/mysql dialect expressions")


# ── 运行 ──

def main():
    print("=" * 60)
    print("X平台传播特征分析模块 — 单元测试")
    print("=" * 60)

    test = TestXPropagationAnalyzer()
    methods = [m for m in dir(test) if m.startswith("test_")]

    passed, failed = 0, 0
    for name in sorted(methods):
        test.setup_method()
        try:
            getattr(test, name)()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print()
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
