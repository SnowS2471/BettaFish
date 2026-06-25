"""
Deep Search Agent 主类（InsightEngine 核心编排器）

职责概述
--------
InsightEngine 是「私有舆情数据库挖掘型」Agent：不访问外网，只查询本地由
MindSpider 爬虫写入的社媒数据库，再配合微调情感模型，产出深度舆情分析报告。

整体流程（research() 四步）：
    1. 生成报告结构  ReportStructureNode   -> 把 query 拆成 5 个分析段落
    2. 逐段落处理    _process_paragraphs   -> 首次搜索+总结，再做 N 轮反思补充
    3. 汇总最终报告  ReportFormattingNode  -> 把各段落拼成 Markdown
    4. 保存报告      _save_report          -> 写入 OUTPUT_DIR/*.md

单段落内的数据流：
    LLM 产出检索词 -> keyword_optimizer 优化成网民口语 -> MediaCrawlerDB 模板化 SQL
    -> 去重 -> 聚类采样 -> 情感分析 -> 结果喂回 LLM 生成/更新段落正文。

与其它子模块的关系：
    - llms/    OpenAI 兼容的 LLM 客户端
    - nodes/   各处理节点（搜索 / 总结 / 反思 / 结构 / 格式化）
    - state/   贯穿全程的可变状态对象 State（段落、搜索历史、总结都挂在上面）
    - tools/   数据库查询、关键词优化、情感分析
    - utils/   文本处理、数据库连接
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer  # 句向量模型，用于对搜索结果聚类采样
from sklearn.cluster import KMeans                      # KMeans 聚类，给搜索结果分簇、去冗余

from .llms import LLMClient
from .nodes import (
    FirstSearchNode,        # 首次搜索：根据段落标题/内容生成检索词
    FirstSummaryNode,       # 首次总结：把搜索结果写成段落正文
    ReflectionNode,         # 反思搜索：找出信息缺口并生成补充检索词
    ReflectionSummaryNode,  # 反思总结：用补充结果更新段落正文
    ReportFormattingNode,   # 报告格式化：把各段落汇总成最终 Markdown
    ReportStructureNode,    # 报告结构：把 query 拆成多个段落大纲
)
from .state import State
from .tools import (
    DBResponse,                       # 数据库查询的统一返回封装
    MediaCrawlerDB,                   # 本地舆情数据库查询工具集（SQL 模板，非 LLM 生成 SQL）
    keyword_optimizer,                # 关键词优化中间件（Qwen，把查询词转成网民口语）
    multilingual_sentiment_analyzer,  # 多语言情感分析器（全局单例，懒加载模型）
)
from .utils import format_search_results_for_prompt
from .utils.config import Settings, settings

# ============ 聚类采样开关与参数 ============
# 单次搜索去重后若结果过多，用句向量聚类挑出代表性样本，避免把海量重复内容塞给 LLM
# （既省 token，又能提升喂给 LLM 的信息多样性）。
ENABLE_CLUSTERING: bool = True   # 是否启用聚类采样
MAX_CLUSTERED_RESULTS: int = 50  # 聚类后最大返回结果数
RESULTS_PER_CLUSTER: int = 5     # 每个聚类返回的结果数（簇内按热度取前 N）


class DeepSearchAgent:
    """Deep Search Agent 主类。

    一个实例对应一次完整的研究会话：内部持有一个 LLM 客户端、一套数据库查询工具、
    一个情感分析器，以及贯穿全程、不断被各节点读写的 State 对象。
    """

    def __init__(self, config: Optional[Settings] = None):
        """
        初始化Deep Search Agent

        Args:
            config: 可选配置对象（不填则用全局settings）
        """
        # 允许调用方传入独立配置（Streamlit 会按需构造一个临时 Settings），否则用全局单例
        self.config = config or settings

        # 初始化LLM客户端（读取 INSIGHT_ENGINE_* 配置，默认 Kimi）
        self.llm_client = self._initialize_llm()

        # 初始化搜索工具集：本地舆情数据库查询客户端（封装了 5 个模板化 SQL 工具）
        self.search_agency = MediaCrawlerDB()

        # 聚类用的句向量模型较重，采用懒加载：首次需要采样时才真正下载/加载
        self._clustering_model = None

        # 情感分析器使用全局单例（同样懒加载模型），多个 Agent 实例共享同一份权重
        self.sentiment_analyzer = multilingual_sentiment_analyzer

        # 初始化各处理节点（它们共享同一个 llm_client）
        self._initialize_nodes()

        # 全局状态对象：段落、搜索历史、总结、最终报告都挂在这里
        self.state = State()

        # 确保报告输出目录存在（如 insight_engine_streamlit_reports/）
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        logger.info(f"Insight Agent已初始化")
        logger.info(f"使用LLM: {self.llm_client.get_model_info()}")
        logger.info(f"搜索工具集: MediaCrawlerDB (支持5种本地数据库查询工具)")
        logger.info(f"情感分析: WeiboMultilingualSentiment (支持22种语言的情感分析)")

    def _initialize_llm(self) -> LLMClient:
        """初始化LLM客户端（使用 INSIGHT_ENGINE_* 三件套：key / model / base_url）"""
        return LLMClient(
            api_key=self.config.INSIGHT_ENGINE_API_KEY,
            model_name=self.config.INSIGHT_ENGINE_MODEL_NAME,
            base_url=self.config.INSIGHT_ENGINE_BASE_URL,
        )

    def _initialize_nodes(self):
        """初始化处理节点。

        五个节点分两类：
        - 搜索类（FirstSearchNode / ReflectionNode）：只产出检索词，不改 State；
        - 状态修改类（First/Reflection SummaryNode、ReportFormattingNode）：会写回 State。
        全部共用同一个 llm_client。
        """
        self.first_search_node = FirstSearchNode(self.llm_client)
        self.reflection_node = ReflectionNode(self.llm_client)
        self.first_summary_node = FirstSummaryNode(self.llm_client)
        self.reflection_summary_node = ReflectionSummaryNode(self.llm_client)
        self.report_formatting_node = ReportFormattingNode(self.llm_client)

    def _get_clustering_model(self):
        """懒加载聚类模型。

        首次调用时才下载/加载多语言句向量模型（较重，约百 MB 级），之后复用。
        """
        if self._clustering_model is None:
            logger.info("  加载聚类模型 (paraphrase-multilingual-MiniLM-L12-v2)...")
            self._clustering_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
        return self._clustering_model

    def _validate_date_format(self, date_str: str) -> bool:
        """
        验证日期格式是否为YYYY-MM-DD

        用于校验 LLM 给出的 start_date/end_date：只有格式与日期都合法，才会把它们
        传给按日期过滤的数据库工具；否则降级为不带时间的全局搜索（见搜索方法）。

        Args:
            date_str: 日期字符串

        Returns:
            是否为有效格式
        """
        if not date_str:
            return False

        # 先用正则卡格式（四位年-两位月-两位日）
        pattern = r"^\d{4}-\d{2}-\d{2}$"
        if not re.match(pattern, date_str):
            return False

        # 再用 strptime 验证日期本身是否存在（如 2025-02-30 会被拦下）
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def _cluster_and_sample_results(
        self,
        results: List,
        max_results: int = MAX_CLUSTERED_RESULTS,
        results_per_cluster: int = RESULTS_PER_CLUSTER,
    ) -> List:
        """
        对搜索结果进行聚类并采样

        思路：把每条结果的文本编码成向量 -> KMeans 分成若干主题簇 ->
        每个簇内按热度取前 N 条，得到一份「去同质化」的代表性子集。
        这样既能覆盖不同话题角度，又能把喂给 LLM 的条数压到 max_results 以内。

        Args:
            results: 搜索结果列表
            max_results: 最大返回结果数
            results_per_cluster: 每个聚类返回的结果数

        Returns:
            采样后的结果列表
        """
        # 结果本就不多则无需聚类，直接返回
        if len(results) <= max_results:
            return results

        try:
            # 1) 提取文本（每条截断到 500 字，控制编码开销）
            texts = [r.title_or_content[:500] for r in results]

            # 2) 句向量编码
            model = self._get_clustering_model()
            embeddings = model.encode(texts, show_progress_bar=False)

            # 3) 计算簇数：约等于 max_results/每簇条数，且夹在 [2, len(results)] 之间
            n_clusters = min(max(2, max_results // results_per_cluster), len(results))

            # 4) KMeans 聚类（固定 random_state 保证可复现）
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            # 5) 从每个簇内按热度降序取前 results_per_cluster 条
            sampled_results = []
            for cluster_id in range(n_clusters):
                cluster_indices = np.flatnonzero(labels == cluster_id)
                cluster_results = [(results[i], i) for i in cluster_indices]
                # 注：全局 LIKE 搜索得到的结果 hotness_score 多为 0，此处排序近似无序
                cluster_results.sort(
                    key=lambda x: x[0].hotness_score or 0, reverse=True
                )

                for result, _ in cluster_results[:results_per_cluster]:
                    sampled_results.append(result)
                    if len(sampled_results) >= max_results:
                        break

                if len(sampled_results) >= max_results:
                    break

            logger.info(
                f"  聚类完成: {len(results)} 条 -> {n_clusters} 个主题 -> {len(sampled_results)} 条代表性结果"
            )
            return sampled_results

        except Exception as e:
            # 聚类失败（如模型加载异常）不应阻断主流程，退化为「取前 max_results 条」
            logger.warning(f"  聚类失败，返回前{max_results}条: {str(e)}")
            return results[:max_results]

    def execute_search_tool(self, tool_name: str, query: str, **kwargs) -> DBResponse:
        """
        执行指定的数据库查询工具（集成关键词优化中间件和情感分析）

        这是「检索词 -> 数据库结果」的中枢，整体分三类处理：
          1) search_hot_content：无需 query，直接查热榜，可选做情感分析；
          2) analyze_sentiment：纯情感分析工具，不查库；
          3) 其余按话题检索的工具：先经 keyword_optimizer 把 query 扩成多个网民口语
             关键词，对每个词查库后合并 -> 去重 -> 聚类采样 -> 情感分析。

        说明：在 research() 主流程里，搜索节点的 process_output 只保留 search_query，
        因此实际传入的 tool_name 多数会回退为默认的 "search_topic_globally"。

        Args:
            tool_name: 工具名称，可选值：
                - "search_hot_content": 查找热点内容
                - "search_topic_globally": 全局话题搜索
                - "search_topic_by_date": 按日期搜索话题
                - "get_comments_for_topic": 获取话题评论
                - "search_topic_on_platform": 平台定向搜索
                - "analyze_sentiment": 对查询结果进行情感分析
            query: 搜索关键词/话题
            **kwargs: 额外参数（如start_date, end_date, platform, limit, enable_sentiment等）
                     enable_sentiment: 是否自动对搜索结果进行情感分析（默认True）

        Returns:
            DBResponse对象（可能包含情感分析结果）
        """
        logger.info(f"  → 执行数据库查询工具: {tool_name}")

        # 【分支一】热点内容搜索：本身按热度取榜，无需 query，也就跳过关键词优化
        if tool_name == "search_hot_content":
            time_period = kwargs.get("time_period", "week")
            limit = kwargs.get("limit", 100)
            response = self.search_agency.search_hot_content(
                time_period=time_period, limit=limit
            )

            # 默认对热榜结果做情感分析，并把结果挂到 parameters 里随响应返回
            enable_sentiment = kwargs.get("enable_sentiment", True)
            if enable_sentiment and response.results and len(response.results) > 0:
                logger.info(f"  🎭 开始对热点内容进行情感分析...")
                sentiment_analysis = self._perform_sentiment_analysis(response.results)
                if sentiment_analysis:
                    # 将情感分析结果添加到响应的parameters中
                    response.parameters["sentiment_analysis"] = sentiment_analysis
                    logger.info(f"  ✅ 情感分析完成")

            return response

        # 【分支二】独立情感分析工具：不查数据库，只对给定文本做情感分析
        if tool_name == "analyze_sentiment":
            texts = kwargs.get("texts", query)  # 可以通过texts参数传递，或使用query
            sentiment_result = self.analyze_sentiment_only(texts)

            # 包装成统一的 DBResponse（results 为空，分析结果放在 metadata）
            return DBResponse(
                tool_name="analyze_sentiment",
                parameters={
                    "texts": texts if isinstance(texts, list) else [texts],
                    **kwargs,
                },
                results=[],  # 情感分析不返回搜索结果
                results_count=0,
                metadata=sentiment_result,
            )

        # 【分支三】按话题检索的工具：先做关键词优化（把官方/书面化查询转成网民口语词）
        optimized_response = keyword_optimizer.optimize_keywords(
            original_query=query, context=f"使用{tool_name}工具进行查询"
        )

        logger.info(f"  🔍 原始查询: '{query}'")
        logger.info(f"  ✨ 优化后关键词: {optimized_response.optimized_keywords}")

        # 对每个优化关键词分别查库，把多次结果汇总到 all_results
        all_results = []
        total_count = 0

        for keyword in optimized_response.optimized_keywords:
            logger.info(f"    查询关键词: '{keyword}'")

            try:
                if tool_name == "search_topic_globally":
                    # 统一用配置里的默认上限，刻意忽略 agent/LLM 传入的 limit_per_table
                    limit_per_table = (
                        self.config.DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE
                    )
                    response = self.search_agency.search_topic_globally(
                        topic=keyword, limit_per_table=limit_per_table
                    )
                elif tool_name == "search_topic_by_date":
                    start_date = kwargs.get("start_date")
                    end_date = kwargs.get("end_date")
                    # 同样使用配置默认上限
                    limit_per_table = (
                        self.config.DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE
                    )
                    if not start_date or not end_date:
                        raise ValueError(
                            "search_topic_by_date工具需要start_date和end_date参数"
                        )
                    response = self.search_agency.search_topic_by_date(
                        topic=keyword,
                        start_date=start_date,
                        end_date=end_date,
                        limit_per_table=limit_per_table,
                    )
                elif tool_name == "get_comments_for_topic":
                    # 总配额按关键词个数均摊（避免词多时评论爆量），但每词保底 50 条
                    limit = self.config.DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT // len(
                        optimized_response.optimized_keywords
                    )
                    limit = max(limit, 50)
                    response = self.search_agency.get_comments_for_topic(
                        topic=keyword, limit=limit
                    )
                elif tool_name == "search_topic_on_platform":
                    platform = kwargs.get("platform")
                    start_date = kwargs.get("start_date")
                    end_date = kwargs.get("end_date")
                    # 配额同样按关键词个数均摊，每词保底 30 条
                    limit = self.config.DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT // len(
                        optimized_response.optimized_keywords
                    )
                    limit = max(limit, 30)
                    if not platform:
                        raise ValueError("search_topic_on_platform工具需要platform参数")
                    response = self.search_agency.search_topic_on_platform(
                        platform=platform,
                        topic=keyword,
                        start_date=start_date,
                        end_date=end_date,
                        limit=limit,
                    )
                else:
                    # 工具名未识别时兜底为全局搜索，保证流程不中断
                    logger.info(f"    未知的搜索工具: {tool_name}，使用默认全局搜索")
                    response = self.search_agency.search_topic_globally(
                        topic=keyword,
                        limit_per_table=self.config.DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE,
                    )

                # 累加本关键词的查询结果
                if response.results:
                    logger.info(f"     找到 {len(response.results)} 条结果")
                    all_results.extend(response.results)
                    total_count += len(response.results)
                else:
                    logger.info(f"     未找到结果")

            except Exception as e:
                # 单个关键词查询失败不影响其它关键词，记录后继续
                logger.error(f"      查询'{keyword}'时出错: {str(e)}")
                continue

        # 多关键词结果合并后先去重（按 url / 内容前缀），再视需要做聚类采样压缩条数
        unique_results = self._deduplicate_results(all_results)
        logger.info(f"  总计找到 {total_count} 条结果，去重后 {len(unique_results)} 条")

        if ENABLE_CLUSTERING:
            unique_results = self._cluster_and_sample_results(
                unique_results,
                max_results=MAX_CLUSTERED_RESULTS,
                results_per_cluster=RESULTS_PER_CLUSTER,
            )

        # 打包成统一响应；tool_name 加 _optimized 后缀以表明经过了关键词优化中间件
        integrated_response = DBResponse(
            tool_name=f"{tool_name}_optimized",
            parameters={
                "original_query": query,
                "optimized_keywords": optimized_response.optimized_keywords,
                "optimization_reasoning": optimized_response.reasoning,
                **kwargs,
            },
            results=unique_results,
            results_count=len(unique_results),
        )

        # 默认对最终结果集做情感分析，结果挂到 parameters["sentiment_analysis"]
        enable_sentiment = kwargs.get("enable_sentiment", True)
        if enable_sentiment and unique_results and len(unique_results) > 0:
            logger.info(f"  🎭 开始对搜索结果进行情感分析...")
            sentiment_analysis = self._perform_sentiment_analysis(unique_results)
            if sentiment_analysis:
                # 将情感分析结果添加到响应的parameters中
                integrated_response.parameters["sentiment_analysis"] = (
                    sentiment_analysis
                )
                logger.info(f"  ✅ 情感分析完成")

        return integrated_response

    def _deduplicate_results(self, results: List) -> List:
        """
        去重搜索结果

        多个关键词命中同一条内容时会重复，这里按「URL 优先、无 URL 则取内容前 100 字」
        作为唯一标识做去重，保持首次出现的顺序。
        """
        seen = set()
        unique_results = []

        for result in results:
            # 评论类结果通常没有 url，退化用内容前缀当指纹
            identifier = result.url if result.url else result.title_or_content[:100]
            if identifier not in seen:
                seen.add(identifier)
                unique_results.append(result)

        return unique_results

    def _perform_sentiment_analysis(self, results: List) -> Optional[Dict[str, Any]]:
        """
        对搜索结果执行情感分析（供各搜索分支复用）

        模型首次使用时才加载；若依赖缺失或被禁用则优雅降级（透传原文，不报错）。

        Args:
            results: 搜索结果列表（QueryResult）

        Returns:
            情感分析结果字典，如果失败则返回None
        """
        try:
            # 懒加载：仅在「未初始化且未禁用」时尝试加载模型
            if (
                not self.sentiment_analyzer.is_initialized
                and not self.sentiment_analyzer.is_disabled
            ):
                logger.info("    初始化情感分析模型...")
                if not self.sentiment_analyzer.initialize():
                    logger.info("     情感分析模型初始化失败，将直接透传原始文本")
            elif self.sentiment_analyzer.is_disabled:
                logger.info("     情感分析功能已禁用，直接透传原始文本")

            # QueryResult -> dict，分析器从 content 字段取文本
            results_dict = []
            for result in results:
                result_dict = {
                    "content": result.title_or_content,
                    "platform": result.platform,
                    "author": result.author_nickname,
                    "url": result.url,
                    "publish_time": str(result.publish_time)
                    if result.publish_time
                    else None,
                }
                results_dict.append(result_dict)

            # 批量分析，min_confidence=0.5 用于筛选高置信度样本
            sentiment_analysis = self.sentiment_analyzer.analyze_query_results(
                query_results=results_dict, text_field="content", min_confidence=0.5
            )

            # 只取内层的 sentiment_analysis 子字典（含分布、摘要、高置信样本等）
            return sentiment_analysis.get("sentiment_analysis")

        except Exception as e:
            # 情感分析属于增强项，出错不应影响搜索主流程，返回 None 由调用方忽略
            logger.exception(f"    ❌ 情感分析过程中发生错误: {str(e)}")
            return None

    def analyze_sentiment_only(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """
        独立的情感分析工具

        对应 execute_search_tool 里 "analyze_sentiment" 分支：不查数据库，只对调用方
        给定的文本做情感分析。支持单条与批量两种输入，返回结构统一（含 success、
        total_analyzed、results 等字段），模型不可用时给出 warning 并透传。

        Args:
            texts: 单个文本或文本列表

        Returns:
            情感分析结果
        """
        logger.info(f"  → 执行独立情感分析")

        try:
            # 懒加载模型（与 _perform_sentiment_analysis 相同的初始化/降级逻辑）
            if (
                not self.sentiment_analyzer.is_initialized
                and not self.sentiment_analyzer.is_disabled
            ):
                logger.info("    初始化情感分析模型...")
                if not self.sentiment_analyzer.initialize():
                    logger.info("     情感分析模型初始化失败，将直接透传原始文本")
            elif self.sentiment_analyzer.is_disabled:
                logger.warning("     情感分析功能已禁用，直接透传原始文本")

            # 单文本走 analyze_single_text，多文本走 analyze_batch，分别封装成统一返回结构
            if isinstance(texts, str):
                result = self.sentiment_analyzer.analyze_single_text(texts)
                result_dict = result.__dict__
                response = {
                    "success": result.success and result.analysis_performed,
                    "total_analyzed": 1
                    if result.analysis_performed and result.success
                    else 0,
                    "results": [result_dict],
                }
                if not result.analysis_performed:
                    response["success"] = False
                    response["warning"] = (
                        result.error_message or "情感分析功能不可用，已直接返回原始文本"
                    )
                return response
            else:
                texts_list = list(texts)
                batch_result = self.sentiment_analyzer.analyze_batch(
                    texts_list, show_progress=True
                )
                response = {
                    "success": batch_result.analysis_performed
                    and batch_result.success_count > 0,
                    "total_analyzed": batch_result.total_processed
                    if batch_result.analysis_performed
                    else 0,
                    "success_count": batch_result.success_count,
                    "failed_count": batch_result.failed_count,
                    "average_confidence": batch_result.average_confidence
                    if batch_result.analysis_performed
                    else 0.0,
                    "results": [result.__dict__ for result in batch_result.results],
                }
                if not batch_result.analysis_performed:
                    warning = next(
                        (
                            r.error_message
                            for r in batch_result.results
                            if r.error_message
                        ),
                        "情感分析功能不可用，已直接返回原始文本",
                    )
                    response["success"] = False
                    response["warning"] = warning
                return response

        except Exception as e:
            logger.exception(f"    ❌ 情感分析过程中发生错误: {str(e)}")
            return {"success": False, "error": str(e), "results": []}

    def research(self, query: str, save_report: bool = True) -> str:
        """
        执行深度研究（一体化入口）

        把四个步骤串起来跑完。CLI / 编排器用这个；Streamlit 为了显示进度条则不调它，
        而是手动依次调用下面的 _generate_report_structure / _process_paragraphs 等私有方法。

        Args:
            query: 研究查询
            save_report: 是否保存报告到文件

        Returns:
            最终报告内容
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"开始深度研究: {query}")
        logger.info(f"{'=' * 60}")

        try:
            # Step 1: 生成报告结构（把 query 拆成若干段落）
            self._generate_report_structure(query)

            # Step 2: 逐段落搜索 + 总结 + 反思
            self._process_paragraphs()

            # Step 3: 汇总成最终 Markdown 报告
            final_report = self._generate_final_report()

            # Step 4: 落盘（可选）
            if save_report:
                self._save_report(final_report)

            logger.info("深度研究完成！")

            return final_report

        except Exception as e:
            # 任一步异常都向上抛，交给调用方（Streamlit/编排器）统一展示错误
            logger.exception(f"研究过程中发生错误: {str(e)}")
            raise e

    def cross_platform_research(self, topic: str, **kwargs) -> "CrossPlatformReport":
        """
        执行跨平台舆情对比分析（南非新闻 vs X 推文）

        这是独立于 research() 主流程的旁路入口（毕设扩展），主要供南非专题报告调用：
        分别取新闻端与社交端数据，过滤、对比关键词/情感/时间趋势，再让 LLM 总结。

        Args:
            topic: 分析主题
            **kwargs: 可选参数
                news_limit: 新闻搜索数量上限 (默认 100)
                tweet_limit: 推文搜索数量上限 (默认 200)
                relevance_threshold: 推文相关性阈值 (默认 0.35)

        Returns:
            CrossPlatformReport 结构化对比报告
        """
        # 延迟导入：该节点依赖较重（相关性过滤等），仅在真正用到时才加载
        from .nodes.cross_platform_node import CrossPlatformAnalysisNode

        logger.info(f"\n{'=' * 60}")
        logger.info(f"开始跨平台对比分析: {topic}")
        logger.info(f"{'=' * 60}")

        node = CrossPlatformAnalysisNode(
            llm_client=self.llm_client,
            search_db=self.search_agency,
            sentiment_analyzer=self.sentiment_analyzer,
            embedding_model=self._get_clustering_model(),
        )
        report = node.run(topic, **kwargs)

        logger.info(f"跨平台分析完成: {topic}")
        return report

    def x_propagation_analysis(self, topic: str = "", **kwargs) -> "XPropagationReport":
        """
        执行X平台传播特征分析

        同样是独立旁路入口（毕设扩展）：只针对 twitter_tweet 表，统计推文量、时间分布、
        关键词/话题标签、活跃账号、热门推文与互动热度等传播特征，再让 LLM 出总结。

        Args:
            topic: 分析主题（可选，为空则分析全部推文）
            **kwargs: 可选参数
                start_date: 开始日期 YYYY-MM-DD
                end_date: 结束日期 YYYY-MM-DD
                top_n: 排行榜数量 (默认 20)

        Returns:
            XPropagationReport 结构化传播分析报告
        """
        # 延迟导入，避免主流程加载这个仅供专题使用的分析节点
        from .nodes.x_propagation_node import XPropagationAnalysisNode

        logger.info(f"\n{'=' * 60}")
        logger.info(f"开始X平台传播特征分析: {topic or '全部'}")
        logger.info(f"{'=' * 60}")

        node = XPropagationAnalysisNode(llm_client=self.llm_client)
        report = node.run(topic, **kwargs)

        logger.info(f"X平台传播分析完成: {topic or '全部'}")
        return report

    def _generate_report_structure(self, query: str):
        """生成报告结构（Step 1）。

        交给 ReportStructureNode：调 LLM 把 query 规划成多个段落（标题 + 预期内容），
        并写入 self.state（含 query 与 report_title）。
        """
        logger.info(f"\n[步骤 1] 生成报告结构...")

        # 节点需要 query 才能规划，因此在构造时就把 query 传进去
        report_structure_node = ReportStructureNode(self.llm_client, query)

        # mutate_state 会就地生成段落并返回更新后的 state
        self.state = report_structure_node.mutate_state(state=self.state)

        _message = f"报告结构已生成，共 {len(self.state.paragraphs)} 个段落:"
        for i, paragraph in enumerate(self.state.paragraphs, 1):
            _message += f"\n  {i}. {paragraph.title}"
        logger.info(_message)

    def _process_paragraphs(self):
        """处理所有段落（Step 2）。

        逐个段落串行处理：先做一次「搜索 + 总结」，再做 MAX_REFLECTIONS 轮反思补充，
        最后标记该段落完成。段落之间相互独立、顺序执行。
        """
        total_paragraphs = len(self.state.paragraphs)

        for i in range(total_paragraphs):
            logger.info(
                f"\n[步骤 2.{i + 1}] 处理段落: {self.state.paragraphs[i].title}"
            )
            logger.info("-" * 50)

            # 2a. 首次搜索 + 首次总结，得到段落初稿
            self._initial_search_and_summary(i)

            # 2b. 多轮反思：找缺口 -> 补充搜索 -> 更新段落内容
            self._reflection_loop(i)

            # 2c. 标记该段落研究完成
            self.state.paragraphs[i].research.mark_completed()

            progress = (i + 1) / total_paragraphs * 100
            logger.info(f"段落处理完成 ({progress:.1f}%)")

    def _initial_search_and_summary(self, paragraph_index: int):
        """执行初始搜索和总结（Step 2a）。

        流程：用段落标题/内容让 LLM 生成检索词 -> 组装查询参数 -> 查库 ->
        把结果转成 LLM 友好格式 -> 写入搜索历史 -> 让总结节点写出段落初稿。
        """
        paragraph = self.state.paragraphs[paragraph_index]

        # 喂给搜索节点的输入：段落标题 + 预期内容
        search_input = {"title": paragraph.title, "content": paragraph.content}

        # 让 LLM 生成检索词（及理由）
        logger.info("  - 生成搜索查询...")
        search_output = self.first_search_node.run(search_input)
        search_query = search_output["search_query"]
        # 注：FirstSearchNode.process_output 通常只回传 search_query/reasoning，
        # 故这里的 search_tool 多数会取默认值 search_topic_globally（详见 process_output）
        search_tool = search_output.get(
            "search_tool", "search_topic_globally"
        )  # 默认工具
        reasoning = search_output["reasoning"]

        logger.info(f"  - 搜索查询: {search_query}")
        logger.info(f"  - 选择的工具: {search_tool}")
        logger.info(f"  - 推理: {reasoning}")

        # 执行搜索
        logger.info("  - 执行数据库查询...")

        # 根据所选工具组装额外参数；日期/平台缺失或非法时，统一降级为全局搜索
        search_kwargs = {}

        # 需要日期的工具：校验 LLM 给的 start_date/end_date，非法则回退 search_topic_globally
        if search_tool in ["search_topic_by_date", "search_topic_on_platform"]:
            start_date = search_output.get("start_date")
            end_date = search_output.get("end_date")

            if start_date and end_date:
                # 验证日期格式
                if self._validate_date_format(
                    start_date
                ) and self._validate_date_format(end_date):
                    search_kwargs["start_date"] = start_date
                    search_kwargs["end_date"] = end_date
                    logger.info(f"  - 时间范围: {start_date} 到 {end_date}")
                else:
                    logger.info(f"    日期格式错误（应为YYYY-MM-DD），改用全局搜索")
                    logger.info(
                        f"      提供的日期: start_date={start_date}, end_date={end_date}"
                    )
                    search_tool = "search_topic_globally"
            elif search_tool == "search_topic_by_date":
                logger.info(f"    search_topic_by_date工具缺少时间参数，改用全局搜索")
                search_tool = "search_topic_globally"

        # 处理需要平台参数的工具
        if search_tool == "search_topic_on_platform":
            platform = search_output.get("platform")
            if platform:
                search_kwargs["platform"] = platform
                logger.info(f"  - 指定平台: {platform}")
            else:
                logger.warning(
                    f"    search_topic_on_platform工具缺少平台参数，改用全局搜索"
                )
                search_tool = "search_topic_globally"

        # 处理限制参数，使用配置文件中的默认值而不是agent提供的参数
        if search_tool == "search_hot_content":
            time_period = search_output.get("time_period", "week")
            limit = self.config.DEFAULT_SEARCH_HOT_CONTENT_LIMIT
            search_kwargs["time_period"] = time_period
            search_kwargs["limit"] = limit
        elif search_tool in ["search_topic_globally", "search_topic_by_date"]:
            if search_tool == "search_topic_globally":
                limit_per_table = (
                    self.config.DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE
                )
            else:  # search_topic_by_date
                limit_per_table = (
                    self.config.DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE
                )
            search_kwargs["limit_per_table"] = limit_per_table
        elif search_tool in ["get_comments_for_topic", "search_topic_on_platform"]:
            if search_tool == "get_comments_for_topic":
                limit = self.config.DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT
            else:  # search_topic_on_platform
                limit = self.config.DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT
            search_kwargs["limit"] = limit

        search_response = self.execute_search_tool(
            search_tool, search_query, **search_kwargs
        )

        # 转换为兼容格式
        search_results = []
        if search_response and search_response.results:
            # 使用配置文件控制传递给LLM的结果数量，0表示不限制
            if self.config.MAX_SEARCH_RESULTS_FOR_LLM > 0:
                max_results = min(
                    len(search_response.results), self.config.MAX_SEARCH_RESULTS_FOR_LLM
                )
            else:
                max_results = len(search_response.results)  # 不限制，传递所有结果
            for result in search_response.results[:max_results]:
                search_results.append(
                    {
                        "title": result.title_or_content,
                        "url": result.url or "",
                        "content": result.title_or_content,
                        "score": result.hotness_score,
                        "raw_content": result.title_or_content,
                        "published_date": result.publish_time.isoformat()
                        if result.publish_time
                        else None,
                        "platform": result.platform,
                        "content_type": result.content_type,
                        "author": result.author_nickname,
                        "engagement": result.engagement,
                    }
                )

        if search_results:
            _message = f"  - 找到 {len(search_results)} 个搜索结果"
            for j, result in enumerate(search_results, 1):
                date_info = (
                    f" (发布于: {result.get('published_date', 'N/A')})"
                    if result.get("published_date")
                    else ""
                )
                _message += f"\n    {j}. {result['title'][:50]}...{date_info}"
            logger.info(_message)
        else:
            logger.info("  - 未找到搜索结果")

        # 把本次结果并入段落的搜索历史（State 持久记录，便于回溯与序列化）
        paragraph.research.add_search_results(search_query, search_results)

        # 生成初始总结：把搜索结果交给总结节点，写出该段落的初稿
        logger.info("  - 生成初始总结...")
        summary_input = {
            "title": paragraph.title,
            "content": paragraph.content,
            "search_query": search_query,
            # format_search_results_for_prompt 会按 MAX_CONTENT_LENGTH 截断，避免超长上下文
            "search_results": format_search_results_for_prompt(
                search_results, self.config.MAX_CONTENT_LENGTH
            ),
        }

        # mutate_state 内部会调 LLM 生成总结并写入 paragraph.research.latest_summary
        self.state = self.first_summary_node.mutate_state(
            summary_input, self.state, paragraph_index
        )

        logger.info("  - 初始总结完成")

    def _reflection_loop(self, paragraph_index: int):
        """执行反思循环（Step 2b）。

        固定循环 MAX_REFLECTIONS 次（不是按质量动态停止）。每轮把「段落当前内容」连同
        标题一起交给反思节点，让 LLM 找出信息缺口、生成补充检索词，再查库并据此更新段落。
        其余装配逻辑与 _initial_search_and_summary 基本一致。
        """
        paragraph = self.state.paragraphs[paragraph_index]

        for reflection_i in range(self.config.MAX_REFLECTIONS):
            logger.info(f"  - 反思 {reflection_i + 1}/{self.config.MAX_REFLECTIONS}...")

            # 反思输入比首次多一个 paragraph_latest_state，让 LLM 知道「已经写了什么」
            reflection_input = {
                "title": paragraph.title,
                "content": paragraph.content,
                "paragraph_latest_state": paragraph.research.latest_summary,
            }

            # 让 LLM 基于当前内容找缺口并生成补充检索词
            reflection_output = self.reflection_node.run(reflection_input)
            search_query = reflection_output["search_query"]
            # 同 _initial：search_tool 多数会回退默认值（process_output 只回传 query/reasoning）
            search_tool = reflection_output.get(
                "search_tool", "search_topic_globally"
            )  # 默认工具
            reasoning = reflection_output["reasoning"]

            logger.info(f"    反思查询: {search_query}")
            logger.info(f"    选择的工具: {search_tool}")
            logger.info(f"    反思推理: {reasoning}")

            # 执行反思搜索
            # 处理特殊参数
            search_kwargs = {}

            # 处理需要日期的工具
            if search_tool in ["search_topic_by_date", "search_topic_on_platform"]:
                start_date = reflection_output.get("start_date")
                end_date = reflection_output.get("end_date")

                if start_date and end_date:
                    # 验证日期格式
                    if self._validate_date_format(
                        start_date
                    ) and self._validate_date_format(end_date):
                        search_kwargs["start_date"] = start_date
                        search_kwargs["end_date"] = end_date
                        logger.info(f"    时间范围: {start_date} 到 {end_date}")
                    else:
                        logger.info(
                            f"      日期格式错误（应为YYYY-MM-DD），改用全局搜索"
                        )
                        logger.info(
                            f"        提供的日期: start_date={start_date}, end_date={end_date}"
                        )
                        search_tool = "search_topic_globally"
                elif search_tool == "search_topic_by_date":
                    logger.warning(
                        f"      search_topic_by_date工具缺少时间参数，改用全局搜索"
                    )
                    search_tool = "search_topic_globally"

            # 处理需要平台参数的工具
            if search_tool == "search_topic_on_platform":
                platform = reflection_output.get("platform")
                if platform:
                    search_kwargs["platform"] = platform
                    logger.info(f"    指定平台: {platform}")
                else:
                    logger.warning(
                        f"      search_topic_on_platform工具缺少平台参数，改用全局搜索"
                    )
                    search_tool = "search_topic_globally"

            # 处理限制参数
            if search_tool == "search_hot_content":
                time_period = reflection_output.get("time_period", "week")
                # 使用配置文件中的默认值，不允许agent控制limit参数
                limit = self.config.DEFAULT_SEARCH_HOT_CONTENT_LIMIT
                search_kwargs["time_period"] = time_period
                search_kwargs["limit"] = limit
            elif search_tool in ["search_topic_globally", "search_topic_by_date"]:
                # 使用配置文件中的默认值，不允许agent控制limit_per_table参数
                if search_tool == "search_topic_globally":
                    limit_per_table = (
                        self.config.DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE
                    )
                else:  # search_topic_by_date
                    limit_per_table = (
                        self.config.DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE
                    )
                search_kwargs["limit_per_table"] = limit_per_table
            elif search_tool in ["get_comments_for_topic", "search_topic_on_platform"]:
                # 使用配置文件中的默认值，不允许agent控制limit参数
                if search_tool == "get_comments_for_topic":
                    limit = self.config.DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT
                else:  # search_topic_on_platform
                    limit = self.config.DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT
                search_kwargs["limit"] = limit

            search_response = self.execute_search_tool(
                search_tool, search_query, **search_kwargs
            )

            # 转换为兼容格式
            search_results = []
            if search_response and search_response.results:
                # 使用配置文件控制传递给LLM的结果数量，0表示不限制
                if self.config.MAX_SEARCH_RESULTS_FOR_LLM > 0:
                    max_results = min(
                        len(search_response.results),
                        self.config.MAX_SEARCH_RESULTS_FOR_LLM,
                    )
                else:
                    max_results = len(search_response.results)  # 不限制，传递所有结果
                for result in search_response.results[:max_results]:
                    search_results.append(
                        {
                            "title": result.title_or_content,
                            "url": result.url or "",
                            "content": result.title_or_content,
                            "score": result.hotness_score,
                            "raw_content": result.title_or_content,
                            "published_date": result.publish_time.isoformat()
                            if result.publish_time
                            else None,
                            "platform": result.platform,
                            "content_type": result.content_type,
                            "author": result.author_nickname,
                            "engagement": result.engagement,
                        }
                    )

            if search_results:
                _message = f"    找到 {len(search_results)} 个反思搜索结果"
                for j, result in enumerate(search_results, 1):
                    date_info = (
                        f" (发布于: {result.get('published_date', 'N/A')})"
                        if result.get("published_date")
                        else ""
                    )
                    _message += f"\n      {j}. {result['title'][:50]}...{date_info}"
                logger.info(_message)
            else:
                logger.info("    未找到反思搜索结果")

            # 把本轮反思搜索结果并入段落搜索历史
            paragraph.research.add_search_results(search_query, search_results)

            # 反思总结输入额外带上 paragraph_latest_state，让 LLM 在原文基础上「增补/修订」
            reflection_summary_input = {
                "title": paragraph.title,
                "content": paragraph.content,
                "search_query": search_query,
                "search_results": format_search_results_for_prompt(
                    search_results, self.config.MAX_CONTENT_LENGTH
                ),
                "paragraph_latest_state": paragraph.research.latest_summary,
            }

            # mutate_state 会覆盖 latest_summary 并把 reflection_iteration 加一
            self.state = self.reflection_summary_node.mutate_state(
                reflection_summary_input, self.state, paragraph_index
            )

            logger.info(f"    反思 {reflection_i + 1} 完成")

    def _generate_final_report(self) -> str:
        """生成最终报告（Step 3）。

        把每个段落的「标题 + 最新总结」收集起来交给格式化节点，由 LLM 拼成完整 Markdown
        报告；LLM 失败时退化为本地手工拼接（format_report_manually），保证总有产出。
        """
        logger.info(f"\n[步骤 3] 生成最终报告...")

        # 汇总各段落的最终内容（latest_summary 即多轮反思后的成稿）
        report_data = []
        for paragraph in self.state.paragraphs:
            report_data.append(
                {
                    "title": paragraph.title,
                    "paragraph_latest_state": paragraph.research.latest_summary,
                }
            )

        # 优先让 LLM 排版；异常时用本地兜底排版，避免整篇报告失败
        try:
            final_report = self.report_formatting_node.run(report_data)
        except Exception as e:
            logger.exception(f"LLM格式化失败，使用备用方法: {str(e)}")
            final_report = self.report_formatting_node.format_report_manually(
                report_data, self.state.report_title
            )

        # 写回状态并标记整篇完成
        self.state.final_report = final_report
        self.state.mark_completed()

        logger.info("最终报告生成完成")
        return final_report

    def _save_report(self, report_content: str):
        """保存报告到文件（Step 4）。

        文件名形如 deep_search_report_<净化后的query>_<时间戳>.md，落到 OUTPUT_DIR；
        若开启 SAVE_INTERMEDIATE_STATES，则额外把整个 State 序列化为 json 便于复盘。
        """
        # 时间戳 + 净化查询词（去掉非法文件名字符、空格转下划线、截断 30 字）构成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(
            c for c in self.state.query if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        query_safe = query_safe.replace(" ", "_")[:30]

        filename = f"deep_search_report_{query_safe}_{timestamp}.md"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)

        # 写出 Markdown 报告（这份文件后续会被 ReportEngine 读取作为输入）
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"报告已保存到: {filepath}")

        # 可选：把完整状态（段落、搜索历史等）另存为 json，方便调试与复现
        if self.config.SAVE_INTERMEDIATE_STATES:
            state_filename = f"state_{query_safe}_{timestamp}.json"
            state_filepath = os.path.join(self.config.OUTPUT_DIR, state_filename)
            self.state.save_to_file(state_filepath)
            logger.info(f"状态已保存到: {state_filepath}")

    def get_progress_summary(self) -> Dict[str, Any]:
        """获取进度摘要（已完成段落数、百分比等，供前端展示进度）"""
        return self.state.get_progress_summary()

    def load_state(self, filepath: str):
        """从文件加载状态（用断点续跑或复盘历史会话）"""
        self.state = State.load_from_file(filepath)
        logger.info(f"状态已从 {filepath} 加载")

    def save_state(self, filepath: str):
        """保存状态到文件（手动落盘当前 State 快照）"""
        self.state.save_to_file(filepath)
        logger.info(f"状态已保存到 {filepath}")


def create_agent(config_file: Optional[str] = None) -> DeepSearchAgent:
    """
    创建Deep Search Agent实例的便捷函数

    Args:
        config_file: 配置文件路径（当前未使用，保留以兼容旧调用方）

    Returns:
        DeepSearchAgent实例
    """
    # 用空参数构造 Settings：pydantic 会自动从 .env / 环境变量读取真实配置
    config = Settings()
    return DeepSearchAgent(config)
