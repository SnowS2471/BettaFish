"""
Deep Search Agent 主类（QueryEngine 核心编排器）

职责概述
--------
QueryEngine 是「网络/新闻搜索型」Agent：用 Tavily 检索公开网页新闻，配合 LLM 做
深度新闻分析报告。与 InsightEngine 共用同一套节点图骨架（结构→搜索→总结→反思→格式化），
但数据源是公网而非私有库，因此**没有**关键词优化、聚类采样、情感分析等环节。

整体流程（research() 四步，与 InsightEngine 一致）：
    1. 生成报告结构  ReportStructureNode   -> 把 query 拆成 ≤5 个段落
    2. 逐段落处理    _process_paragraphs   -> 首次搜索+总结，再做 N 轮反思补充
    3. 汇总最终报告  ReportFormattingNode  -> 拼成 Markdown
    4. 保存报告      _save_report          -> 写入 OUTPUT_DIR/*.md

单段落数据流：LLM 产出检索词 -> Tavily 搜索 -> 取前 10 条 -> 喂回 LLM 生成/更新段落。
"""

import json
import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List

from .llms import LLMClient
from .nodes import (
    ReportStructureNode,    # 报告结构：把 query 拆成多个段落大纲
    FirstSearchNode,        # 首次搜索：根据段落生成检索词
    ReflectionNode,         # 反思搜索：找信息缺口并生成补充检索词
    FirstSummaryNode,       # 首次总结：把搜索结果写成段落正文
    ReflectionSummaryNode,  # 反思总结：用补充结果更新段落正文
    ReportFormattingNode    # 报告格式化：各段落汇总成最终 Markdown
)
from .state import State
from .tools import TavilyNewsAgency, TavilyResponse  # Tavily 网络新闻搜索工具集
from .utils import Settings, format_search_results_for_prompt
from loguru import logger

class DeepSearchAgent:
    """Deep Search Agent 主类。

    一个实例对应一次研究会话：持有 LLM 客户端、Tavily 搜索工具集，以及贯穿全程、
    被各节点不断读写的 State 对象。
    """

    def __init__(self, config: Optional[Settings] = None):
        """
        初始化Deep Search Agent

        Args:
            config: 配置对象，如果不提供则自动加载
        """
        # 允许传入独立配置（Streamlit 会构造临时 Settings），否则用全局单例
        from .utils.config import settings
        self.config = config or settings

        # 初始化LLM客户端（读取 QUERY_ENGINE_* 配置，默认 DeepSeek）
        self.llm_client = self._initialize_llm()

        # 初始化搜索工具集：Tavily 网络新闻搜索（封装了 6 个搜索工具）
        self.search_agency = TavilyNewsAgency(api_key=self.config.TAVILY_API_KEY)

        # 初始化各处理节点（共享同一个 llm_client）
        self._initialize_nodes()

        # 全局状态对象：段落、搜索历史、总结、最终报告都挂在这里
        self.state = State()

        # 确保报告输出目录存在（如 query_engine_streamlit_reports/）
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        logger.info(f"Query Agent已初始化")
        logger.info(f"使用LLM: {self.llm_client.get_model_info()}")
        logger.info(f"搜索工具集: TavilyNewsAgency (支持6种搜索工具)")

    def _initialize_llm(self) -> LLMClient:
        """初始化LLM客户端（使用 QUERY_ENGINE_* 三件套：key / model / base_url）"""
        return LLMClient(
            api_key=self.config.QUERY_ENGINE_API_KEY,
            model_name=self.config.QUERY_ENGINE_MODEL_NAME,
            base_url=self.config.QUERY_ENGINE_BASE_URL,
        )

    def _initialize_nodes(self):
        """初始化处理节点。

        搜索类（First/Reflection SearchNode）只产出检索词、不改 State；
        状态修改类（First/Reflection SummaryNode、ReportFormattingNode）会写回 State。
        """
        self.first_search_node = FirstSearchNode(self.llm_client)
        self.reflection_node = ReflectionNode(self.llm_client)
        self.first_summary_node = FirstSummaryNode(self.llm_client)
        self.reflection_summary_node = ReflectionSummaryNode(self.llm_client)
        self.report_formatting_node = ReportFormattingNode(self.llm_client)
    
    def _validate_date_format(self, date_str: str) -> bool:
        """
        验证日期格式是否为YYYY-MM-DD

        用于校验 LLM 给出的 start_date/end_date：格式与日期都合法才会传给 search_news_by_date，
        否则降级为 basic_search_news。

        Args:
            date_str: 日期字符串

        Returns:
            是否为有效格式
        """
        if not date_str:
            return False

        # 先用正则卡格式（四位年-两位月-两位日）
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(pattern, date_str):
            return False

        # 再用 strptime 验证日期本身是否存在（如 2025-02-30 会被拦下）
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    def execute_search_tool(self, tool_name: str, query: str, **kwargs) -> TavilyResponse:
        """
        执行指定的搜索工具（按工具名直接分发到 TavilyNewsAgency 的对应方法）

        与 InsightEngine 不同，这里没有关键词优化/聚类/情感分析等中间件，是一层薄分发。
        说明：搜索节点的 process_output 只回传 search_query，故主流程里 tool_name 实际多为
        默认的 "basic_search_news"（其余 5 个工具一般不会被走到，详见 _initial_search_and_summary）。

        Args:
            tool_name: 工具名称，可选值：
                - "basic_search_news": 基础新闻搜索（快速、通用）
                - "deep_search_news": 深度新闻分析
                - "search_news_last_24_hours": 24小时内最新新闻
                - "search_news_last_week": 本周新闻
                - "search_images_for_news": 新闻图片搜索
                - "search_news_by_date": 按日期范围搜索新闻
            query: 搜索查询
            **kwargs: 额外参数（如start_date, end_date, max_results）

        Returns:
            TavilyResponse对象
        """
        logger.info(f"  → 执行搜索工具: {tool_name}")

        if tool_name == "basic_search_news":
            max_results = kwargs.get("max_results", 7)
            return self.search_agency.basic_search_news(query, max_results)
        elif tool_name == "deep_search_news":
            return self.search_agency.deep_search_news(query)
        elif tool_name == "search_news_last_24_hours":
            return self.search_agency.search_news_last_24_hours(query)
        elif tool_name == "search_news_last_week":
            return self.search_agency.search_news_last_week(query)
        elif tool_name == "search_images_for_news":
            return self.search_agency.search_images_for_news(query)
        elif tool_name == "search_news_by_date":
            # 唯一需要时间参数的工具；缺参数时上游已降级，这里再兜底校验
            start_date = kwargs.get("start_date")
            end_date = kwargs.get("end_date")
            if not start_date or not end_date:
                raise ValueError("search_news_by_date工具需要start_date和end_date参数")
            return self.search_agency.search_news_by_date(query, start_date, end_date)
        else:
            # 工具名未识别时兜底为基础搜索，保证流程不中断
            logger.warning(f"  ⚠️  未知的搜索工具: {tool_name}，使用默认基础搜索")
            return self.search_agency.basic_search_news(query)
    
    def research(self, query: str, save_report: bool = True) -> str:
        """
        执行深度研究（一体化入口）

        把四个步骤串起来跑完。CLI/编排器用这个；Streamlit 为显示进度条则不调它，而是手动
        依次调用下面的 _generate_report_structure / _process_paragraphs 等私有方法。

        Args:
            query: 研究查询
            save_report: 是否保存报告到文件

        Returns:
            最终报告内容
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"开始深度研究: {query}")
        logger.info(f"{'='*60}")

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

            logger.info(f"\n{'='*60}")
            logger.info("深度研究完成！")
            logger.info(f"{'='*60}")

            return final_report

        except Exception as e:
            # 打印完整堆栈后向上抛，交给调用方（Streamlit/编排器）统一展示
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"研究过程中发生错误: {str(e)} \n错误堆栈: {error_traceback}")
            raise e

    def _generate_report_structure(self, query: str):
        """生成报告结构（Step 1）：交给 ReportStructureNode 把 query 规划成 ≤5 个段落并写入 state。"""
        logger.info(f"\n[步骤 1] 生成报告结构...")

        # 节点构造时需要 query 才能规划
        report_structure_node = ReportStructureNode(self.llm_client, query)

        # mutate_state 就地生成段落并返回更新后的 state
        self.state = report_structure_node.mutate_state(state=self.state)

        _message = f"报告结构已生成，共 {len(self.state.paragraphs)} 个段落:"
        for i, paragraph in enumerate(self.state.paragraphs, 1):
            _message += f"\n  {i}. {paragraph.title}"
        logger.info(_message)

    def _process_paragraphs(self):
        """处理所有段落（Step 2）：逐段落「首次搜索+总结」再做 MAX_REFLECTIONS 轮反思，串行执行。"""
        total_paragraphs = len(self.state.paragraphs)

        for i in range(total_paragraphs):
            logger.info(f"\n[步骤 2.{i+1}] 处理段落: {self.state.paragraphs[i].title}")
            logger.info("-" * 50)

            # 2a. 首次搜索 + 首次总结，得到段落初稿
            self._initial_search_and_summary(i)

            # 2b. 多轮反思：找缺口 -> 补充搜索 -> 更新段落
            self._reflection_loop(i)

            # 2c. 标记该段落研究完成
            self.state.paragraphs[i].research.mark_completed()

            progress = (i + 1) / total_paragraphs * 100
            logger.info(f"段落处理完成 ({progress:.1f}%)")
    
    def _initial_search_and_summary(self, paragraph_index: int):
        """执行初始搜索和总结（Step 2a）。

        用段落标题/内容让 LLM 生成检索词 -> Tavily 搜索 -> 取前 10 条转成统一格式 ->
        写入搜索历史 -> 让总结节点写出段落初稿。
        """
        paragraph = self.state.paragraphs[paragraph_index]

        # 喂给搜索节点的输入：段落标题 + 预期内容
        search_input = {
            "title": paragraph.title,
            "content": paragraph.content
        }

        # 让 LLM 生成检索词（及理由）
        logger.info("  - 生成搜索查询...")
        search_output = self.first_search_node.run(search_input)
        search_query = search_output["search_query"]
        # 注：FirstSearchNode.process_output 通常只回传 search_query/reasoning，故 search_tool
        # 多数会取默认值 basic_search_news（详见 search_node.process_output）
        search_tool = search_output.get("search_tool", "basic_search_news")  # 默认工具
        reasoning = search_output["reasoning"]
        
        logger.info(f"  - 搜索查询: {search_query}")
        logger.info(f"  - 选择的工具: {search_tool}")
        logger.info(f"  - 推理: {reasoning}")
        
        # 执行搜索
        logger.info("  - 执行网络搜索...")
        
        # 处理search_news_by_date的特殊参数
        search_kwargs = {}
        if search_tool == "search_news_by_date":
            start_date = search_output.get("start_date")
            end_date = search_output.get("end_date")
            
            if start_date and end_date:
                # 验证日期格式
                if self._validate_date_format(start_date) and self._validate_date_format(end_date):
                    search_kwargs["start_date"] = start_date
                    search_kwargs["end_date"] = end_date
                    logger.info(f"  - 时间范围: {start_date} 到 {end_date}")
                else:
                    logger.info(f"  ⚠️  日期格式错误（应为YYYY-MM-DD），改用基础搜索")
                    logger.info(f"      提供的日期: start_date={start_date}, end_date={end_date}")
                    search_tool = "basic_search_news"
            else:
                logger.info(f"  ⚠️  search_news_by_date工具缺少时间参数，改用基础搜索")
                search_tool = "basic_search_news"
        
        search_response = self.execute_search_tool(search_tool, search_query, **search_kwargs)
        
        # 转换为兼容格式
        search_results = []
        if search_response and search_response.results:
            # 每种搜索工具都有其特定的结果数量，这里取前10个作为上限
            max_results = min(len(search_response.results), 10)
            for result in search_response.results[:max_results]:
                search_results.append({
                    'title': result.title,
                    'url': result.url,
                    'content': result.content,
                    'score': result.score,
                    'raw_content': result.raw_content,
                    'published_date': result.published_date  # 新增字段
                })
        
        if search_results:
            _message = f"  - 找到 {len(search_results)} 个搜索结果"
            for j, result in enumerate(search_results, 1):
                date_info = f" (发布于: {result.get('published_date', 'N/A')})" if result.get('published_date') else ""
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
            # 按 SEARCH_CONTENT_MAX_LENGTH 截断每条内容，避免上下文过长
            "search_results": format_search_results_for_prompt(
                search_results, self.config.SEARCH_CONTENT_MAX_LENGTH
            )
        }

        # mutate_state 内部会调 LLM 生成总结并写入 paragraph.research.latest_summary
        self.state = self.first_summary_node.mutate_state(
            summary_input, self.state, paragraph_index
        )

        logger.info("  - 初始总结完成")
    
    def _reflection_loop(self, paragraph_index: int):
        """执行反思循环（Step 2b）。

        固定循环 MAX_REFLECTIONS 次（默认 2，不按质量动态停）。每轮把段落当前内容连同标题
        交给反思节点找缺口、生成补充检索词，再 Tavily 搜索并据此更新段落。
        """
        paragraph = self.state.paragraphs[paragraph_index]

        for reflection_i in range(self.config.MAX_REFLECTIONS):
            logger.info(f"  - 反思 {reflection_i + 1}/{self.config.MAX_REFLECTIONS}...")

            # 反思输入比首次多一个 paragraph_latest_state，让 LLM 知道「已经写了什么」
            reflection_input = {
                "title": paragraph.title,
                "content": paragraph.content,
                "paragraph_latest_state": paragraph.research.latest_summary
            }

            # 让 LLM 基于当前内容找缺口并生成补充检索词
            reflection_output = self.reflection_node.run(reflection_input)
            search_query = reflection_output["search_query"]
            # 同 _initial：search_tool 多数会回退默认值（process_output 只回传 query/reasoning）
            search_tool = reflection_output.get("search_tool", "basic_search_news")  # 默认工具
            reasoning = reflection_output["reasoning"]
            
            logger.info(f"    反思查询: {search_query}")
            logger.info(f"    选择的工具: {search_tool}")
            logger.info(f"    反思推理: {reasoning}")
            
            # 执行反思搜索
            # 处理search_news_by_date的特殊参数
            search_kwargs = {}
            if search_tool == "search_news_by_date":
                start_date = reflection_output.get("start_date")
                end_date = reflection_output.get("end_date")
                
                if start_date and end_date:
                    # 验证日期格式
                    if self._validate_date_format(start_date) and self._validate_date_format(end_date):
                        search_kwargs["start_date"] = start_date
                        search_kwargs["end_date"] = end_date
                        logger.info(f"    时间范围: {start_date} 到 {end_date}")
                    else:
                        logger.info(f"    ⚠️  日期格式错误（应为YYYY-MM-DD），改用基础搜索")
                        logger.info(f"        提供的日期: start_date={start_date}, end_date={end_date}")
                        search_tool = "basic_search_news"
                else:
                    logger.info(f"    ⚠️  search_news_by_date工具缺少时间参数，改用基础搜索")
                    search_tool = "basic_search_news"
            
            search_response = self.execute_search_tool(search_tool, search_query, **search_kwargs)
            
            # 转换为兼容格式
            search_results = []
            if search_response and search_response.results:
                # 每种搜索工具都有其特定的结果数量，这里取前10个作为上限
                max_results = min(len(search_response.results), 10)
                for result in search_response.results[:max_results]:
                    search_results.append({
                        'title': result.title,
                        'url': result.url,
                        'content': result.content,
                        'score': result.score,
                        'raw_content': result.raw_content,
                        'published_date': result.published_date
                    })
            
            if search_results:
                logger.info(f"    找到 {len(search_results)} 个反思搜索结果")
                for j, result in enumerate(search_results, 1):
                    date_info = f" (发布于: {result.get('published_date', 'N/A')})" if result.get('published_date') else ""
                    logger.info(f"      {j}. {result['title'][:50]}...{date_info}")
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
                    search_results, self.config.SEARCH_CONTENT_MAX_LENGTH
                ),
                "paragraph_latest_state": paragraph.research.latest_summary
            }

            # mutate_state 会覆盖 latest_summary 并把 reflection_iteration 加一
            self.state = self.reflection_summary_node.mutate_state(
                reflection_summary_input, self.state, paragraph_index
            )

            logger.info(f"    反思 {reflection_i + 1} 完成")

    def _generate_final_report(self) -> str:
        """生成最终报告（Step 3）。

        收集每段「标题 + 最新总结」交给格式化节点由 LLM 拼成 Markdown；LLM 失败时退化为
        本地手工拼接（format_report_manually），保证总有产出。
        """
        logger.info(f"\n[步骤 3] 生成最终报告...")

        # 汇总各段落最终内容（latest_summary 即多轮反思后的成稿）
        report_data = []
        for paragraph in self.state.paragraphs:
            report_data.append({
                "title": paragraph.title,
                "paragraph_latest_state": paragraph.research.latest_summary
            })

        # 优先让 LLM 排版；异常时用本地兜底排版，避免整篇报告失败
        try:
            final_report = self.report_formatting_node.run(report_data)
        except Exception as e:
            logger.error(f"LLM格式化失败，使用备用方法: {str(e)}")
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
        # 时间戳 + 净化查询词（去非法字符、空格转下划线、截断 30 字）构成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(c for c in self.state.query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        query_safe = query_safe.replace(' ', '_')[:30]

        filename = f"deep_search_report_{query_safe}_{timestamp}.md"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)

        # 写出 Markdown 报告（这份文件后续会被 ReportEngine 读取作为输入）
        with open(filepath, 'w', encoding='utf-8') as f:
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
        """从文件加载状态（断点续跑/复盘历史会话）"""
        self.state = State.load_from_file(filepath)
        logger.info(f"状态已从 {filepath} 加载")

    def save_state(self, filepath: str):
        """保存状态到文件（手动落盘当前 State 快照）"""
        self.state.save_to_file(filepath)
        logger.info(f"状态已保存到 {filepath}")


def create_agent() -> DeepSearchAgent:
    """
    创建Deep Search Agent实例的便捷函数

    用空参数构造 Settings：pydantic 会自动从 .env / 环境变量读取真实配置。

    Returns:
        DeepSearchAgent实例
    """
    from .utils.config import Settings
    config = Settings()
    return DeepSearchAgent(config)
