"""
根据模板目录与多源报告，生成整本报告的标题/目录/主题设计。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from loguru import logger

from ..core import TemplateSection
from ..prompts import (
    SYSTEM_PROMPT_DOCUMENT_LAYOUT,
    build_document_layout_prompt,
)
from ..utils.json_parser import RobustJSONParser, JSONParseError
from .base_node import BaseNode


class DocumentLayoutNode(BaseNode):
    """
    负责生成全局标题、目录与Hero设计。

    结合模板切片、报告摘要与论坛讨论，指导整本书的视觉与结构基调。
    """

    def __init__(self, llm_client):
        """记录LLM客户端并设置节点名字，供BaseNode日志使用"""
        super().__init__(llm_client, "DocumentLayoutNode")
        # 初始化鲁棒JSON解析器，启用所有修复策略
        self.json_parser = RobustJSONParser(
            enable_json_repair=True,
            enable_llm_repair=False,  # 可以根据需要启用LLM修复
            max_repair_attempts=3,
        )

    def run(
        self,
        sections: List[TemplateSection],
        template_markdown: str,
        reports: Dict[str, str],
        forum_logs: str,
        query: str,
        template_overview: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        综合模板+多源内容，生成全书的标题、目录结构与主题色板。

        参数:
            sections: 模板切片后的章节列表。
            template_markdown: 模板原文，用于LLM理解上下文。
            reports: 三个引擎的内容映射。
            forum_logs: 论坛讨论摘要。
            query: 用户查询词。
            template_overview: 预生成的模板概览，可复用以减少提示词长度。

        返回:
            dict: 包含 title/subtitle/toc/hero/themeTokens 等设计信息的字典。
        """
        # 将模板原文、切片结构与多源报告一并喂给LLM，便于其理解层级与素材
        payload = {
            "query": query,
            "template": {
                "raw": template_markdown,
                "sections": [section.to_dict() for section in sections],
            },
            "templateOverview": template_overview
            or {
                "title": sections[0].title if sections else "",
                "chapters": [section.to_dict() for section in sections],
            },
            "reports": reports,
            "forumLogs": forum_logs,
        }

        # 南非专题模板：注入视觉与结构提示
        template_name = (template_overview or {}).get("template_name", "")
        is_sa_template = "南非" in template_name
        if is_sa_template:
            payload["_saTemplateHints"] = self._sa_template_hints(sections)

        user_message = build_document_layout_prompt(payload)
        response = self.llm_client.stream_invoke_to_string(
            SYSTEM_PROMPT_DOCUMENT_LAYOUT,
            user_message,
            temperature=0.3,
            top_p=0.9,
        )
        design = self._parse_response(response)

        # 南非专题模板：注入默认视觉元素
        if is_sa_template:
            design = self._apply_sa_defaults(design)

        logger.info("文档标题/目录设计已生成")
        return design

    @staticmethod
    def _sa_template_hints(sections: List[TemplateSection]) -> Dict[str, Any]:
        """
        为南非专题模板生成结构化的视觉与布局提示。

        Args:
            sections: 模板章节列表

        Returns:
            包含 themeTokens、hero 建议、layoutNotes 等提示的字典
        """
        return {
            "themeTokens": {
                "primary": "#007A4D",
                "secondary": "#FFB81C",
                "accent": "#DE3831",
                "dark": "#001489",
                "neutral": "#333333",
                "bgLight": "#FAFAFA",
                "description": "南非国旗配色主题：绿金红蓝黑",
            },
            "hero": {
                "summary": "本报告整合南非主流新闻媒体与X/Twitter平台数据，针对专题进行跨语言、跨平台舆情分析",
                "highlights": [
                    "多源数据整合：覆盖南非5大新闻网站与X平台社交媒体数据",
                    "跨语言分析：系统后台利用翻译能力增强多语言数据检索，提升分析覆盖率",
                    "跨平台对比：新闻媒体与社交媒体主题/情感/趋势多维度对比",
                ],
            },
            "layoutNotes": (
                "建议封面加入'南非专题'标识，使用绿金配色主题；"
                "第4章舆情焦点章节使用KPI卡片和表格展示核心议题分布；"
                "第6-7章跨平台对比使用左右对比布局强化视觉对比效果"
            ),
        }

    @staticmethod
    def _apply_sa_defaults(design: Dict[str, Any]) -> Dict[str, Any]:
        """
        当 LLM 未返回南非相关视觉元素时，注入默认值。

        Args:
            design: LLM 返回的设计结果

        Returns:
            补全后的设计结果
        """
        if not design:
            return design

        # 补全 hero
        hero = design.setdefault("hero", {})
        if not hero.get("summary"):
            hero["summary"] = (
                "本报告整合南非主流新闻媒体与X/Twitter平台数据，"
                "针对专题进行跨语言、跨平台舆情分析"
            )
        if not hero.get("highlights"):
            hero["highlights"] = [
                "多源数据整合：覆盖南非5大新闻网站与X平台社交媒体数据",
                "跨语言分析：系统后台利用翻译能力增强多语言数据检索",
                "跨平台对比：新闻媒体与社交媒体主题/情感/趋势多维度对比",
            ]

        # 补全主题色
        theme = design.setdefault("themeTokens", {})
        if not theme:
            theme.update({
                "primary": "#007A4D",
                "secondary": "#FFB81C",
                "accent": "#DE3831",
                "dark": "#001489",
                "neutral": "#333333",
                "bgLight": "#FAFAFA",
            })

        return design

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """
        解析LLM返回的JSON文本，若失败则抛出友好错误。

        使用鲁棒JSON解析器进行多重修复尝试：
        1. 清理markdown标记和思考内容
        2. 本地语法修复（括号平衡、逗号补全、控制字符转义等）
        3. 使用json_repair库进行高级修复
        4. 可选的LLM辅助修复

        参数:
            raw: LLM原始返回字符串，允许带```包裹、思考内容等。

        返回:
            dict: 结构化的设计稿。

        异常:
            ValueError: 当响应为空或JSON解析失败时抛出。
        """
        try:
            result = self.json_parser.parse(
                raw,
                context_name="文档设计",
                # 目录字段已更名为 tocPlan，这里跟随最新Schema校验
                expected_keys=["title", "tocPlan", "hero"],
            )
            # 验证关键字段的类型
            if not isinstance(result.get("title"), str):
                logger.warning("文档设计缺少title字段或类型错误，使用默认值")
                result.setdefault("title", "未命名报告")

            # 处理tocPlan字段
            toc_plan = result.get("tocPlan", [])
            if not isinstance(toc_plan, list):
                logger.warning("文档设计缺少tocPlan字段或类型错误，使用空列表")
                result["tocPlan"] = []
            else:
                # 清理tocPlan中的description字段
                result["tocPlan"] = self._clean_toc_plan_descriptions(toc_plan)

            if not isinstance(result.get("hero"), dict):
                logger.warning("文档设计缺少hero字段或类型错误，使用空对象")
                result.setdefault("hero", {})

            return result
        except JSONParseError as exc:
            # 转换为原有的异常类型以保持向后兼容
            raise ValueError(f"文档设计JSON解析失败: {exc}") from exc

    def _clean_toc_plan_descriptions(self, toc_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        清理tocPlan中每个条目的description字段，移除可能的JSON片段。

        参数:
            toc_plan: 原始的目录计划列表

        返回:
            List[Dict[str, Any]]: 清理后的目录计划列表
        """
        import re

        def clean_text(text: Any) -> str:
            """清理文本中的JSON片段"""
            if not text or not isinstance(text, str):
                return ""

            cleaned = text

            # 移除以逗号+空白+{开头的不完整JSON对象
            cleaned = re.sub(r',\s*\{[^}]*$', '', cleaned)

            # 移除以逗号+空白+[开头的不完整JSON数组
            cleaned = re.sub(r',\s*\[[^\]]*$', '', cleaned)

            # 移除孤立的 { 加上后续内容（如果没有匹配的 }）
            open_brace_pos = cleaned.rfind('{')
            if open_brace_pos != -1:
                close_brace_pos = cleaned.rfind('}')
                if close_brace_pos < open_brace_pos:
                    cleaned = cleaned[:open_brace_pos].rstrip(',，、 \t\n')

            # 移除孤立的 [ 加上后续内容（如果没有匹配的 ]）
            open_bracket_pos = cleaned.rfind('[')
            if open_bracket_pos != -1:
                close_bracket_pos = cleaned.rfind(']')
                if close_bracket_pos < open_bracket_pos:
                    cleaned = cleaned[:open_bracket_pos].rstrip(',，、 \t\n')

            # 移除看起来像JSON键值对的片段
            cleaned = re.sub(r',?\s*"[^"]+"\s*:\s*"[^"]*$', '', cleaned)
            cleaned = re.sub(r',?\s*"[^"]+"\s*:\s*[^,}\]]*$', '', cleaned)

            # 清理末尾的逗号和空白
            cleaned = cleaned.rstrip(',，、 \t\n')

            return cleaned.strip()

        cleaned_plan = []
        for entry in toc_plan:
            if not isinstance(entry, dict):
                continue

            # 清理description字段
            if "description" in entry:
                original_desc = entry["description"]
                cleaned_desc = clean_text(original_desc)

                if cleaned_desc != original_desc:
                    logger.warning(
                        f"清理目录项 '{entry.get('display', 'unknown')}' 的description字段中的JSON片段:\n"
                        f"  原文: {original_desc[:100]}...\n"
                        f"  清理后: {cleaned_desc[:100]}..."
                    )
                    entry["description"] = cleaned_desc

            cleaned_plan.append(entry)

        return cleaned_plan


__all__ = ["DocumentLayoutNode"]
