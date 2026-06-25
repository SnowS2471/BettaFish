"""
总结节点实现

把搜索结果转写成段落正文：
- FirstSummaryNode：生成段落初稿。
- ReflectionSummaryNode：在已有内容上增补/修订。
两者生成前都会①附加平台元信息（区分新闻端/社交端）、②读取 ForumEngine 写在 forum.log 的
[HOST] 主持人发言并前置进 prompt——这是各 Agent 通过论坛协同、避免同质化的关键注入点。
"""

import json
from typing import Dict, Any, List
from json.decoder import JSONDecodeError
from loguru import logger

from .base_node import StateMutationNode
from ..state.state import State
from ..prompts import SYSTEM_PROMPT_FIRST_SUMMARY, SYSTEM_PROMPT_REFLECTION_SUMMARY
from ..utils.text_processing import (
    remove_reasoning_from_output,
    clean_json_tags,
    extract_clean_response,
    fix_incomplete_json,
    format_search_results_for_prompt,
    partition_by_platform,
    merge_bilingual_results,
    detect_content_language,
)

# 导入论坛读取工具：用于读取 ForumEngine 写在 logs/forum.log 的 [HOST] 主持人发言。
# 该模块在项目根的 utils/ 下，故把根目录加入 sys.path；导入失败则降级（跳过 HOST 读取）。
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from utils.forum_reader import get_latest_host_speech, format_host_speech_for_prompt
    FORUM_READER_AVAILABLE = True
except ImportError:
    FORUM_READER_AVAILABLE = False
    logger.warning("无法导入forum_reader模块，将跳过HOST发言读取功能")


class FirstSummaryNode(StateMutationNode):
    """根据搜索结果生成段落首次总结的节点。

    生成前会：①附加平台元信息(_enrich_input_with_platform_meta)帮 LLM 区分新闻端/社交端；
    ②读取最新 [HOST] 主持人发言并前置进 prompt。结果写回 state.paragraphs[i].research.latest_summary。
    """
    
    def __init__(self, llm_client):
        """
        初始化首次总结节点
        
        Args:
            llm_client: LLM客户端
        """
        super().__init__(llm_client, "FirstSummaryNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                required_fields = ["title", "content", "search_query", "search_results"]
                return all(field in data for field in required_fields)
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            required_fields = ["title", "content", "search_query", "search_results"]
            return all(field in input_data for field in required_fields)
        return False
    
    def _enrich_input_with_platform_meta(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        为输入数据附加平台元信息，使总结节点能区分新闻端与社交端。

        对 search_results 进行：
        - 按 platform 分组统计
        - 对 sa_news 结果标记翻译状态
        - 附加平台分组摘要到 data['_platform_meta']

        Args:
            data: 原始输入数据

        Returns:
            附加了 _platform_meta 的数据副本
        """
        results = data.get('search_results', [])
        if not results:
            return data

        enriched = dict(data)
        grouped = partition_by_platform(results)
        platform_summary = {}
        for platform_name, items in grouped.items():
            has_translation_count = sum(1 for r in items if r.get('has_translation'))
            platform_summary[platform_name] = {
                'count': len(items),
                'has_translation_count': has_translation_count,
                'sample_titles': [r.get('title', '')[:60] for r in items[:3]],
                'languages': set(
                    detect_content_language(r.get('content', '')) for r in items[:10]
                ),
            }

        enriched['_platform_meta'] = {
            'platform_groups': list(grouped.keys()),
            'platform_summary': {k: {'count': v['count'], 'has_translation_count': v['has_translation_count']} for k, v in platform_summary.items()},
            'news_platforms': [p for p in grouped if p in ('南非新闻', 'sa_news', '新闻媒体')],
            'social_platforms': [p for p in grouped if p in ('X平台', '微博', '抖音', 'B站', '小红书', '知乎', '快手', '贴吧')],
        }
        return enriched

    def run(self, input_data: Any, **kwargs) -> str:
        """
        调用LLM生成段落总结

        Args:
            input_data: 包含title、content、search_query和search_results的数据
            **kwargs: 额外参数

        Returns:
            段落总结内容
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("输入数据格式错误")

            # 准备输入数据
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data.copy() if isinstance(input_data, dict) else input_data

            # 为 sa_news 结果标记翻译状态
            if 'search_results' in data:
                data['search_results'] = merge_bilingual_results(data['search_results'])

            # 附加平台元信息，帮助 LLM 区分新闻端与社交端
            data = self._enrich_input_with_platform_meta(data)

            # 读取最新的HOST发言（如果可用）
            if FORUM_READER_AVAILABLE:
                try:
                    host_speech = get_latest_host_speech()
                    if host_speech:
                        # 将HOST发言添加到输入数据中
                        data['host_speech'] = host_speech
                        logger.info(f"已读取HOST发言，长度: {len(host_speech)}字符")
                except Exception as e:
                    logger.exception(f"读取HOST发言失败: {str(e)}")

            # 转换为JSON字符串
            message = json.dumps(data, ensure_ascii=False)

            # 如果有HOST发言，添加到消息前面作为参考
            if FORUM_READER_AVAILABLE and 'host_speech' in data and data['host_speech']:
                formatted_host = format_host_speech_for_prompt(data['host_speech'])
                message = formatted_host + "\n" + message

            logger.info("正在生成首次段落总结")

            # 调用LLM（流式，安全拼接UTF-8）
            response = self.llm_client.stream_invoke_to_string(SYSTEM_PROMPT_FIRST_SUMMARY, message)

            # 处理响应
            processed_response = self.process_output(response)

            logger.info("成功生成首次段落总结")
            return processed_response

        except Exception as e:
            logger.exception(f"生成首次总结失败: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> str:
        """
        处理LLM输出，提取段落内容
        
        Args:
            output: LLM原始输出
            
        Returns:
            段落内容
        """
        try:
            # 清理响应文本
            cleaned_output = remove_reasoning_from_output(output)
            cleaned_output = clean_json_tags(cleaned_output)
            
            # 记录清理后的输出用于调试
            logger.info(f"清理后的输出: {cleaned_output}")
            
            # 解析JSON
            try:
                result = json.loads(cleaned_output)
                logger.info("JSON解析成功")
            except JSONDecodeError as e:
                logger.error(f"JSON解析失败: {str(e)}")
                # 尝试修复JSON
                fixed_json = fix_incomplete_json(cleaned_output)
                if fixed_json:
                    try:
                        result = json.loads(fixed_json)
                        logger.info("JSON修复成功")
                    except JSONDecodeError:
                        logger.exception("JSON修复失败，直接使用清理后的文本")
                        # 如果不是JSON格式，直接返回清理后的文本
                        return cleaned_output
                else:
                    logger.exception("无法修复JSON，直接使用清理后的文本")
                    # 如果不是JSON格式，直接返回清理后的文本
                    return cleaned_output
            
            # 提取段落内容
            if isinstance(result, dict):
                paragraph_content = result.get("paragraph_latest_state", "")
                if paragraph_content:
                    return paragraph_content
            
            # 如果提取失败，返回原始清理后的文本
            return cleaned_output
            
        except Exception as e:
            logger.exception(f"处理输出失败: {str(e)}")
            return "段落总结生成失败"
    
    def mutate_state(self, input_data: Any, state: State, paragraph_index: int, **kwargs) -> State:
        """
        更新段落的最新总结到状态
        
        Args:
            input_data: 输入数据
            state: 当前状态
            paragraph_index: 段落索引
            **kwargs: 额外参数
            
        Returns:
            更新后的状态
        """
        try:
            # 生成总结
            summary = self.run(input_data, **kwargs)
            
            # 更新状态
            if 0 <= paragraph_index < len(state.paragraphs):
                state.paragraphs[paragraph_index].research.latest_summary = summary
                logger.info(f"已更新段落 {paragraph_index} 的首次总结")
            else:
                raise ValueError(f"段落索引 {paragraph_index} 超出范围")
            
            state.update_timestamp()
            return state
            
        except Exception as e:
            logger.exception(f"状态更新失败: {str(e)}")
            raise e


class ReflectionSummaryNode(StateMutationNode):
    """根据反思搜索结果「更新」段落总结的节点。

    同样会附加平台元信息、读取 [HOST] 发言；在原段落基础上增补/修订，写回 latest_summary
    并 increment_reflection（反思计数 +1）。
    """
    
    def __init__(self, llm_client):
        """
        初始化反思总结节点
        
        Args:
            llm_client: LLM客户端
        """
        super().__init__(llm_client, "ReflectionSummaryNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                required_fields = ["title", "content", "search_query", "search_results", "paragraph_latest_state"]
                return all(field in data for field in required_fields)
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            required_fields = ["title", "content", "search_query", "search_results", "paragraph_latest_state"]
            return all(field in input_data for field in required_fields)
        return False
    
    def _enrich_input_with_platform_meta(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        为反思输入数据附加平台元信息。

        Args:
            data: 原始输入数据

        Returns:
            附加了 _platform_meta 的数据副本
        """
        results = data.get('search_results', [])
        if not results:
            return data

        enriched = dict(data)
        grouped = partition_by_platform(results)
        platform_summary = {}
        for platform_name, items in grouped.items():
            platform_summary[platform_name] = {
                'count': len(items),
                'has_translation_count': sum(1 for r in items if r.get('has_translation')),
            }

        enriched['_platform_meta'] = {
            'platform_groups': list(grouped.keys()),
            'platform_summary': platform_summary,
            'news_platforms': [p for p in grouped if p in ('南非新闻', 'sa_news', '新闻媒体')],
            'social_platforms': [p for p in grouped if p in ('X平台', '微博', '抖音', 'B站', '小红书', '知乎', '快手', '贴吧')],
        }
        return enriched

    def run(self, input_data: Any, **kwargs) -> str:
        """
        调用LLM更新段落内容

        Args:
            input_data: 包含完整反思信息的数据
            **kwargs: 额外参数

        Returns:
            更新后的段落内容
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("输入数据格式错误")

            # 准备输入数据
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data.copy() if isinstance(input_data, dict) else input_data

            # 为 sa_news 结果标记翻译状态
            if 'search_results' in data:
                data['search_results'] = merge_bilingual_results(data['search_results'])

            # 附加平台元信息
            data = self._enrich_input_with_platform_meta(data)

            # 读取最新的HOST发言（如果可用）
            if FORUM_READER_AVAILABLE:
                try:
                    host_speech = get_latest_host_speech()
                    if host_speech:
                        # 将HOST发言添加到输入数据中
                        data['host_speech'] = host_speech
                        logger.info(f"已读取HOST发言，长度: {len(host_speech)}字符")
                except Exception as e:
                    logger.exception(f"读取HOST发言失败: {str(e)}")

            # 转换为JSON字符串
            message = json.dumps(data, ensure_ascii=False)

            # 如果有HOST发言，添加到消息前面作为参考
            if FORUM_READER_AVAILABLE and 'host_speech' in data and data['host_speech']:
                formatted_host = format_host_speech_for_prompt(data['host_speech'])
                message = formatted_host + "\n" + message

            logger.info("正在生成反思总结")

            # 调用LLM（流式，安全拼接UTF-8）
            response = self.llm_client.stream_invoke_to_string(SYSTEM_PROMPT_REFLECTION_SUMMARY, message)

            # 处理响应
            processed_response = self.process_output(response)

            logger.info("成功生成反思总结")
            return processed_response

        except Exception as e:
            logger.exception(f"生成反思总结失败: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> str:
        """
        处理LLM输出，提取更新后的段落内容
        
        Args:
            output: LLM原始输出
            
        Returns:
            更新后的段落内容
        """
        try:
            # 清理响应文本
            cleaned_output = remove_reasoning_from_output(output)
            cleaned_output = clean_json_tags(cleaned_output)
            
            # 记录清理后的输出用于调试
            logger.info(f"清理后的输出: {cleaned_output}")
            
            # 解析JSON
            try:
                result = json.loads(cleaned_output)
                logger.info("JSON解析成功")
            except JSONDecodeError as e:
                logger.error(f"JSON解析失败: {str(e)}")
                # 尝试修复JSON
                fixed_json = fix_incomplete_json(cleaned_output)
                if fixed_json:
                    try:
                        result = json.loads(fixed_json)
                        logger.info("JSON修复成功")
                    except JSONDecodeError:
                        logger.error("JSON修复失败，直接使用清理后的文本")
                        # 如果不是JSON格式，直接返回清理后的文本
                        return cleaned_output
                else:
                    logger.error("无法修复JSON，直接使用清理后的文本")
                    # 如果不是JSON格式，直接返回清理后的文本
                    return cleaned_output
            
            # 提取更新后的段落内容
            if isinstance(result, dict):
                updated_content = result.get("updated_paragraph_latest_state", "")
                if updated_content:
                    return updated_content
            
            # 如果提取失败，返回原始清理后的文本
            return cleaned_output
            
        except Exception as e:
            logger.exception(f"处理输出失败: {str(e)}")
            return "反思总结生成失败"
    
    def mutate_state(self, input_data: Any, state: State, paragraph_index: int, **kwargs) -> State:
        """
        将更新后的总结写入状态
        
        Args:
            input_data: 输入数据
            state: 当前状态
            paragraph_index: 段落索引
            **kwargs: 额外参数
            
        Returns:
            更新后的状态
        """
        try:
            # 生成更新后的总结
            updated_summary = self.run(input_data, **kwargs)
            
            # 更新状态
            if 0 <= paragraph_index < len(state.paragraphs):
                state.paragraphs[paragraph_index].research.latest_summary = updated_summary
                state.paragraphs[paragraph_index].research.increment_reflection()
                logger.info(f"已更新段落 {paragraph_index} 的反思总结")
            else:
                raise ValueError(f"段落索引 {paragraph_index} 超出范围")
            
            state.update_timestamp()
            return state
            
        except Exception as e:
            logger.exception(f"状态更新失败: {str(e)}")
            raise e
