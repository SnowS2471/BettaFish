"""
搜索节点实现

包含两个「只产出检索词、不修改 State」的节点：
- FirstSearchNode：每个段落首次检索时，依据标题/内容生成检索词。
- ReflectionNode：反思阶段，基于段落当前内容找信息缺口、生成补充检索词。
两者都对 LLM 输出做多级 JSON 容错后解析；另含跨平台检索辅助方法（detect_topic_region 等）。

重要：两个节点的 process_output 最终只回传 {search_query, reasoning}，会丢弃 LLM 可能给出的
search_tool/platform/start_date 等字段，导致 agent 侧实际只使用默认的 search_topic_globally。
"""

import json
from typing import Dict, Any
from json.decoder import JSONDecodeError
from loguru import logger

from .base_node import BaseNode
from ..prompts import SYSTEM_PROMPT_FIRST_SEARCH, SYSTEM_PROMPT_REFLECTION
from ..utils.text_processing import (
    remove_reasoning_from_output,
    clean_json_tags,
    extract_clean_response,
    fix_incomplete_json
)


class FirstSearchNode(BaseNode):
    """为段落生成首次检索词的节点：输入段落标题/内容，让 LLM 产出 search_query(+reasoning)。"""
    
    def __init__(self, llm_client):
        """
        初始化首次搜索节点
        
        Args:
            llm_client: LLM客户端
        """
        super().__init__(llm_client, "FirstSearchNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                return "title" in data and "content" in data
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            return "title" in input_data and "content" in input_data
        return False
    
    def run(self, input_data: Any, **kwargs) -> Dict[str, str]:
        """
        调用LLM生成搜索查询和理由
        
        Args:
            input_data: 包含title和content的字符串或字典
            **kwargs: 额外参数
            
        Returns:
            包含search_query和reasoning的字典
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("输入数据格式错误，需要包含title和content字段")
            
            # 准备输入数据
            if isinstance(input_data, str):
                message = input_data
            else:
                message = json.dumps(input_data, ensure_ascii=False)
            
            logger.info("正在生成首次搜索查询")
            
            # 调用LLM（流式，安全拼接UTF-8）
            response = self.llm_client.stream_invoke_to_string(SYSTEM_PROMPT_FIRST_SEARCH, message)
            
            # 处理响应
            processed_response = self.process_output(response)
            
            logger.info(f"生成搜索查询: {processed_response.get('search_query', 'N/A')}")
            return processed_response
            
        except Exception as e:
            logger.exception(f"生成首次搜索查询失败: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> Dict[str, str]:
        """
        处理LLM输出，提取搜索查询和推理
        
        Args:
            output: LLM原始输出
            
        Returns:
            包含search_query和reasoning的字典
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
                # 使用更强大的提取方法
                result = extract_clean_response(cleaned_output)
                if "error" in result:
                    logger.error("JSON解析失败，尝试修复...")
                    # 尝试修复JSON
                    fixed_json = fix_incomplete_json(cleaned_output)
                    if fixed_json:
                        try:
                            result = json.loads(fixed_json)
                            logger.info("JSON修复成功")
                        except JSONDecodeError:
                            logger.error("JSON修复失败")
                            # 返回默认查询
                            return self._get_default_search_query()
                    else:
                        logger.error("无法修复JSON，使用默认查询")
                        return self._get_default_search_query()
            
            # 验证和清理结果
            search_query = result.get("search_query", "")
            reasoning = result.get("reasoning", "")
            
            if not search_query:
                logger.warning("未找到搜索查询，使用默认查询")
                return self._get_default_search_query()

            # 注意：此处只回传 search_query 与 reasoning，刻意丢弃 LLM 可能给出的
            # search_tool/platform/start_date/end_date/time_period。所以 agent 侧拿不到工具选择，
            # 最终总会回退默认的 search_topic_globally（见 agent._initial_search_and_summary）。
            return {
                "search_query": search_query,
                "reasoning": reasoning
            }
            
        except Exception as e:
            self.log_error(f"处理输出失败: {str(e)}")
            # 返回默认查询
            return self._get_default_search_query()
    
    def _get_default_search_query(self) -> Dict[str, str]:
        """
        获取默认搜索查询，根据话题特征选择合适的平台。

        Returns:
            默认的搜索查询字典
        """
        return {
            "search_query": "相关主题研究",
            "reasoning": "由于解析失败，使用默认搜索查询"
        }

    @staticmethod
    def detect_topic_region(topic: str) -> Dict[str, bool]:
        """
        根据话题关键词检测是否涉及南非/X平台等国际舆情场景。

        Args:
            topic: 话题文本

        Returns:
            平台建议字典，如 {'sa_news': True, 'x': True, 'domestic': True}
        """
        topic_lower = topic.lower()
        sa_keywords = ['南非', 'south africa', 'sa', 'pretoria', 'johannesburg', 'cape town',
                       '开普敦', '约翰内斯堡', '比勒陀利亚', '祖玛', '拉马福萨', 'anc',
                       'news24', 'iol', 'sowetan', 'mail & guardian', 'sunday times']
        x_keywords = ['twitter', 'x平台', 'x 平台', '推特', '推文', 'tweet', 'elon musk',
                      'x.com', '@', 'hashtag']

        has_sa = any(kw in topic_lower for kw in sa_keywords)
        has_x = any(kw in topic_lower for kw in x_keywords)

        return {
            'sa_news': has_sa,
            'x': has_x,
            'domestic': True,  # 国内平台始终搜索
        }


class ReflectionNode(BaseNode):
    """反思节点：基于段落当前内容找信息缺口，产出补充检索词(+reasoning)。"""
    
    def __init__(self, llm_client):
        """
        初始化反思节点
        
        Args:
            llm_client: LLM客户端
        """
        super().__init__(llm_client, "ReflectionNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                required_fields = ["title", "content", "paragraph_latest_state"]
                return all(field in data for field in required_fields)
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            required_fields = ["title", "content", "paragraph_latest_state"]
            return all(field in input_data for field in required_fields)
        return False
    
    def run(self, input_data: Any, **kwargs) -> Dict[str, str]:
        """
        调用LLM反思并生成搜索查询
        
        Args:
            input_data: 包含title、content和paragraph_latest_state的字符串或字典
            **kwargs: 额外参数
            
        Returns:
            包含search_query和reasoning的字典
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("输入数据格式错误，需要包含title、content和paragraph_latest_state字段")
            
            # 准备输入数据
            if isinstance(input_data, str):
                message = input_data
            else:
                message = json.dumps(input_data, ensure_ascii=False)
            
            logger.info("正在进行反思并生成新搜索查询")
            
            # 调用LLM（流式，安全拼接UTF-8）
            response = self.llm_client.stream_invoke_to_string(SYSTEM_PROMPT_REFLECTION, message)
            
            # 处理响应
            processed_response = self.process_output(response)
            
            logger.info(f"反思生成搜索查询: {processed_response.get('search_query', 'N/A')}")
            return processed_response
            
        except Exception as e:
            logger.exception(f"反思生成搜索查询失败: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> Dict[str, str]:
        """
        处理LLM输出，提取搜索查询和推理
        
        Args:
            output: LLM原始输出
            
        Returns:
            包含search_query和reasoning的字典
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
                # 使用更强大的提取方法
                result = extract_clean_response(cleaned_output)
                if "error" in result:
                    logger.error("JSON解析失败，尝试修复...")
                    # 尝试修复JSON
                    fixed_json = fix_incomplete_json(cleaned_output)
                    if fixed_json:
                        try:
                            result = json.loads(fixed_json)
                            logger.info("JSON修复成功")
                        except JSONDecodeError:
                            logger.error("JSON修复失败")
                            # 返回默认查询
                            return self._get_default_reflection_query()
                    else:
                        logger.error("无法修复JSON，使用默认查询")
                        return self._get_default_reflection_query()
            
            # 验证和清理结果
            search_query = result.get("search_query", "")
            reasoning = result.get("reasoning", "")
            
            if not search_query:
                logger.warning("未找到搜索查询，使用默认查询")
                return self._get_default_reflection_query()

            # 同 FirstSearchNode：只回传 search_query 与 reasoning，丢弃工具/平台/日期等字段。
            return {
                "search_query": search_query,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.exception(f"处理输出失败: {str(e)}")
            # 返回默认查询
            return self._get_default_reflection_query()
    
    def _get_default_reflection_query(self) -> Dict[str, str]:
        """
        获取默认反思搜索查询

        Returns:
            默认的反思搜索查询字典
        """
        return {
            "search_query": "深度研究补充信息",
            "reasoning": "由于解析失败，使用默认反思搜索查询"
        }

    @staticmethod
    def build_cross_platform_query(
        topic: str,
        current_platforms: set,
        start_date: str = None,
        end_date: str = None,
    ) -> list:
        """
        在反思阶段构建跨平台补充搜索查询。

        当检测到当前搜索仅覆盖国内平台时，自动生成针对 x 和/或 sa_news
        平台的并行搜索查询，实现新闻媒体与社交媒体的联合数据采集。

        Args:
            topic: 搜索话题
            current_platforms: 已搜索的平台集合
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD

        Returns:
            补充搜索配置列表，每项包含 platform/topic/start_date/end_date
        """
        topic_info = FirstSearchNode.detect_topic_region(topic)
        supplementary = []

        cross_platforms = {
            'sa_news': '南非新闻',
            'x': 'X平台',
        }
        for platform, label in cross_platforms.items():
            if platform in current_platforms:
                continue
            if topic_info.get(platform):
                supplementary.append({
                    'platform': platform,
                    'label': label,
                    'topic': topic,
                    'start_date': start_date,
                    'end_date': end_date,
                    'reason': f'话题涉及{label}相关关键词，补充搜索以覆盖国际舆情',
                })

        return supplementary
