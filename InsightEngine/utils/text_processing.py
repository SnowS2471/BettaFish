"""
文本处理工具函数

主要解决 LLM 输出「不规整」的问题，供各节点解析使用：
- JSON 清洗/修复：去 ```json 代码围栏、剥离推理前言、补全被截断的 JSON
  （remove_reasoning_from_output / clean_json_tags / extract_clean_response /
   fix_incomplete_json / fix_aggressive_json）；
- 结果格式化：format_search_results_for_prompt / truncate_content；
- 跨语言/跨平台（毕设扩展）：detect_content_language / merge_bilingual_results /
  normalize_platform_text / partition_by_platform。
"""

import re
import json
import unicodedata
from typing import Dict, Any, List, Optional
from json.decoder import JSONDecodeError


def clean_json_tags(text: str) -> str:
    """
    清理文本中的JSON标签
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    # 移除```json 和 ```标签
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'```', '', text)
    
    return text.strip()


def clean_markdown_tags(text: str) -> str:
    """
    清理文本中的Markdown标签
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    # 移除```markdown 和 ```标签
    text = re.sub(r'```markdown\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'```', '', text)
    
    return text.strip()


def remove_reasoning_from_output(text: str) -> str:
    """
    移除输出中的推理过程文本
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    # 查找JSON开始位置
    json_start = -1
    
    # 尝试找到第一个 { 或 [
    for i, char in enumerate(text):
        if char in '{[':
            json_start = i
            break
    
    if json_start != -1:
        # 从JSON开始位置截取
        return text[json_start:].strip()
    
    # 如果没有找到JSON标记，尝试其他方法
    # 移除常见的推理标识
    patterns = [
        r'(?:reasoning|推理|思考|分析)[:：]\s*.*?(?=\{|\[)',  # 移除推理部分
        r'(?:explanation|解释|说明)[:：]\s*.*?(?=\{|\[)',   # 移除解释部分
        r'^.*?(?=\{|\[)',  # 移除JSON前的所有文本
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text.strip()


def extract_clean_response(text: str) -> Dict[str, Any]:
    """
    提取并清理响应中的JSON内容
    
    Args:
        text: 原始响应文本
        
    Returns:
        解析后的JSON字典
    """
    # 清理文本
    cleaned_text = clean_json_tags(text)
    cleaned_text = remove_reasoning_from_output(cleaned_text)
    
    # 尝试直接解析
    try:
        return json.loads(cleaned_text)
    except JSONDecodeError:
        pass
    
    # 尝试修复不完整的JSON
    fixed_text = fix_incomplete_json(cleaned_text)
    if fixed_text:
        try:
            return json.loads(fixed_text)
        except JSONDecodeError:
            pass
    
    # 尝试查找JSON对象
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except JSONDecodeError:
            pass
    
    # 尝试查找JSON数组
    array_pattern = r'\[.*\]'
    match = re.search(array_pattern, cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except JSONDecodeError:
            pass
    
    # 如果所有方法都失败，返回错误信息
    print(f"无法解析JSON响应: {cleaned_text[:200]}...")
    return {"error": "JSON解析失败", "raw_text": cleaned_text}


def fix_incomplete_json(text: str) -> str:
    """
    修复不完整的JSON响应
    
    Args:
        text: 原始文本
        
    Returns:
        修复后的JSON文本，如果无法修复则返回空字符串
    """
    # 移除多余的逗号和空白
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # 检查是否已经是有效的JSON
    try:
        json.loads(text)
        return text
    except JSONDecodeError:
        pass
    
    # 检查是否缺少开头的数组符号
    if text.strip().startswith('{') and not text.strip().startswith('['):
        # 如果以对象开始，尝试包装成数组
        if text.count('{') > 1:
            # 多个对象，包装成数组
            text = '[' + text + ']'
        else:
            # 单个对象，包装成数组
            text = '[' + text + ']'
    
    # 检查是否缺少结尾的数组符号
    if text.strip().endswith('}') and not text.strip().endswith(']'):
        # 如果以对象结束，尝试包装成数组
        if text.count('}') > 1:
            # 多个对象，包装成数组
            text = '[' + text + ']'
        else:
            # 单个对象，包装成数组
            text = '[' + text + ']'
    
    # 检查括号是否匹配
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    
    # 修复不匹配的括号
    if open_braces > close_braces:
        text += '}' * (open_braces - close_braces)
    if open_brackets > close_brackets:
        text += ']' * (open_brackets - close_brackets)
    
    # 验证修复后的JSON是否有效
    try:
        json.loads(text)
        return text
    except JSONDecodeError:
        # 如果仍然无效，尝试更激进的修复
        return fix_aggressive_json(text)


def fix_aggressive_json(text: str) -> str:
    """
    更激进的JSON修复方法
    
    Args:
        text: 原始文本
        
    Returns:
        修复后的JSON文本
    """
    # 查找所有可能的JSON对象
    objects = re.findall(r'\{[^{}]*\}', text)
    
    if len(objects) >= 2:
        # 如果有多个对象，包装成数组
        return '[' + ','.join(objects) + ']'
    elif len(objects) == 1:
        # 如果只有一个对象，包装成数组
        return '[' + objects[0] + ']'
    else:
        # 如果没有找到对象，返回空数组
        return '[]'


def update_state_with_search_results(search_results: List[Dict[str, Any]], 
                                   paragraph_index: int, state: Any) -> Any:
    """
    将搜索结果更新到状态中
    
    Args:
        search_results: 搜索结果列表
        paragraph_index: 段落索引
        state: 状态对象
        
    Returns:
        更新后的状态对象
    """
    if 0 <= paragraph_index < len(state.paragraphs):
        # 获取最后一次搜索的查询（假设是当前查询）
        current_query = ""
        if search_results:
            # 从搜索结果推断查询（这里需要改进以获取实际查询）
            current_query = "搜索查询"
        
        # 添加搜索结果到状态
        state.paragraphs[paragraph_index].research.add_search_results(
            current_query, search_results
        )
    
    return state


def validate_json_schema(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    验证JSON数据是否包含必需字段
    
    Args:
        data: 要验证的数据
        required_fields: 必需字段列表
        
    Returns:
        验证是否通过
    """
    return all(field in data for field in required_fields)


def truncate_content(content: str, max_length: int = 20000) -> str:
    """
    截断内容到指定长度
    
    Args:
        content: 原始内容
        max_length: 最大长度
        
    Returns:
        截断后的内容
    """
    if len(content) <= max_length:
        return content
    
    # 尝试在单词边界截断
    truncated = content[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # 如果最后一个空格位置合理
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."


def format_search_results_for_prompt(search_results: List[Dict[str, Any]],
                                   max_length: int = 20000) -> List[str]:
    """
    格式化搜索结果用于提示词

    Args:
        search_results: 搜索结果列表
        max_length: 每个结果的最大长度

    Returns:
        格式化后的内容列表
    """
    formatted_results = []

    for result in search_results:
        content = result.get('content', '')
        if content:
            truncated_content = truncate_content(content, max_length)
            formatted_results.append(truncated_content)

    return formatted_results


# ===== 跨语言/跨平台文本处理（毕设扩展） =====

def detect_content_language(text: str) -> str:
    """
    检测文本主要语言。

    基于Unicode区间统计，快速判断文本为中/英/混合。
    用于对南非新闻原文（英文）和译文（中文）做不同处理。

    Args:
        text: 输入文本

    Returns:
        'zh' | 'en' | 'mixed'
    """
    if not text:
        return 'en'

    cjk_count = 0
    latin_count = 0

    for ch in text:
        cp = ord(ch)
        # CJK统一表意文字 + 中文标点周边
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
            0x3000 <= cp <= 0x303F or 0xFF00 <= cp <= 0xFFEF or
            0x2E80 <= cp <= 0x2FDF):
            cjk_count += 1
        elif ch.isalpha() and cp < 0x2000:
            latin_count += 1

    total = cjk_count + latin_count
    if total == 0:
        return 'en'

    cjk_ratio = cjk_count / total
    if cjk_ratio > 0.6:
        return 'zh'
    elif cjk_ratio < 0.2:
        return 'en'
    else:
        return 'mixed'


def merge_bilingual_results(
    results: List[Dict[str, Any]],
    content_field: str = 'content',
    content_zh_field: str = 'content_zh',
    title_field: str = 'title',
    title_zh_field: str = 'title_zh',
) -> List[Dict[str, Any]]:
    """
    将同一文章的原文与译文配对合并。

    针对 sa_news 平台搜索结果，检测每条结果是否同时包含原文和译文，
    统一为 {content, content_zh, title, title_zh, has_translation} 结构。

    Args:
        results: 原始搜索结果列表
        content_field: 原文内容字段名
        content_zh_field: 译文内容字段名
        title_field: 原文标题字段名
        title_zh_field: 中文标题字段名

    Returns:
        合并后的结果列表，每项增加 has_translation 标记
    """
    merged = []
    for r in results:
        item = dict(r) if isinstance(r, dict) else r._asdict() if hasattr(r, '_asdict') else {'content': str(r)}
        has_title_zh = bool(item.get(title_zh_field))
        has_content_zh = bool(item.get(content_zh_field))
        item['has_translation'] = has_title_zh or has_content_zh
        item.setdefault(content_zh_field, '')
        item.setdefault(title_zh_field, '')
        merged.append(item)
    return merged


def normalize_platform_text(text: str, platform: str) -> str:
    """
    统一不同平台文本格式，供 LLM 总结使用。

    - x/twitter: 保留 hashtag 和 @mention，清理多余换行
    - sa_news/news: 截断过长导语，统一段落间距
    - 其他平台: 仅做基础空白规范化

    Args:
        text: 原始文本
        platform: 平台标识 ('x', 'sa_news', 'twitter', 'news', 等)

    Returns:
        标准化后的文本
    """
    if not text:
        return ''

    # 统一 Unicode 规范化
    text = unicodedata.normalize('NFKC', text)

    # 基础清理：合并多余空白
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    platform_lower = platform.lower()
    if platform_lower in ('x', 'twitter', 'tweet'):
        # 推文保留 @mention 和 #hashtag，但限制连续换行
        text = re.sub(r'\n{2,}', ' ', text)
    elif platform_lower in ('sa_news', 'news', 'news_article'):
        # 新闻文本限制最大长度，保留段落结构
        if len(text) > 6000:
            text = text[:6000].rsplit(' ', 1)[0] + '...'
    else:
        text = re.sub(r'\n{2,}', '\n\n', text)

    return text.strip()


def partition_by_platform(
    results: List[Dict[str, Any]],
    platform_map: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    将混合平台的搜索结果按平台分组。

    Args:
        results: 搜索结果列表，每项需含 platform 或 source_table 字段
        platform_map: 可选的 platform -> display_name 映射

    Returns:
        {platform_name: [results]} 的分组字典
    """
    default_map = {
        'x': 'X平台',
        'twitter': 'X平台',
        'sa_news': '南非新闻',
        'news': '新闻媒体',
        'weibo': '微博',
        'douyin': '抖音',
        'bilibili': 'B站',
        'xhs': '小红书',
        'zhihu': '知乎',
        'kuaishou': '快手',
        'tieba': '贴吧',
    }
    if platform_map is None:
        platform_map = default_map

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        raw_platform = r.get('platform') or r.get('source_table') or 'unknown'
        label = platform_map.get(raw_platform, raw_platform)
        groups.setdefault(label, []).append(r)
    return groups
