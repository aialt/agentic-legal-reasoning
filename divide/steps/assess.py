# -*- coding: utf-8 -*-
"""
Stage 1: assess - Question Type assessment
"""
from services.llm_interface import llm_api_call
from configs import settings

def get_question_type_minimal_change(user_input: str) -> str:
    """
    Calls the LLM to determine if the question is 'simple' or 'complex'. (Concise modified version)
    """
    system_prompt = """你是一个高级法律问题路由器。你的唯一任务是分析用户输入，并严格按照规则将其分类为 'simple' 或 'complex'。

        # 规则:
        1.  **简单问题 (simple)**: 指的是可以通过**一次直接知识查询**就回答的问题。包括：
            - 查询**某个具体法条的原文内容**（如“《刑法》第一百二十条”）。
            - 查询**一个明确的法律概念定义**（如“什么是'正当防卫'？”）。
        2.  **复杂问题 (complex)**: 除上述“简单问题”之外的所有其他问题，特别是那些**描述了具体场景、包含人物事件**或**请求法律分析**的问题。

        # 输出格式:
        你的输出必须是**一个不包含任何其他文本**的JSON对象，格式如下：{ "type": "complex" } 或 { "type": "simple" }"""

    full_prompt = f"{system_prompt}\n\n请对以下问题进行分类：\n问题：{user_input}"
    
    result = llm_api_call(full_prompt, settings.TRIAGE_DECOMPOSE_MODEL_URL)
    
    return result.get("type", "complex") if result else "complex"
