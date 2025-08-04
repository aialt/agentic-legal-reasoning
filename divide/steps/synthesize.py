# -*- coding: utf-8 -*-
"""
Stage 4: Synthesize - Final Answer Generation and Polishing
"""
from services.llm_interface import llm_generation_call
from configs import settings

def generate_final_answer(original_query: str, context: str) -> str:
    """
    Based on the original query and the intermediate results from previous steps,
    generates the final, comprehensive answer.
    """
    # Note: This prompt is kept in Chinese as it is the direct instruction for the Chinese-language model.
    system_prompt = """你是一位顶级的中国法律分析专家。你的任务是基于“背景信息”中提供的初步分析，结合你自己的专业知识，为用户的“原始问题”生成一份结构清晰、逻辑严谨、专业且全面的最终回答。

    # 指示:
    1.  **综合与提炼**: 不要简单复述背景信息。你需要对其中的内容进行批判性地思考、整合、提炼，并补充你自己的专业见解。
    2.  **结构清晰**: 你的回答应该有清晰的逻辑层次，例如可以使用标题、列表（1, 2, 3）等方式来组织内容。
    3.  **专业严谨**: 使用准确的法律术语，避免口语化和模糊的表达。
    4.  **聚焦问题**: 最终回答必须直接、完整地回应用户的原始问题。"""

    full_prompt = (
        f"{system_prompt}\n\n"
        f"--- 背景信息 ---\n"
        f"用户的原始问题: {original_query}\n\n"
        f"初步分析材料:\n{context}\n\n"
        f"--- 最终回答 ---\n"
        f"请现在开始生成你的最终回答："
    )

    return llm_generation_call(full_prompt, settings.SYNTHESIZE_AGENT_MODEL_URL)
