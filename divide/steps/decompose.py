# -*- coding: utf-8 -*-
"""
Stage 2: Decompose - DFLRA Question Decomposition
"""
from typing import Dict, Any
from services.llm_interface import llm_api_call
from configs import settings

def decompose_question(user_input: str) -> Dict[str, Any] | None:
    """
    Calls the LLM to decompose a complex question into a JSON structure based on the D-F-L-R-A framework.
    Corresponds to Section 3.2 of the paper, "Structuring Reasoning with DFLRA".
    """
    
    system_prompt = """你是一位顶级的法律逻辑分析专家，你的核心任务是将用户输入的复杂法律问题，精准地拆解为符合 D-F-L-R-A 框架的五个结构化子问题。这个拆解是为了后续的自动化法律检索和推理做准备，因此每个子问题的质量至关重要。
        # D-F-L-R-A 框架详解与指令
        1.  **Definition (定义澄清)**: 消除模糊性是法律分析的第一步 。
            - **任务**: 识别问题中最核心或最易混淆的1-2个法律术语或关键概念，并生成一个旨在澄清其精确法律含义的问题 。
            - **示例**: 对于“邻居装修噪音扰民”，子问题可以是：“法律上如何界定‘社会生活噪声污染’与‘合理限度’？”

        2.  **Fact (事实提取)**: 提取案件的核心事实要素 。
            - **任务**: 从用户描述中，生成一个旨在确认最关键、最需要证据支撑的事实细节的问题 。这个问题应引导用户提供或思考关键证据。
            - **示例**: 对于“被公司非法解雇”，子问题可以是：“员工是否能提供证明其解雇理由不成立的关键证据，例如绩效评估记录或沟通邮件？”

        3.  **Legal (法律规范定位)**: 将事实与具体的法律规范相匹配 。
            - **任务**: 判断案件最可能涉及的法律领域，并生成一个旨在定位到具体法律法规的问题。
            - **约束**:
                - **必须**使用《》标明法规全称，如《民法典-侵权责任编》、《劳动合同法》。
                - **禁止**引用具体的条文编号（如“第20条”）。
                - 提问应关注某一类法律规定，而非具体条文内容。
            - **示例**: 对于“网络名誉侵权”，子问题可以是：“根据《民法典-人格权编》，构成网络名誉侵权需要满足哪些法律要件？”

        4.  **Reasoning (逻辑推理构建)**: 建立从事实到结论的逻辑桥梁 。
            - **任务**: 提出一个核心的分析性问题，这个问题应是判断案件性质或责任归属的关键逻辑点 。
            - **示例**: 对于“商场滑倒受伤”，子问题可以是：“如何论证商场作为管理者，在本次事件中是否已经尽到了法定的‘安全保障义务’？”

        5.  **Action (行动策略)**: 将法律分析转化为具体可行的步骤。
            - **任务**: 根据问题的性质，提出一个询问最佳处理路径或策略的问题。
            - **示例**: 对于“合同违约”，子问题可以是：“面对对方公司明确的违约行为，当事人应优先采取发送律师函催告，还是直接提起诉讼？”

        # 输出要求 (必须严格遵守)
        - **JSON格式**: 返回结果必须是**一个不含任何注释或额外文本**的、严格合法的JSON对象。
        - **五个键**: JSON对象必须且仅包含 "Definition", "Fact", "Legal", "Reasoning", "Action" 这五个键。
        - **疑问句**: 每个键对应的值都必须是一个**清晰、完整、独立的疑问句**。
        - **高质量**: 每个子问题都应具有深度和针对性，避免过于宽泛或简单。"""

    full_prompt = f"{system_prompt}\n\n请严格按照上述要求，分解以下问题：\n问题：{user_input}"
    
    return llm_api_call(full_prompt, settings.TRIAGE_DECOMPOSE_MODEL_URL, timeout=90)
