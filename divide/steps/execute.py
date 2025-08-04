# -*- coding: utf-8 -*-
"""
Stage 3: Execute - Execute Sub-tasks

This module is responsible for building and providing a LangChain Agent Executor.
This Executor acts as a "Q&A expert," solely responsible for answering the specific, decomposed sub-questions.
"""
from langchain.agents import Tool, AgentExecutor
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from services.llm_interface import LangChainLocalLLM
from divide.tools.web_search_tool import bocha_websearch_tool
from divide.tools.rag_tool import rag_law_search

def build_answering_agent() -> AgentExecutor:
    """
    Builds an Agent Executor focused on answering specific questions.
    It is equipped with two tools: RAG and Web Search.
    """
    # 1. Initialize the LLM used by the Agent
    llm = LangChainLocalLLM()

    # 2. Define the tools available to the Agent
    tools = [
        Tool.from_function(
            func=rag_law_search,
            name="LegalRAGSearch",
            description="""**【首选工具】** 当你需要查询中国的法律、法规、司法解释的具体条文时，调用此工具。输入必须是一个清晰的法律问题，最好包含具体的法律名称，如“《刑法》中关于正当防卫的规定”。"""
        ),
        Tool.from_function(
            func=bocha_websearch_tool,
            name="BochaWebSearch",
            description="""**【备用工具】** 仅在法律知识库（LegalRAGSearch）无法提供所需信息，且你需要查询与案件相关的、实时的、公开的外部背景信息（如公司状态、最新新闻事件）时使用。禁止用它来查询法律条文。"""
        ),
    ]
    
    # 3. Design the instructions (System Prompt) for the Agent's reasoning and actions.
    # Note: This prompt is kept in Chinese as it is the direct instruction for the Chinese-language model.
    system_prompt = """你是一个专业、严谨的中国法律问答助手。你的任务是根据用户提出的【当前问题】，智能地选择并利用工具，提供准确的回答。

        **思考与行动原则:**
        1.  **分析问题**: 理解问题的核心意图。是需要查法条，还是查背景信息？
        2.  **工具选择**:
            * 如果问题涉及具体的法律规定，**优先且必须**调用 `LegalRAGSearch`。
            * 如果需要查询实时的、非法律条文的背景信息，可以调用 `BochaWebSearch`。
            * 如果凭借自己的知识就能回答（比如一个简单的法律概念），可以直接回答，无需调用工具。
        3.  **整合答案**: 基于工具返回的信息或自身知识，生成一个清晰、有依据的回答。

        **--- 处理《民法典》的特别指令 ---**
        当你需要查询《民法典》时，你【必须】根据问题上下文判断它属于哪个分编，并在调用`LegalRAGSearch`时使用最精确的法规全名。例如：
        -   用户问：“离婚时财产如何分割？” -> 你的 `action_input` 应该是：`"《民法典-婚姻家庭编》中关于离婚财产分割的规定"`。
        -   用户问：“买的房子有质量问题怎么办？” -> 你的 `action_input` 应该是：`"《民法典-合同编》中关于买卖合同质量不符的规定"`。
        -   用户问：“关于居住权的规定是什么？” -> 你的 `action_input` 应该是：`"《民法典-物权编》中关于居住权的规定"`。"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # 4. Create the Agent and the AgentExecutor
    # The code correctly uses the StructuredChatAgent for this purpose.
    agent = StructuredChatAgent.from_llm_and_tools(llm=llm, tools=tools, prompt=prompt)
    
    # The AgentExecutor is correctly initialized.
    # The 'verbose=True' flag is helpful for debugging the agent's thought process.
    # The 'handle_parsing_errors' provides a robust fallback mechanism.
    return AgentExecutor.from_agent_and_tools(
        agent=agent, 
        tools=tools, 
        verbose=True, # Prints the agent's thought process to the console
        handle_parsing_errors="Please check that your output is in strict JSON format." # Error handling
    )
