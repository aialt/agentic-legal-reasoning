# -*- coding: utf-8 -*-
"""
Core Workflow Orchestration Module

This module implements the four-stage Agent workflow,
chaining together the assess, Decompose, Execute, and Synthesize steps.
"""
import json
from langchain.agents import AgentExecutor

from .steps import assess, decompose, synthesize

def run_workflow(user_query: str, answering_agent: AgentExecutor):
    """
    Executes the complete "Divide and Enhance" Agent workflow.

    Args:
        user_query (str): The user's original legal question.
        answering_agent (AgentExecutor): A pre-built answering agent for executing sub-tasks.

    Returns:
        str: The generated final legal opinion.
    """
    print("\n" + "="*50)
    print("🚀 Starting Agent workflow execution...")
    print(f"Original Query: {user_query}")
    print("="*50 + "\n")
    
    final_context_for_synthesis = ""

    # --- Stage 1: assess ---
    print("--- [Stage 1/4] assess: assessing question type ---")
    question_type = assess.get_question_type(user_query)
    print(f"assessment Result: {question_type}\n")

    if question_type == "complex":
        # --- Stage 2: Decompose (for complex questions only) ---
        print("--- [Stage 2/4] Decompose: Performing DFLRA decomposition ---")
        structured_questions = decompose.decompose_question(user_query)

        if structured_questions and all(k in structured_questions for k in ["Definition", "Fact", "Legal", "Reasoning", "Action"]):
            print("✅ Question decomposition successful:")
            print(json.dumps(structured_questions, indent=2, ensure_ascii=False))
            
            # --- Stage 3: Execute (for complex questions only) ---
            print("\n--- [Stage 3/4] Execute: Executing sub-tasks one by one ---")
            answers = {}
            for key in ["Definition", "Fact", "Legal", "Reasoning", "Action"]:
                sub_question = structured_questions.get(key, "")
                if not sub_question: continue
                
                print(f"\n▶️ Executing sub-task [{key}]: {sub_question}")
                try:
                    response = answering_agent.invoke({"input": sub_question, "chat_history": []})
                    answers[key] = response.get('output', '[No answer]')
                except Exception as e:
                    answers[key] = f"[Failed to answer: {e}]"
                print(f"◀️ Sub-task [{key}] completed.")
            
            # Prepare the context for the Synthesize step
            context_parts = []
            for key in ["Definition", "Fact", "Legal", "Reasoning", "Action"]:
                q = structured_questions.get(key, 'N/A')
                a = answers.get(key, 'N/A')
                context_parts.append(f"[{key}]\nQuestion: {q}\nPreliminary Answer: {a}\n")
            final_context_for_synthesis = "\n".join(context_parts)

        else:
            print("⚠️ Decomposition failed or format is incomplete, downgrading to simple question processing.")
            question_type = "simple"

    # Path for simple questions or after a complex question is downgraded
    if question_type == "simple":
        print("\n--- [Stage 3/4] Execute: Directly processing simple question ---")
        try:
            response = answering_agent.invoke({"input": user_query, "chat_history": []})
            direct_answer = response.get('output', '[No answer]')
            final_context_for_synthesis = f"Direct Answer: {direct_answer}"
        except Exception as e:
            final_context_for_synthesis = f"[Failed to get direct answer: {e}]"
        print("✅ Direct processing complete.")
    
    # --- Stage 4: Synthesize ---
    if final_context_for_synthesis:
        print("\n--- [Stage 4/4] Synthesize: Generating final comprehensive opinion ---")
        final_answer = synthesize.generate_final_answer(user_query, final_context_for_synthesis)
        print("✅ Final opinion generated.")
        return final_answer
    else:
        return "❌ Workflow execution failed, no valid content was generated."
