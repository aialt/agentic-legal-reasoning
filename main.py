# -*- coding: utf-8 -*-
"""
Main entry point for the Legal AI Assistant.
"""
from divide.steps.execute import build_answering_agent
from divide.agent_workflow import run_workflow

def main():
    # --- 1. Initialization ---
    # According to the paper's framework, the Execute Agent is a core component and needs to be pre-built.
    print("正在构建问答Agent，请稍候...")
    try:
        answering_agent = build_answering_agent()
        print("✅ Answering agent built successfully!")
    except Exception as e:
        print(f"❌ Fatal Error: Could not build the answering agent. The program cannot start. Details: {e}")
        return

    print("\n=======================================================")
    print("⚖️  欢迎使用 Divide and Enhance 法律智能助手")
    print("=======================================================")
    print("您可以开始提问了。输入 'exit' 或 '退出' 即可结束程序。")

    # --- 2. Main Loop ---
    while True:
        user_query = input("\n💬 请输入您的问题:\n> ")
        if user_query.strip().lower() in ["exit", "退出"]:
            print("\n👋 感谢使用，再见！")
            break
        
        if not user_query.strip():
            continue

        try:
            # --- 3. Invoke the core workflow ---
            final_answer = run_workflow(user_query, answering_agent)
            
            # --- 4. Print the final result ---
            print("\n" + "="*50)
            print("📄【最终综合法律意见】")
            print("="*50)
            print(final_answer)

        except Exception as e:
            print(f"\n❌ An unexpected error occurred while processing the request: {e}")
            # More detailed error logging can be added here
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
