# -*- coding: utf-8 -*-
"""
LLM Model Interaction Interface Module

This module encapsulates all direct communication with the local large language model services.
It includes:
1. Direct API request functions (for Triage, Decompose, Synthesize).
2. A custom LangChain LLM class (for the Execute Agent).
"""
import requests
import json
import re
from typing import Optional, List, Dict, Any

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from configs import settings # Import unified configurations

def _extract_json_from_response(text: str) -> Dict[str, Any] | None:
    """Robustly extracts a JSON object from the text returned by the LLM."""
    # Prioritize matching Markdown format
    matches = re.findall(r'```(?:json)?\s*({[\s\S]*?})\s*```', text)
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass # If parsing fails, continue to the next method

    # Second, match a raw JSON object
    matches = re.findall(r'({[\s\S]*?})', text)
    for match in matches:
        try:
            # Simple validation to ensure it looks like a JSON
            if '"' in match and ':' in match:
                return json.loads(match)
        except json.JSONDecodeError:
            continue
            
    return None

def llm_api_call(prompt: str, model_url: str, timeout: int = 60) -> Dict[str, Any] | None:
    """
    Sends a direct API request to the local LLM service and returns the parsed JSON result.
    This is the foundation for the Triage and Decompose steps.
    """
    raw_output = ""
    try:
        response = requests.post(model_url, json={"prompt": prompt}, timeout=timeout)
        response.raise_for_status()
        raw_output = response.text.strip()
        
        # The server returns in the format {"text": "{\"key\": \"value\"}"}, requiring double parsing
        outer_json = json.loads(raw_output)
        inner_json_str = outer_json.get("text", str(outer_json))
        
        # Attempt to parse the inner JSON string
        result_json = _extract_json_from_response(inner_json_str)
        if result_json:
            return result_json
        else:
            # If it cannot be parsed as JSON, print a warning and return None
            print(f"⚠️ LLM API Call Warning: Could not extract a valid JSON from the model response.\nOriginal inner response: {inner_json_str}")
            return None

    except Exception as e:
        print(f"❌ LLM API Call Failed: {e}\nModel URL: {model_url}\nRaw Output: {raw_output}")
        return None


def llm_generation_call(prompt: str, model_url: str, timeout: int = 120) -> str:
    """
    Sends a generative API request to the local LLM service and returns the plain text result.
    This is the foundation for the Synthesize step.
    """
    raw_output = ""
    try:
        response = requests.post(model_url, json={"prompt": prompt}, timeout=timeout)
        response.raise_for_status()
        raw_output = response.text.strip()
        
        data = json.loads(raw_output)
        return data.get("text", f"[Generation failed: {raw_output}]").strip()
    
    except Exception as e:
        print(f"❌ LLM Generation Call Failed: {e}\nModel URL: {model_url}\nRaw Output: {raw_output}")
        return f"[Error during final summary generation: {e}]"


class LangChainLocalLLM(LLM):
    """
    A custom local LLM wrapper for the LangChain Agent.
    It is responsible for formatting the LangChain prompt, sending it to the local service,
    and formatting the local service's JSON output back into the Action format expected by LangChain.
    """
    model_url: str = settings.SYNTHESIZE_AGENT_MODEL_URL

    @property
    def _llm_type(self) -> str:
        return "local-agent-llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        # This System Prompt is specifically designed for tool calling, guiding the model to output JSON
        system_prompt = """你是一个工具助手，只能用如下 JSON 调用工具或给出最终回答：
                        {
                          "action": "工具名称或Final Answer",
                          "action_input": "输入给工具的字符串或最终的回答"
                        }
                        不要返回注释或额外文字。必须只返回 JSON。"""
        
        full_prompt = f"{system_prompt}\n\nHuman: {prompt}\nAssistant:"
        
        try:
            response = requests.post(self.model_url, json={"prompt": full_prompt})
            response.raise_for_status()
            raw_output = response.json().get("text", "").strip()
            print(f"\n[DEBUG] Agent LLM Raw Output:\n{raw_output}\n")

            # Robustly extract JSON
            json_obj = _extract_json_from_response(raw_output)
            
            if json_obj and "action" in json_obj and "action_input" in json_obj:
                print(f"[DEBUG] Extracted Agent JSON:\n{json.dumps(json_obj, ensure_ascii=False, indent=2)}\n")
                # Format for LangChain's expected output
                return f"Action:\n```json\n{json.dumps(json_obj, ensure_ascii=False)}\n```"
            else:
                # If the output is not valid JSON or has the wrong format, treat it as a Final Answer
                print(f"[WARN] Agent LLM output was not standard JSON, treating as Final Answer.")
                fallback_action = {
                    "action": "Final Answer",
                    "action_input": raw_output
                }
                return f"Action:\n```json\n{json.dumps(fallback_action, ensure_ascii=False)}\n```"

        except Exception as e:
            print(f"❌ Agent LLM call or parsing failed: {e}")
            # In case of a critical error, also return a Final Answer to avoid interrupting the flow
            error_action = {
                "action": "Final Answer",
                "action_input": f"An error occurred during processing: {e}"
            }
            return f"Action:\n```json\n{json.dumps(error_action, ensure_ascii=False)}\n```"
