# -*- coding: utf-8 -*-
"""
External Web Search Tool
This module uses the Bocha API.
"""
import requests
from langchain.tools import tool
from configs import settings

@tool
def bocha_websearch_tool(query: str, count: int = 3) -> str:
    """
    【External Information Query Tool】
    Use this tool to perform real-time web searches to obtain public, timely background information outside the legal domain.
    """
    url = 'https://api.bochaai.com/v1/web-search'
    headers = {
        'Authorization': f'Bearer {settings.BOCHA_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {"query": query, "count": count, "summary": True}

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        results = response.json()

        if results.get("code") != 200:
            return f"API Error: {results.get('msg', 'Unknown error')}"

        webpages = results.get("data", {}).get("webPages", {}).get("value", [])
        if not webpages:
            return "No relevant web results found."

        # Format the output for better readability by the LLM
        formatted_results = []
        for i, page in enumerate(webpages):
            formatted_results.append(
                f"[{i+1}] Title: {page.get('name', 'N/A')}\n"
                f"Source: {page.get('siteName', 'N/A')}\n"
                f"Snippet: {page.get('summary', 'N/A')}"
            )
        return "\n---\n".join(formatted_results)

    except Exception as e:
        return f"Web search failed: {str(e)}"
