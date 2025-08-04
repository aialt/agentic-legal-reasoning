# -*- coding: utf-8 -*-
"""
RAG System Utility Functions Module
"""
import re
from typing import List, Tuple
from fuzzywuzzy import fuzz
from cn2an import an2cn

def normalize_article_number(text: str) -> str:
    """
    Normalizes an article number from Arabic numerals to Chinese numerals.
    Example: "第120条" -> "第一百二十条"
    """
    match = re.match(r"第(\d+)条", text)
    if match:
        try:
            num = int(match.group(1))
            chinese_num = an2cn(num)
            return f"第{chinese_num}条"
        except Exception:
            return text
    return text

def fuzzy_match_law_name(law_name: str, all_law_names: List[str], threshold: int = 50) -> str:
    """
    Fuzzily finds the best matching law name from a list of all law names.
    """
    best_match = ""
    best_score = 0
    for candidate in all_law_names:
        score = fuzz.token_set_ratio(law_name, candidate)
        if score > best_score and score >= threshold:
            best_match = candidate
            best_score = score
    return best_match

def extract_and_match_law_name(query: str, all_law_names: List[str]) -> Tuple[str, str]:
    """
    Extracts and fuzzily matches a law name from the query.
    
    Returns:
        Tuple[str, str]: A tuple containing (matched_law_name, query_with_law_name_removed).
    """
    # Prioritize matching book title marks "《》"
    match = re.search(r"《(.+?)》", query)
    if match:
        raw_name = match.group(1).strip()
        query_wo_law = query.replace(f"《{raw_name}》", "").strip()
        matched_name = fuzzy_match_law_name(raw_name, all_law_names)
        return matched_name, query_wo_law

    # If no book title marks are found, attempt to fuzzily match within the entire query
    matched_name = fuzzy_match_law_name(query, all_law_names)
    if matched_name:
        # Remove the matched law name from the query to avoid redundancy
        query_wo_law = query.replace(matched_name, "").strip()
        return matched_name, query_wo_law

    return "", query
