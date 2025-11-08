# core.py
from __future__ import annotations

import os
import re
import math
import uuid
import itertools
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit
from datetime import datetime
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
try:
    from ddgs import DDGS  # type: ignore
except ImportError:
    from duckduckgo_search import DDGS  # type: ignore

# Initialize LLM (Gemini via LangChain integration)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_output_tokens=None,
    timeout=60,
    max_retries=3,
)

ACADEMIC_SITES_FILTER = (
    "site:arxiv.org OR site:neurips.cc OR site:icml.cc OR site:iclr.cc OR "
    "site:aaai.org OR site:ijcai.org OR site:thecvf.com OR site:kdd.org OR "
    "site:sigcomm.org OR site:usenix.org OR site:ieeexplore.ieee.org"
)


def parse_year_from_text(text: str) -> Optional[int]:
    """Extract publication year from text."""
    years = re.findall(r"\b(19|20)\d{2}\b", text or "")
    return int(years[0]) if years else None


def _normalize_url(u: str) -> str:
    if not u:
        return ""
    try:
        parts = urlsplit(u.strip())
        return urlunsplit(
            (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", "")
        )
    except Exception:
        return u.strip().rstrip("/").lower()


def _safe_ddgs_text_call(
    ddgs: DDGS,
    query: str,
    region: str,
    safesearch: str,
    timelimit: Optional[str],
    max_results: Optional[int],
    backend: Optional[str] = None,
    retries: int = 2,
) -> List[Dict[str, Any]]:
    """
    Call DDGS().text with graceful handling of different library signatures and backend fallbacks.
    """
    candidate_backends = []
    if backend:
        candidate_backends.append(backend)
    candidate_backends.extend(
        [b for b in ["lite", "html", "api", "auto"] if b != backend]
    )

    for b in candidate_backends:
        for _ in range(max(1, retries)):
            try:
                res = ddgs.text(
                    query,
                    region=region,
                    safesearch=safesearch,
                    timelimit=timelimit,
                    backend=b,
                    max_results=max_results,
                )
                if res is None:
                    results = []
                elif isinstance(res, list):
                    results = res
                else:
                    results = list(res)
            except TypeError:
                try:
                    res = ddgs.text(
                        query,
                        region=region,
                        safesearch=safesearch,
                        timelimit=timelimit,
                    )
                    results = list(res) if res is not None else []
                    if max_results:
                        results = results[:max_results]
                except Exception:
                    results = []
            except Exception:
                results = []

            if results:
                return results
    return []


def _build_query_prompt() -> ChatPromptTemplate:
    """
    Prompt to generate 2–3 short keyword queries for academic literature search.
    """
    return ChatPromptTemplate.from_template(
        """
Act as a query planner for academic literature search.
Given a topic, produce 2–3 distinct, short keyword-based queries optimized for academic sources.
Requirements:
- Be concise (each query < 12 words).
- Avoid punctuation except site: filters or boolean OR if needed.
- Prefer neutral, general keywords and important synonyms.
- Return ONLY the queries, one per line, no numbering or extra text.

Topic:
{topic}
""".strip()
    )


def generate_search_queries(topic: str, k: int = 3) -> List[str]:
    """
    Use the LLM to propose 2–3 concise queries for web search.
    """
    prompt = _build_query_prompt()
    msgs = prompt.format_messages(topic=(topic or "").strip())
    try:
        out = (llm.invoke(msgs).content or "").strip()
    except Exception:
        out = ""

    queries = [q.strip() for q in out.splitlines() if q.strip()]
    seen = set()
    deduped = []
    for q in queries:
        if q.lower() not in seen:
            deduped.append(q)
            seen.add(q.lower())

    base = (topic or "").strip()
    if len(deduped) < 2:
        fallbacks = [
            base,
            f"{base} method comparison",
            f"{base} benchmarks",
            f"{base} survey review",
        ]
        for fb in fallbacks:
            if fb and fb.lower() not in seen:
                deduped.append(fb)
                seen.add(fb.lower())
            if len(deduped) >= max(2, k):
                break

    return deduped[: max(2, k)]


def fetch_literature_results_multi(
    topic: str,
    region: str = "wt-wt",
    max_results: int = 20,
    safesearch: str = "moderate",
    timelimit: Optional[str] = None,
    backend: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch academic results via DuckDuckGo across multiple LLM-generated queries.
    """
    queries = generate_search_queries(topic, k=3)
    per_query = max(3, math.ceil(max_results / max(1, len(queries))))
    results: List[Dict[str, Any]] = []

    try:
        with DDGS() as ddgs:
            for q in queries:
                q_aug = f"{q} {ACADEMIC_SITES_FILTER}"
                rows = _safe_ddgs_text_call(
                    ddgs,
                    q_aug,
                    region=region,
                    safesearch=safesearch,
                    timelimit=timelimit,
                    max_results=per_query,
                    backend=backend,
                    retries=2,
                )
                for r in rows or []:
                    results.append(
                        {
                            "title": r.get("title", "") or "",
                            "body": r.get("body", "") or "",
                            "link": r.get("href", "") or "",
                            "source": r.get("source", "web") or "web",
                            "query_used": q,
                        }
                    )
    except Exception:
        return []

    deduped: List[Dict[str, Any]] = []
    seen_links = set()
    for row in results:
        norm = _normalize_url(row.get("link", ""))
        if norm and norm not in seen_links:
            deduped.append(row)
            seen_links.add(norm)

    return deduped[:max_results]


def _build_table_prompt() -> ChatPromptTemplate:
    """
    Prompt to produce a Markdown table for literature review.
    """
    return ChatPromptTemplate.from_template(
        """
You are a meticulous academic research analyst specializing in synthesizing scholarly publications.
You will examine the provided list of paper titles and abstracts in detail.

Your objective is to produce a high-quality, chronologically sorted (latest → oldest) literature review table in Markdown format.

For each paper, you must:
- Accurately determine the Year (from metadata, title, or context; estimate if unclear).
- Identify and list the Title in full.
- Extract or infer Authors from the text; if not stated, write 'N/A'.
- Summarize Key Contribution / Findings in 1–2 precise, academically phrased sentences.
- Record Citation Count if mentioned; if not, write 'N/A'.
- Provide the Source Link if present; if absent, write 'N/A'.

Additional requirements:
- If publication venue (journal/conference) is mentioned, briefly note it in parentheses after the year.
- Use neutral, scholarly tone and avoid unnecessary adjectives.
- Ensure all summaries focus on the core novel contribution, methodology highlights, and notable results.
- Maintain uniform formatting for all rows and ensure alignment of columns in Markdown.
- Double-check chronological order: newest year first, oldest last.

Topic: {topic}

Papers:
{compiled_text}

Now output ONLY the Markdown table. Do not include commentary before or after the table.
""".strip()
    )


def _build_chat_prompt() -> ChatPromptTemplate:
    """Prompt for normal chat responses (no web formatting)."""
    return ChatPromptTemplate.from_template(
        """
You are a Financial Sustainability Analytics Assistant integrated into a Gradio-based web application. Your role is to help users understand their spending patterns through a sustainability lens while providing actionable insights for reducing environmental impact.

## Core Responsibilities

1. **Spending Analysis**
   - Analyze credit card transactions (date, description, credit, debit)
   - Auto-categorize transactions based on merchant descriptions using intelligent keyword matching
   - Calculate sustainability scores (0-100) based on spending category weights
   - High sustainability: groceries (70), public_transport (90), utilities (60), healthcare (80), education (85)
   - Low sustainability: gas (20), rideshare (25), online_shopping (35), shopping (30)

2. **Seasonal Pattern Recognition**
   - Separate spending data into Northern Hemisphere seasons: Winter (Dec-Feb), Spring (Mar-May), Summer (Jun-Aug), Fall (Sep-Nov)
   - Group categories by similarity: Essential Spending, Transportation, Food & Entertainment, Discretionary, Education & Growth, Travel & Leisure
   - Identify seasonal trends and spending variations
   - Provide season-specific insights and recommendations

3. **Data Visualization**
   - Generate pie charts showing overall category spending distribution
   - Create grouped bar charts comparing spending across seasons by category group
   - Display line charts tracking cumulative spending over time
   - Ensure all visualizations are clear, interactive (Plotly), and actionable

4. **LLM-Powered Insights**
   - Use Gemini 2.5 Flash Lite to generate nuanced sustainability analysis
   - Provide personalized recommendations based on actual spending patterns
   - Highlight positive sustainable behaviors and opportunities for improvement
   - Suggest eco-friendly alternatives for high-impact spending categories
   - Keep analysis grounded in user's specific data, not generic advice

5. **Research Chat**
   - Support literature review generation with web search (DuckDuckGo academic sources)
   - Maintain multi-turn conversation history per user session
   - Generate chronologically sorted Markdown tables for academic papers
   - Provide concise, helpful responses for research-related queries

## Technical Guidelines

1. **Data Processing**
   - Handle date formats flexibly (support DD/MM/YYYY, MM/DD/YYYY, ISO8601)
   - Filter zero/null debit values
   - Maintain data integrity throughout transformations
   - Use pandas for all dataframe operations

2. **Category Mapping**
   - Match descriptions to categories using keyword lists (e.g., "Whole Foods" → groceries, "Uber" → rideshare)
   - Default unknown categories to "shopping"
   - Support 12 primary categories with extensible design

3. **Sustainability Scoring**
   - Calculate weighted average: (sum of amount × weight) / total_amount
   - Scale 0-100 for interpretability
   - Provide breakdown by category with percentages

4. **Error Handling**
   - Provide clear, actionable error messages
   - Validate required columns: date, description, credit, debit
   - Gracefully handle missing or malformed data
   - Return user-friendly explanations for failures

5. **UI/UX Principles**
   - Two-tab interface: Sustainability Analysis and Research Chat
   - Clear visual hierarchy with markdown headers
   - Progress bars for sustainability scores
   - Session-persistent state for user continuity
   - Responsive design with Gradio Soft theme

## Tone and Style

- **Sustainability Messaging**: Encouraging, non-judgmental, action-oriented
- **Data Presentation**: Clear, specific with numbers and percentages
- **Recommendations**: Practical, realistic, culturally aware
- **Academic**: Professional, scholarly, evidence-based
- **Overall**: Helpful, informative, accessible to non-technical users

## Key Outputs

1. Markdown reports with sustainability scores and breakdowns
2. Interactive Plotly visualizations (pie, bar, line charts)
3. Seasonal spending trends and category insights
4. LLM-generated personalized recommendations
5. Academic literature review tables with web search results
6. Conversational research chat with context preservation

## Do Not

- Make judgmental statements about user spending
- Provide generic, template-based advice
- Ignore data-driven insights in favor of assumptions
- Fail to respect user privacy (no external data sharing)
- Provide financial advice beyond sustainability context


Your Response:
""".strip()
    )


def extract_category_from_description(description: str) -> str:
    """
    Infer spending category from transaction description.
    """
    description_lower = description.lower()

    category_keywords = {
        "groceries": [
            "grocery",
            "whole foods",
            "trader joe",
            "safeway",
            "kroger",
            "supermarket",
            "market",
        ],
        "gas": ["shell", "chevron", "bp", "exxon", "texaco", "gas station", "fuel"],
        "public_transport": [
            "metro",
            "transit",
            "bus",
            "train",
            "subway",
            "transportation",
        ],
        "restaurants": [
            "restaurant",
            "cafe",
            "coffee",
            "pizza",
            "burger",
            "dining",
            "food delivery",
            "doordash",
            "ubereats",
        ],
        "shopping": ["mall", "store", "retail", "shop"],
        "utilities": ["electric", "water", "gas bill", "internet", "phone bill"],
        "entertainment": ["movie", "cinema", "concert", "theater", "streaming"],
        "healthcare": ["pharmacy", "doctor", "hospital", "medical", "clinic"],
        "education": ["school", "university", "tuition", "course"],
        "online_shopping": ["amazon", "ebay", "online store"],
        "rideshare": ["uber", "lyft", "taxi", "cab"],
        "travel": ["airline", "hotel", "booking", "airbnb"],
    }

    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in description_lower:
                return category

    return "shopping"


def get_season(date_str: str) -> str:
    """
    Determine season from date string.
    Northern Hemisphere: Winter (Dec-Feb), Spring (Mar-May), Summer (Jun-Aug), Fall (Sep-Nov)
    """
    try:
        date_obj = pd.to_datetime(date_str)
        month = date_obj.month

        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:  # 9, 10, 11
            return "Fall"
    except Exception:
        return "Unknown"


def separate_by_seasons(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Separate dataframe by seasons based on date column.
    Returns dict with seasonal dataframes.
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain 'date' column")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    df["season"] = df["date"].apply(lambda x: get_season(x))

    seasonal_dfs = {}
    for season in ["Winter", "Spring", "Summer", "Fall"]:
        seasonal_dfs[season] = df[df["season"] == season].copy()

    return seasonal_dfs


def group_categories_by_similarity(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Group spending categories by similarity and sustainability impact.
    Returns dict with grouped categories.
    """
    category_groups = {
        "Essential Spending": ["groceries", "utilities", "healthcare"],
        "Transportation": ["gas", "public_transport", "rideshare"],
        "Food & Entertainment": ["restaurants", "entertainment"],
        "Discretionary": ["shopping", "online_shopping"],
        "Education & Growth": ["education"],
        "Travel & Leisure": ["travel"],
    }

    return category_groups


def aggregate_by_group(
    df: pd.DataFrame, category_groups: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Aggregate spending by category groups.
    Returns dict with group totals.
    """
    if "category" not in df.columns or "debit" not in df.columns:
        return {}

    group_totals = {}

    for group_name, categories in category_groups.items():
        total = df[df["category"].isin(categories)]["debit"].sum()
        group_totals[group_name] = total

    # Add uncategorized spending
    known_categories = set()
    for cats in category_groups.values():
        known_categories.update(cats)

    uncategorized = df[~df["category"].isin(known_categories)]["debit"].sum()
    if uncategorized > 0:
        group_totals["Other"] = uncategorized

    return group_totals


def _build_seasonal_analysis_prompt() -> ChatPromptTemplate:
    """Prompt for LLM to analyze seasonal spending patterns."""
    return ChatPromptTemplate.from_template(
        """
You are a financial sustainability advisor analyzing seasonal spending patterns.

Seasonal Spending Data:
{seasonal_summary}

Category Groups Summary:
{category_groups_summary}

Provide a comprehensive seasonal analysis including:
1. Seasonal spending trends and variations
2. Which categories drive spending in each season
3. Sustainability insights for each season
4. Actionable recommendations for seasonal spending optimization
5. Opportunities to reduce environmental impact by season

Format your response with clear markdown headers and be specific with numbers.
""".strip()
    )


def analyze_seasonal_patterns(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Separate data by seasons, group categories, and generate LLM analysis.
    Returns tuple of (seasonal_dfs, llm_analysis).
    """
    if "date" not in df.columns or "debit" not in df.columns:
        return {}, "Error: DataFrame must contain 'date' and 'debit' columns"

    # Add category column if not present
    if "category" not in df.columns and "description" in df.columns:
        df = df.copy()
        df["category"] = df["description"].apply(extract_category_from_description)

    # Parse date with dayfirst=True for DD/MM/YYYY format
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")

    # Separate by seasons
    seasonal_dfs = separate_by_seasons(df)

    # Get category groups
    category_groups = group_categories_by_similarity(df)

    # Aggregate by group for each season
    seasonal_summary = {}
    for season, season_df in seasonal_dfs.items():
        if len(season_df) > 0:
            seasonal_summary[season] = aggregate_by_group(season_df, category_groups)

    # Create summary text for LLM
    seasonal_text = ""
    for season, groups in seasonal_summary.items():
        seasonal_text += f"\n**{season}**:\n"
        total = sum(groups.values())
        seasonal_text += f"  Total Spending: ${total:.2f}\n"
        for group, amount in sorted(groups.items(), key=lambda x: x[1], reverse=True):
            percentage = (amount / total * 100) if total > 0 else 0
            seasonal_text += f"  - {group}: ${amount:.2f} ({percentage:.1f}%)\n"

    category_groups_text = "Category Groups:\n"
    for group_name, categories in category_groups.items():
        category_groups_text += f"  - {group_name}: {', '.join(categories)}\n"

    # Use LLM to analyze
    prompt = _build_seasonal_analysis_prompt()
    msgs = prompt.format_messages(
        seasonal_summary=seasonal_text, category_groups_summary=category_groups_text
    )

    try:
        llm_analysis = llm.invoke(msgs).content
    except Exception as e:
        llm_analysis = f"Error generating analysis: {str(e)}"

    return seasonal_dfs, llm_analysis, seasonal_summary


def generate_pie_chart_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate pie chart data for category spending.
    Returns dict suitable for plotting.
    """
    if "category" not in df.columns or "debit" not in df.columns:
        return {}

    category_totals = df.groupby("category")["debit"].sum().sort_values(ascending=False)

    return {
        "labels": category_totals.index.tolist(),
        "values": category_totals.values.tolist(),
        "title": "Spending by Category",
    }


def generate_seasonal_pie_chart_data(
    seasonal_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, Any]]:
    """
    Generate pie chart data for each season.
    Returns dict with pie chart data for each season.
    """
    seasonal_charts = {}

    for season, season_df in seasonal_dfs.items():
        if len(season_df) > 0:
            seasonal_charts[season] = generate_pie_chart_data(season_df)

    return seasonal_charts


def generate_bar_chart_data(
    seasonal_summary: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Generate bar chart data for seasonal spending comparison.
    Returns dict suitable for plotting.
    """
    seasons = list(seasonal_summary.keys())
    group_names = set()

    for groups in seasonal_summary.values():
        group_names.update(groups.keys())

    group_names = sorted(list(group_names))

    data = {"seasons": seasons, "groups": group_names, "values": {}}

    for group in group_names:
        data["values"][group] = [
            seasonal_summary.get(season, {}).get(group, 0) for season in seasons
        ]

    return data


def generate_line_chart_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate line chart data for cumulative spending over time.
    Returns dict suitable for plotting.
    """
    if "date" not in df.columns or "debit" not in df.columns:
        return {}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    df = df.sort_values("date")
    df["cumulative"] = df["debit"].cumsum()

    return {
        "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "cumulative": df["cumulative"].tolist(),
        "title": "Cumulative Spending Over Time",
    }


def literature_review_table(
    topic: str,
    region: str = "us-en",
    max_results: int = 20,
    safesearch: str = "moderate",
    timelimit: Optional[str] = None,
    backend: Optional[str] = None,
) -> str:
    """
    Generate a literature review as a Markdown TABLE using multi-query web results.
    """
    articles = fetch_literature_results_multi(
        topic=topic,
        region=region,
        max_results=max_results,
        safesearch=safesearch,
        timelimit=timelimit,
        backend=backend,
    )

    if not articles:
        return (
            "| Intent | Reply |\n"
            "|--------|-------|\n"
            "| Info | No academic sources found for this topic; try refining the query or checking the connection. |\n"
        )

    compiled_text = ""
    for art in articles:
        compiled_text += (
            f"Title: {art.get('title', '')}\n"
            f"Abstract: {art.get('body', '')}\n"
            f"Source: {art.get('source', '')}\n"
            f"Link: {art.get('link', '')}\n\n"
        )

    prompt = _build_table_prompt()
    msgs = prompt.format_messages(topic=topic, compiled_text=compiled_text)

    try:
        response = llm.invoke(msgs).content
    except Exception as e:
        return (
            "| Intent | Reply |\n"
            "|--------|-------|\n"
            f"| Error | Error generating literature table: {str(e)} |\n"
        )

    if not isinstance(response, str) or "|" not in response:
        rows = []
        header = "| Year | Title | Authors | Key Contribution / Findings | Citations | Source |\n"
        sep = "|------|-------|---------|-----------------------------|-----------|--------|\n"
        for art in articles[: min(10, len(articles))]:
            title = art.get("title") or "Untitled"
            year = parse_year_from_text(art.get("body", "")) or "N/A"
            link = art.get("link") or ""
            rows.append(f"| {year} | {title} | N/A | N/A | N/A | {link} |\n")
        response = header + sep + "".join(rows)

    return response


def chat_response(message: str) -> str:
    """Generate normal conversational response (no table, no web)."""
    prompt = _build_chat_prompt()
    msgs = prompt.format_messages(message=message)

    try:
        response = llm.invoke(msgs).content
    except Exception as e:
        return f"I apologize, but an error occurred: {str(e)}\nPlease try again or rephrase the question."

    if not isinstance(response, str):
        return (
            "I apologize, but I couldn't generate a proper response. Please try again."
        )
    return response


def answer_as_table(
    message: str,
    region: str = "us-en",
    max_results: int = 20,
    safesearch: str = "moderate",
    timelimit: Optional[str] = None,
    backend: Optional[str] = None,
    force_web: bool = False,
) -> str:
    """
    Routing:
    - If force_web is True: return a Markdown TABLE (web).
    - If force_web is False: return plain chat text (no web).
    """
    message = (message or "").strip()
    if not message:
        return ""

    if force_web:
        return literature_review_table(
            message,
            region=region,
            max_results=max_results,
            safesearch=safesearch,
            timelimit=timelimit,
            backend=backend,
        )

    return chat_response(message)
