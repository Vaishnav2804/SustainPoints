# app.py
import os
from dotenv import load_dotenv
import uuid
import pandas as pd
import gradio as gr
import json

load_dotenv()

from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableParallel

from core import (
    answer_as_table,
    extract_category_from_description,
    analyze_seasonal_patterns,
    generate_pie_chart_data,
    generate_seasonal_pie_chart_data,
    generate_bar_chart_data,
    generate_line_chart_data,
)

# Prompt scaffolding
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that analyzes credit card spending for sustainability.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def calculate_sustainability_score(df):
    """
    Calculate sustainability score based on spending categories.
    Returns score (0-100) and breakdown.
    """
    category_weights = {
        "groceries": 70,
        "public_transport": 90,
        "utilities": 60,
        "restaurants": 50,
        "gas": 20,
        "shopping": 30,
        "travel": 40,
        "entertainment": 50,
        "healthcare": 80,
        "education": 85,
        "online_shopping": 35,
        "rideshare": 25,
        "default": 50,
    }

    if "debit" not in df.columns or "description" not in df.columns:
        return None, "CSV must contain 'debit' and 'description' columns"

    df = df[df["debit"] > 0].copy()

    total_debit = df["debit"].sum()
    if total_debit == 0:
        return 0, "No spending data found"

    weighted_sum = 0
    category_breakdown = {}

    for _, row in df.iterrows():
        description = str(row["description"]).lower().strip()
        amount = float(row["debit"])

        category = extract_category_from_description(description)

        weight = category_weights.get(category, category_weights["default"])
        weighted_sum += amount * weight

        if category not in category_breakdown:
            category_breakdown[category] = {"amount": 0, "weight": weight}
        category_breakdown[category]["amount"] += amount

    score = weighted_sum / total_debit
    return score, category_breakdown


def create_pie_chart(chart_data):
    """Create plotly pie chart from chart data."""
    import plotly.graph_objects as go

    if not chart_data or "labels" not in chart_data:
        return None

    fig = go.Figure(
        data=[
            go.Pie(
                labels=chart_data["labels"],
                values=chart_data["values"],
                title=chart_data.get("title", "Spending Distribution"),
            )
        ]
    )

    fig.update_layout(height=500)
    return fig


def create_bar_chart(chart_data):
    """Create plotly bar chart for seasonal comparison."""
    import plotly.graph_objects as go

    if not chart_data or "seasons" not in chart_data:
        return None

    fig = go.Figure()

    for group in chart_data["groups"]:
        fig.add_trace(
            go.Bar(
                name=group,
                x=chart_data["seasons"],
                y=chart_data["values"].get(group, []),
            )
        )

    fig.update_layout(
        barmode="group",
        title="Spending by Category Across Seasons",
        xaxis_title="Season",
        yaxis_title="Amount ($)",
        height=500,
    )

    return fig


def create_line_chart(chart_data):
    """Create plotly line chart for cumulative spending."""
    import plotly.graph_objects as go

    if not chart_data or "dates" not in chart_data:
        return None

    fig = go.Figure(
        data=go.Scatter(
            x=chart_data["dates"],
            y=chart_data["cumulative"],
            mode="lines+markers",
            fill="tozeroy",
            name="Cumulative Spending",
        )
    )

    fig.update_layout(
        title=chart_data.get("title", "Cumulative Spending Over Time"),
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        height=500,
    )

    return fig


def analyze_credit_card_data(file):
    """Analyze uploaded credit card CSV with seasonal breakdown and visualizations."""
    if file is None:
        return (
            "Please upload a CSV file containing your credit card transactions.",
            None,
            None,
            None,
            None,
        )

    try:
        df = pd.read_csv(file.name)

        # Validate columns
        if "debit" not in df.columns or "description" not in df.columns:
            return (
                "Error: CSV must contain 'debit' and 'description' columns",
                None,
                None,
                None,
                None,
            )

        # Add category column
        df["category"] = df["description"].apply(extract_category_from_description)

        # Calculate overall sustainability score
        score, breakdown = calculate_sustainability_score(df)

        if score is None:
            return f"Error: {breakdown}", None, None, None, None

        total_debit = df["debit"].sum()

        # Generate main report
        report = "# Sustainability Analysis Report\n\n"
        report += f"**Sustainability Score**: {score:.1f}/100\n\n"
        report += f"**Total Spending (Debit)**: ${total_debit:.2f}\n\n"

        # Progress bar
        progress_bars = int(score / 10)
        report += f"Progress: {'█' * progress_bars}{'░' * (10 - progress_bars)}\n\n"

        # Analyze seasonal patterns
        seasonal_dfs, llm_analysis, seasonal_summary = analyze_seasonal_patterns(df)

        report += "## Seasonal Analysis\n\n"
        report += llm_analysis
        report += "\n\n"

        # Category breakdown
        report += "## Category Breakdown\n\n"
        for cat, data in sorted(
            breakdown.items(), key=lambda x: x[1]["amount"], reverse=True
        ):
            pct = (data["amount"] / total_debit) * 100
            report += f"- **{cat.title()}**: ${data['amount']:.2f} ({pct:.1f}%)\n"

        # Generate chart data
        pie_data = generate_pie_chart_data(df)
        pie_chart = create_pie_chart(pie_data)

        seasonal_pie_data = generate_seasonal_pie_chart_data(seasonal_dfs)

        bar_data = generate_bar_chart_data(seasonal_summary)
        bar_chart = create_bar_chart(bar_data)

        line_data = generate_line_chart_data(df)
        line_chart = create_line_chart(line_data)

        return report, pie_chart, bar_chart, line_chart, seasonal_pie_data

    except Exception as e:
        return (
            f"Error processing file: {str(e)}\n\nPlease ensure your CSV has 'date', 'description', 'credit', and 'debit' columns.",
            None,
            None,
            None,
            None,
        )


def identity(inputs: dict) -> dict:
    return {
        "question": (inputs.get("question") or "").strip(),
        "use_web": bool(inputs.get("use_web", False)),
        "region": (inputs.get("region") or "us-en"),
        "safesearch": (inputs.get("safesearch") or "moderate"),
        "timelimit": (inputs.get("timelimit") or None),
        "backend": (inputs.get("backend") or None),
        "max_results": int(inputs.get("max_results") or 20),
    }


id_runnable = RunnableLambda(identity)


def _orchestrate(inputs: dict) -> str:
    text = (inputs.get("question") or "").strip()
    use_web = bool(inputs.get("use_web", False))
    region = inputs.get("region") or "us-en"
    safesearch = inputs.get("safesearch") or "moderate"
    timelimit = inputs.get("timelimit") or None
    backend = inputs.get("backend") or None
    max_results = int(inputs.get("max_results") or 20)

    if not text:
        return (
            "| Intent | Reply |\n"
            "|--------|-------|\n"
            "| Help | Please enter a research topic or upload a credit card CSV. |\n"
        )

    return answer_as_table(
        text,
        region=region,
        max_results=max_results,
        safesearch=safesearch,
        timelimit=timelimit,
        backend=backend,
        force_web=use_web,
    )


core_runnable = RunnableLambda(_orchestrate)

combined = (
    RunnableParallel(prompt=prompt, data=id_runnable).pick("data")
) | core_runnable

_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


with_history = RunnableWithMessageHistory(
    combined,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


def respond(
    message,
    history,
    use_web,
    session_state,
    region,
    safesearch,
    timelimit,
    backend,
    max_results,
):
    text = (message.get("text") if isinstance(message, dict) else message) or ""
    text = text.strip()

    if not text:
        return (
            "| Intent | Reply |\n"
            "|--------|-------|\n"
            "| Help | Please enter a research topic or upload a CSV file. |\n"
        ), session_state

    session_id = session_state.get("session_id")
    if not session_id:
        session_id = f"conv-{uuid.uuid4().hex}"
        session_state["session_id"] = session_id

    try:
        output = with_history.invoke(
            {
                "question": text,
                "use_web": bool(use_web),
                "region": (region or "us-en"),
                "safesearch": (safesearch or "moderate"),
                "timelimit": (timelimit or None),
                "backend": (backend or None),
                "max_results": int(max_results or 20),
            },
            config={"configurable": {"session_id": session_id}},
        )
        return output, session_state
    except Exception as e:
        return (
            f"| Intent | Reply |\n|--------|-------|\n| Error | {str(e)} |\n"
        ), session_state


with gr.Blocks(
    title="Sustainability & Research Assistant", theme=gr.themes.Soft()
) as demo:
    gr.Markdown("# 🌱 Sustainability & Research Assistant")
    gr.Markdown(
        "Upload your credit card CSV to analyze sustainability with seasonal insights, or use the chat for research help."
    )

    session_state = gr.State({"session_id": None})

    with gr.Tab("📊 Sustainability Analysis"):
        gr.Markdown("### Upload Credit Card Transactions")
        gr.Markdown(
            "Your CSV should have columns: `date`, `description`, `credit`, `debit`"
        )
        gr.Markdown(
            "**Note**: Categories are auto-detected from transaction descriptions."
        )

        file_upload = gr.File(label="Upload CSV", file_types=[".csv"])
        analyze_btn = gr.Button("Analyze Sustainability", variant="primary", scale=2)

        with gr.Group():
            sustainability_output = gr.Markdown()

        with gr.Group():
            gr.Markdown("### Visualizations")

            with gr.Row():
                pie_chart = gr.Plot(label="Overall Spending Distribution")
                bar_chart = gr.Plot(label="Seasonal Spending Comparison")

            line_chart = gr.Plot(label="Cumulative Spending Over Time")

        seasonal_data = gr.State({})

        analyze_btn.click(
            fn=analyze_credit_card_data,
            inputs=[file_upload],
            outputs=[
                sustainability_output,
                pie_chart,
                bar_chart,
                line_chart,
                seasonal_data,
            ],
        )

    with gr.Tab("🔍 Research Chat"):
        with gr.Row():
            use_web = gr.Checkbox(
                label="Use web search (academic sources)", value=False
            )
            region = gr.Dropdown(
                choices=["us-en", "wt-wt", "uk-en", "ca-en", "in-en", "de-de", "fr-fr"],
                value="us-en",
                label="Region",
            )
            safesearch = gr.Dropdown(
                choices=["on", "moderate", "off"], value="moderate", label="SafeSearch"
            )
            timelimit = gr.Dropdown(
                choices=[None, "d", "w", "m", "y"], value=None, label="Time limit"
            )
            backend = gr.Dropdown(
                choices=[None, "api", "html", "lite"], value=None, label="DDG backend"
            )
            max_results = gr.Slider(
                minimum=5, maximum=50, value=20, step=1, label="Max results"
            )

        chat = gr.ChatInterface(
            fn=respond,
            additional_inputs=[
                use_web,
                session_state,
                region,
                safesearch,
                timelimit,
                backend,
                max_results,
            ],
            additional_outputs=[session_state],
            type="messages",
            title="Research Chat",
            description="Toggle checkbox for web search and literature review tables, or chat for quick help.",
            save_history=True,
        )

if __name__ == "__main__":
    demo.launch()
