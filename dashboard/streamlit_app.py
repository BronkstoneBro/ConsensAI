"""ConsensAI Dashboard - Streamlit application for visualizing consensus metrics."""

import httpx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_TIMEOUT = 60.0


def get_api_client() -> httpx.Client:
    """Create HTTP client for API calls."""
    return httpx.Client(base_url=API_BASE_URL, timeout=DEFAULT_TIMEOUT)


# --- API Functions ---


def fetch_metrics() -> dict | None:
    """Fetch aggregated metrics from API."""
    try:
        with get_api_client() as client:
            response = client.get("/metrics")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        st.error(f"Failed to fetch metrics: {e}")
        return None


def fetch_benchmark_list() -> list[dict] | None:
    """Fetch list of benchmark runs."""
    try:
        with get_api_client() as client:
            response = client.get("/benchmark/list")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return []
        st.error(f"Failed to fetch benchmarks: {e}")
        return None
    except httpx.HTTPError as e:
        st.error(f"Failed to fetch benchmarks: {e}")
        return None


def fetch_benchmark(benchmark_id: str) -> dict | None:
    """Fetch specific benchmark results."""
    try:
        with get_api_client() as client:
            response = client.get(f"/benchmark/{benchmark_id}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        st.error(f"Failed to fetch benchmark {benchmark_id}: {e}")
        return None


def fetch_datasets() -> dict | None:
    """Fetch available datasets."""
    try:
        with get_api_client() as client:
            response = client.get("/datasets")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        st.error(f"Failed to fetch datasets: {e}")
        return None


def run_consensus(question: str, models: list[str] | None = None) -> dict | None:
    """Run consensus analysis on a question."""
    try:
        with get_api_client() as client:
            payload = {"question": question}
            if models:
                payload["models"] = models
            response = client.post("/consensus", json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        st.error(f"Consensus request failed: {e}")
        return None


def run_benchmark(dataset: str, models: list[str] | None = None, limit: int = 10) -> dict | None:
    """Run a benchmark."""
    try:
        with get_api_client() as client:
            payload = {"dataset": dataset, "limit": limit}
            if models:
                payload["models"] = models
            response = client.post("/benchmark", json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        st.error(f"Benchmark request failed: {e}")
        return None


# --- Chart Components ---


def render_accuracy_chart(model_stats: dict[str, dict]) -> None:
    """Render accuracy bar chart by model."""
    if not model_stats:
        st.info("No model statistics available yet. Run a benchmark first.")
        return

    models = list(model_stats.keys())
    accuracies = [stats.get("accuracy", 0) * 100 for stats in model_stats.values()]

    fig = px.bar(
        x=models,
        y=accuracies,
        labels={"x": "Model", "y": "Accuracy (%)"},
        title="Model Accuracy Comparison",
        color=accuracies,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        showlegend=False,
        yaxis_range=[0, 100],
        coloraxis_showscale=False,
    )
    fig.update_traces(
        texttemplate="%{y:.1f}%",
        textposition="outside",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_latency_chart(model_stats: dict[str, dict]) -> None:
    """Render latency bar chart by model."""
    if not model_stats:
        st.info("No latency data available yet. Run a benchmark first.")
        return

    models = list(model_stats.keys())
    latencies = [stats.get("avg_latency_ms", 0) for stats in model_stats.values()]

    fig = px.bar(
        x=models,
        y=latencies,
        labels={"x": "Model", "y": "Average Latency (ms)"},
        title="Model Latency Comparison",
        color=latencies,
        color_continuous_scale="Reds",
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
    )
    fig.update_traces(
        texttemplate="%{y:.0f}ms",
        textposition="outside",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_consensus_pie(consensus_stats: dict[str, float]) -> None:
    """Render consensus distribution pie chart."""
    if not consensus_stats:
        st.info("No consensus statistics available yet.")
        return

    labels = ["Unanimous", "Majority", "Disagreement"]
    values = [
        consensus_stats.get("unanimous_rate", 0) * 100,
        consensus_stats.get("majority_rate", 0) * 100,
        consensus_stats.get("disagreement_rate", 0) * 100,
    ]

    # Skip if all zeros
    if sum(values) == 0:
        st.info("No consensus data recorded yet. Run a benchmark first.")
        return

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=["#2ecc71", "#f39c12", "#e74c3c"],
                textinfo="label+percent",
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Consensus Distribution",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_combined_model_chart(model_stats: dict[str, dict]) -> None:
    """Render combined accuracy and latency chart."""
    if not model_stats:
        st.info("No model statistics available.")
        return

    models = list(model_stats.keys())
    accuracies = [stats.get("accuracy", 0) * 100 for stats in model_stats.values()]
    latencies = [stats.get("avg_latency_ms", 0) for stats in model_stats.values()]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Accuracy (%)", "Latency (ms)"),
        horizontal_spacing=0.15,
    )

    fig.add_trace(
        go.Bar(x=models, y=accuracies, marker_color="#3498db", name="Accuracy"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=models, y=latencies, marker_color="#e74c3c", name="Latency"),
        row=1,
        col=2,
    )

    fig.update_layout(height=400, showlegend=False, title_text="Model Performance Overview")
    fig.update_yaxes(range=[0, 100], row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)


# --- Page Components ---


def render_overview_page() -> None:
    """Render main overview page with metrics."""
    st.header("Overview")

    metrics = fetch_metrics()
    if not metrics:
        st.warning("Could not load metrics. Is the API server running?")
        st.code("uvicorn src.api.main:app --reload --port 8000", language="bash")
        return

    # Key metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Requests", metrics.get("total_requests", 0))
    with col2:
        model_count = len(metrics.get("models", {}))
        st.metric("Models Tracked", model_count)
    with col3:
        consensus = metrics.get("consensus", {})
        unanimous_rate = consensus.get("unanimous_rate", 0) * 100
        st.metric("Unanimous Rate", f"{unanimous_rate:.1f}%")

    st.divider()

    # Charts
    model_stats = metrics.get("models", {})
    consensus_stats = metrics.get("consensus", {})

    col1, col2 = st.columns(2)
    with col1:
        render_accuracy_chart(model_stats)
    with col2:
        render_latency_chart(model_stats)

    st.divider()
    render_consensus_pie(consensus_stats)


def render_benchmark_page() -> None:
    """Render benchmark history and runner page."""
    st.header("Benchmarks")

    tab1, tab2 = st.tabs(["Run Benchmark", "History"])

    with tab1:
        render_benchmark_runner()

    with tab2:
        render_benchmark_history()


def render_benchmark_runner() -> None:
    """Render benchmark runner form."""
    st.subheader("Run New Benchmark")

    datasets_info = fetch_datasets()
    available_datasets = datasets_info.get("datasets", []) if datasets_info else []

    if not available_datasets:
        available_datasets = ["arithmetic", "algebra", "logic"]

    col1, col2 = st.columns(2)
    with col1:
        dataset = st.selectbox("Dataset", available_datasets)
    with col2:
        limit = st.number_input("Task Limit", min_value=1, max_value=100, value=10)

    if st.button("Run Benchmark", type="primary"):
        with st.spinner("Running benchmark..."):
            result = run_benchmark(dataset, limit=limit)
            if result:
                st.success(f"Benchmark completed: {result.get('benchmark_id')}")
                render_benchmark_result(result)


def render_benchmark_history() -> None:
    """Render list of past benchmarks."""
    st.subheader("Benchmark History")

    benchmarks = fetch_benchmark_list()
    if benchmarks is None:
        return

    if not benchmarks:
        st.info("No benchmark runs yet. Run a benchmark to see results here.")
        return

    for bench in benchmarks[:10]:  # Show last 10
        with st.expander(f"{bench.get('benchmark_id')} - {bench.get('dataset')} ({bench.get('status')})"):
            detailed = fetch_benchmark(bench.get("benchmark_id"))
            if detailed:
                render_benchmark_result(detailed)


def render_benchmark_result(result: dict) -> None:
    """Render detailed benchmark result."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", result.get("status", "unknown"))
    with col2:
        st.metric("Dataset", result.get("dataset", ""))
    with col3:
        consensus_stats = result.get("consensus_stats", {})
        total = sum([
            consensus_stats.get("unanimous_agreement", 0),
            consensus_stats.get("majority_agreement", 0),
            consensus_stats.get("disagreement", 0),
        ])
        st.metric("Total Tasks", total)

    # Model results table
    results = result.get("results", {})
    if results:
        st.subheader("Model Performance")
        table_data = []
        for model, stats in results.items():
            table_data.append({
                "Model": model,
                "Accuracy": f"{stats.get('accuracy', 0) * 100:.1f}%",
                "Avg Latency": f"{stats.get('avg_latency_ms', 0):.0f}ms",
                "Correct": stats.get("correct", 0),
                "Incorrect": stats.get("incorrect", 0),
                "Errors": stats.get("errors", 0),
            })
        st.table(table_data)

    # Consensus stats
    consensus_stats = result.get("consensus_stats", {})
    if consensus_stats:
        st.subheader("Consensus Distribution")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unanimous", consensus_stats.get("unanimous_agreement", 0))
        with col2:
            st.metric("Majority", consensus_stats.get("majority_agreement", 0))
        with col3:
            st.metric("Disagreement", consensus_stats.get("disagreement", 0))


def render_interactive_page() -> None:
    """Render interactive consensus test page."""
    st.header("Interactive Test")
    st.markdown("Test the consensus system with your own questions.")

    # Question input
    question = st.text_area(
        "Enter a math question",
        placeholder="Example: What is 15% of 80?",
        height=100,
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        run_button = st.button("Get Consensus", type="primary", disabled=not question)

    if run_button and question:
        with st.spinner("Querying models..."):
            result = run_consensus(question)

        if result:
            render_consensus_result(result)


def render_consensus_result(result: dict) -> None:
    """Render consensus analysis result."""
    consensus = result.get("consensus", {})
    models = result.get("models", {})
    metadata = result.get("metadata", {})

    # Main consensus result
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        answer = consensus.get("answer")
        st.metric("Consensus Answer", answer if answer is not None else "N/A")
    with col2:
        confidence = consensus.get("confidence", 0)
        st.metric("Confidence", f"{confidence * 100:.0f}%")
    with col3:
        agreement = consensus.get("agreement", "unknown")
        color = {
            "unanimous": ":green[Unanimous]",
            "majority": ":orange[Majority]",
            "split": ":red[Split]",
            "disagreement": ":red[Disagreement]",
        }.get(agreement, agreement)
        st.markdown(f"**Agreement:** {color}")

    # Judge info
    if metadata.get("judge_used"):
        st.info(f"Judge model was used: {metadata.get('judge_model')} ({metadata.get('judge_latency_ms', 0):.0f}ms)")

    # Model details
    st.subheader("Model Responses")
    for model_name, model_result in models.items():
        with st.expander(f"{model_name} - Answer: {model_result.get('answer')}"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Latency", f"{model_result.get('latency_ms', 0):.0f}ms")
            with col2:
                tokens = model_result.get("tokens_completion", 0)
                st.metric("Tokens", tokens)

            if model_result.get("steps"):
                st.markdown("**Solution steps:**")
                for i, step in enumerate(model_result.get("steps", []), 1):
                    st.markdown(f"{i}. {step}")

            if model_result.get("error"):
                st.error(f"Error: {model_result.get('error')}")


def render_sidebar() -> str:
    """Render sidebar navigation and return selected page."""
    with st.sidebar:
        st.title("ConsensAI")
        st.caption("Multi-LLM Consensus System")

        st.divider()

        page = st.radio(
            "Navigation",
            ["Overview", "Benchmarks", "Interactive Test"],
            label_visibility="collapsed",
        )

        st.divider()

        # API status
        st.caption("API Status")
        try:
            with get_api_client() as client:
                response = client.get("/metrics", timeout=2.0)
                if response.status_code == 200:
                    st.success("Connected", icon="✓")
                else:
                    st.warning("API Error", icon="!")
        except Exception:
            st.error("Disconnected", icon="✗")

        st.divider()
        st.caption("v0.1.0")

    return page


def main() -> None:
    """Main entry point for Streamlit app."""
    st.set_page_config(
        page_title="ConsensAI Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    page = render_sidebar()

    if page == "Overview":
        render_overview_page()
    elif page == "Benchmarks":
        render_benchmark_page()
    elif page == "Interactive Test":
        render_interactive_page()


if __name__ == "__main__":
    main()
