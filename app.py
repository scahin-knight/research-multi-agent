import streamlit as st
from src.pipeline.pipelinie import run_research_pipeline

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🔎", layout="wide")

st.title("🔎 Multi-Agent Research Assistant")
st.caption("Search → Read → Write → Critique — powered by a chain of LLM agents")

with st.form("research_form"):
    topic = st.text_input("Enter a research topic", placeholder="e.g. Impact of AI on UAV navigation systems")
    submitted = st.form_submit_button("Run Research", use_container_width=True)

if "state" not in st.session_state:
    st.session_state.state = None

if submitted:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        try:
            with st.spinner("Running the research pipeline... this can take a minute"):
                st.session_state.state = run_research_pipeline(topic)
            st.success("Pipeline finished.")
        except Exception as e:
            st.session_state.state = None
            st.error(f"Pipeline failed: {e}")

state = st.session_state.state

if state:
    tab_search, tab_scrape, tab_report, tab_critic = st.tabs(
        ["🔍 Search Results", "📄 Scraped Content", "📝 Final Report", "🧐 Critic Feedback"]
    )

    with tab_search:
        st.markdown(state.get("search_results", "No search results."))

    with tab_scrape:
        st.markdown(state.get("scraped_content", "No scraped content."))

    with tab_report:
        report = state.get("report", "No report generated.")
        st.markdown(report)
        st.download_button(
            "⬇️ Download Report",
            data=report if isinstance(report, str) else str(report),
            file_name=f"{topic.replace(' ', '_')}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with tab_critic:
        st.markdown(state.get("feedback", "No critic feedback."))
else:
    st.info("Enter a topic above and click **Run Research** to get started.")