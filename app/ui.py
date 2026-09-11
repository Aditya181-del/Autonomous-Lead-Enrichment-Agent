"""
Streamlit Operational Dashboard for Autonomous Lead Enrichment Agent.

Provides a clean, intuitive web interface for running multi-domain intelligence
enrichment via the baseline pipeline or the experimental LangGraph orchestrator.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import sys
import time
from typing import Optional

import streamlit as st

from app.config import Settings, get_settings
from app.graph import run_graph_batch
from app.main import (
    DEFAULT_DOMAINS,
    enrich_batch,
    save_output_csv,
    save_output_json,
)
from app.models import CompanyIntelligence
from app.scoring import ConfidenceBreakdown
from app.llm import LLMUsage


st.set_page_config(
    page_title="Lead Enrichment Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> tuple[Settings, str]:
    """Render sidebar configuration controls."""
    st.sidebar.title("⚙️ Orchestration Config")
    st.sidebar.markdown("Configure crawler and agent settings.")

    base_settings = get_settings()

    engine = st.sidebar.radio(
        "Orchestration Engine",
        options=["LangGraph Agentic Graph", "Baseline Production Pipeline"],
        index=0,
        help="Select between the LangGraph state machine orchestrator or the baseline pipeline.",
    )

    concurrency = st.sidebar.slider(
        "Max Concurrent Domains",
        min_value=1,
        max_value=5,
        value=base_settings.max_concurrent_domains,
        help="Bounded number of domains processed in parallel.",
    )

    max_pages = st.sidebar.slider(
        "Max Pages Per Domain",
        min_value=1,
        max_value=10,
        value=base_settings.max_pages_per_domain,
        help="Maximum relevant internal candidate pages to scrape.",
    )

    timeout = st.sidebar.number_input(
        "Request Timeout (seconds)",
        min_value=5,
        max_value=60,
        value=base_settings.request_timeout_seconds,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**LLM Model**: `{base_settings.llm_model}`")
    st.sidebar.markdown(f"**Ollama Host**: `{base_settings.ollama_host}`")

    custom_settings = Settings(
        app_env=base_settings.app_env,
        log_level=base_settings.log_level,
        llm_provider=base_settings.llm_provider,
        llm_model=base_settings.llm_model,
        ollama_host=base_settings.ollama_host,
        request_timeout_seconds=int(timeout),
        max_retries=base_settings.max_retries,
        max_pages_per_domain=int(max_pages),
        max_concurrent_domains=int(concurrency),
    )

    return custom_settings, engine


def render_header() -> None:
    """Render main header and description."""
    st.title("Autonomous Lead Enrichment Agent")
    st.markdown(
        "Extract verified **company overview**, **ideal customer profile (ICP)**, "
        "**leadership members**, and **deterministic contact emails** directly from public website evidence."
    )


def main() -> None:
    settings, engine_choice = render_sidebar()
    render_header()

    default_text = "\n".join(DEFAULT_DOMAINS)
    domain_input = st.text_area(
        "Target Company Domains (one per line):",
        value=default_text,
        height=110,
        help="Enter company domains to enrich (e.g. postman.com, supabase.com, vapi.ai).",
    )

    col1, col2 = st.columns([2, 8])
    with col1:
        start_btn = st.button("🚀 Start Enrichment", type="primary", use_container_width=True)

    if start_btn:
        domains = [d.strip() for d in domain_input.splitlines() if d.strip()]

        if not domains:
            st.error("Please enter at least one target domain.")
            return

        st.markdown("---")
        st.subheader("⚡ Live Execution & Progress")

        progress_bar = st.progress(0.0)
        status_container = st.empty()
        log_expander = st.expander("Detailed Live Logs", expanded=True)
        log_placeholder = log_expander.empty()
        log_lines: list[str] = []

        def log_callback(domain: str, msg: str):
            line = f"{time.strftime('%H:%M:%S')} | **{domain}**: {msg}"
            log_lines.append(line)
            log_placeholder.markdown("\n\n".join(log_lines[-10:]))

        start_time = time.perf_counter()

        status_container.info(f"Running enrichment for {len(domains)} domain(s) using **{engine_choice}**...")

        # Run async batch execution in dedicated isolated worker thread with fresh ProactorEventLoop
        def run_async_batch():
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            if "LangGraph" in engine_choice:
                return asyncio.run(
                    run_graph_batch(domains, settings, on_stage_callback=log_callback)
                )
            else:
                return asyncio.run(
                    enrich_batch(domains, settings)
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async_batch)
            results = future.result()

        elapsed = time.perf_counter() - start_time
        progress_bar.progress(1.0)
        status_container.success(f"Enrichment completed in **{elapsed:.2f}s**!")

        # Save results to output/
        save_output_json(results, "output/output.json")
        save_output_csv(results, "output/output.csv")

        # -------------------------------------------------------------------
        # Render Results Dashboard
        # -------------------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Enriched Intelligence Results")

        # Metrics row
        successful = sum(1 for item in results if item[0].crawl_status == "success")
        total_tokens = sum(item[1].total_tokens for item in results)
        avg_confidence = sum(item[0].confidence_score for item in results) / len(results) if results else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Domains Enriched", f"{successful} / {len(results)}")
        m2.metric("Avg Confidence", f"{avg_confidence:.2f}")
        m3.metric("Total Tokens", f"{total_tokens:,}")
        m4.metric("Total Runtime", f"{elapsed:.2f}s")

        # Detailed cards per domain
        for intelligence, usage, breakdown in results:
            status_icon = "🟢" if intelligence.crawl_status == "success" else ("🟡" if intelligence.crawl_status == "partial" else "🔴")
            with st.container():
                st.markdown(f"### {status_icon} `{intelligence.domain}` — Score: **{intelligence.confidence_score:.2f}**")

                c1, c2 = st.columns([6, 4])
                with c1:
                    st.markdown("**Company Overview:**")
                    st.info(intelligence.company_overview or "(None extracted)")

                    st.markdown("**Target Audience / ICP:**")
                    st.write(intelligence.target_audience or "(None extracted)")

                    st.markdown("**Leadership Team:**")
                    if intelligence.leadership:
                        for member in intelligence.leadership:
                            role_text = f" — *{member.role}*" if member.role else ""
                            li_link = f" [🔗 LinkedIn]({member.linkedin_url})" if member.linkedin_url else ""
                            st.markdown(f"- **{member.name}**{role_text}{li_link}")
                    else:
                        st.caption("No leadership members verified in evidence.")

                with c2:
                    st.markdown("**Deterministic Contact Emails:**")
                    if intelligence.contact_points:
                        for cp in intelligence.contact_points:
                            src = f" ([Source]({cp.source_url}))" if cp.source_url else ""
                            st.markdown(f"- ✉️ `{cp.email}`{src}")
                    else:
                        st.caption("No public contact emails found in evidence.")

                    st.markdown("**Verified Evidence Sources:**")
                    for src in intelligence.sources:
                        st.markdown(f"- [🌐 {src}]({src})")

                    if usage.total_tokens > 0:
                        st.caption(f"Telemetry: In={usage.input_tokens:,} | Out={usage.output_tokens:,} | Latency={usage.latency_seconds:.2f}s")

                with st.expander(f"Inspect Raw JSON & Confidence Breakdown for {intelligence.domain}"):
                    st.json(intelligence.model_dump(mode="json"))
                    if breakdown and breakdown.notes:
                        st.markdown("**Confidence Breakdown Notes:**")
                        for note in breakdown.notes:
                            st.markdown(f"- {note}")

                st.markdown("---")

        # Download buttons
        with open("output/output.json", "r", encoding="utf-8") as f:
            json_data = f.read()

        with open("output/output.csv", "r", encoding="utf-8") as f:
            csv_data = f.read()

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                label="📥 Download output.json",
                data=json_data,
                file_name="output.json",
                mime="application/json",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                label="📥 Download output.csv",
                data=csv_data,
                file_name="output.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
