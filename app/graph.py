"""
LangGraph-based agentic orchestration layer for the Lead Enrichment Agent.

This module provides stateful graph execution, observable transitions,
conditional routing, and bounded recovery over the existing modular services.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional, TypedDict
from urllib.parse import urlparse

from pydantic import HttpUrl

from langgraph.graph import END, START, StateGraph

from app.acquisition import EvidenceAcquisitionService, normalize_input_domain
from app.cleaner import clean_page
from app.config import Settings, get_settings
from app.crawler import BrowserCrawler
from app.discovery import normalize_url
from app.evidence import build_llm_evidence
from app.extractor import extract_entities
from app.llm import LLMExtractionResult, LLMUsage, OllamaExtractor
from app.models import (
    CleanedPageEvidence,
    CompanyIntelligence,
    ContactPoint,
    CrawlResult,
    LLMCompanyExtraction,
    PageEvidence,
    TeamMember,
)
from app.scoring import ConfidenceBreakdown, calculate_confidence_score
from app.utils import get_logger
from app.validators import (
    validate_leadership,
    validate_overview,
    validate_sources,
    validate_target_audience,
)


logger = get_logger(__name__)


class EnrichmentState(TypedDict, total=False):
    """
    Explicit, strongly-typed execution state for the enrichment graph.
    """

    domain: str
    clean_domain: str
    crawler: BrowserCrawler
    settings: Settings
    crawl_result: Optional[CrawlResult]
    all_pages: list[PageEvidence]
    acquired_urls: set[str]
    deterministic_emails: list[str]
    email_source_map: dict[str, HttpUrl]
    observed_linkedin_urls: set[str]
    cleaned_pages: list[CleanedPageEvidence]
    llm_evidence_text: str
    llm_extraction: Optional[LLMCompanyExtraction]
    llm_usage: LLMUsage
    validated_sources: list[HttpUrl]
    validated_leadership: list[TeamMember]
    validated_overview: str
    validated_audience: str
    contact_points: list[ContactPoint]
    confidence_score: float
    confidence_breakdown: Optional[ConfidenceBreakdown]
    final_intelligence: Optional[CompanyIntelligence]
    current_stage: str
    errors: list[str]
    recovery_attempts: int
    max_recovery_attempts: int
    status: str
    on_stage_callback: Optional[Callable[[str, str], None]]


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------


def notify_stage(state: EnrichmentState, stage_name: str, message: str) -> None:
    """Helper to update state stage and trigger UI callback if provided."""
    state["current_stage"] = stage_name
    cb = state.get("on_stage_callback")
    if cb and callable(cb):
        try:
            cb(state.get("clean_domain", state.get("domain", "")), f"[{stage_name.upper()}] {message}")
        except Exception:
            pass


async def node_initialize(state: EnrichmentState) -> dict[str, Any]:
    """Initialize enrichment execution state and normalize target domain."""
    raw_domain = state.get("domain", "")
    clean = normalize_input_domain(raw_domain)
    notify_stage(state, "initialize", f"Initialized target domain: {clean}")

    return {
        "clean_domain": clean,
        "all_pages": [],
        "acquired_urls": set(),
        "deterministic_emails": [],
        "email_source_map": {},
        "observed_linkedin_urls": set(),
        "cleaned_pages": [],
        "llm_evidence_text": "",
        "llm_usage": LLMUsage(),
        "errors": [],
        "recovery_attempts": 0,
        "max_recovery_attempts": 1,
        "status": "running",
        "current_stage": "initialize",
    }


async def node_acquire_evidence(state: EnrichmentState) -> dict[str, Any]:
    """Execute browser acquisition using Playwright and EvidenceAcquisitionService."""
    clean_domain = state["clean_domain"]
    crawler = state["crawler"]
    settings = state["settings"]

    notify_stage(state, "acquire_evidence", f"Acquiring homepage and internal pages for {clean_domain}")
    logger.info("Graph: Acquiring evidence for %s", clean_domain)

    context = None
    try:
        context = await crawler.create_context()
        acquisition_service = EvidenceAcquisitionService(crawler, settings)
        crawl_result = await acquisition_service.collect_domain(context, clean_domain)

        all_pages: list[PageEvidence] = []
        if crawl_result.homepage and crawl_result.homepage.success:
            all_pages.append(crawl_result.homepage)
        all_pages.extend(crawl_result.pages)

        acquired_urls = {
            normalize_url(str(p.url))
            for p in all_pages
            if p.url
        }

        return {
            "crawl_result": crawl_result,
            "all_pages": all_pages,
            "acquired_urls": acquired_urls,
            "current_stage": "acquire_evidence",
        }
    except Exception as exc:
        logger.exception("Graph: Error in acquire_evidence for %s: %s", clean_domain, exc)
        return {
            "crawl_result": None,
            "all_pages": [],
            "acquired_urls": set(),
            "errors": state.get("errors", []) + [f"Acquisition error: {exc}"],
            "status": "failed",
            "current_stage": "acquire_evidence",
        }
    finally:
        if context:
            await context.close()


async def node_prepare_evidence(state: EnrichmentState) -> dict[str, Any]:
    """Run dual-path processing: deterministic entity extraction and semantic cleaning."""
    clean_domain = state["clean_domain"]
    all_pages = state.get("all_pages", [])
    notify_stage(state, "prepare_evidence", f"Processing {len(all_pages)} pages (cleaning + entity extraction)")

    deterministic_emails: set[str] = set()
    observed_linkedin_urls: set[str] = set()
    email_source_map: dict[str, HttpUrl] = {}
    cleaned_pages: list[CleanedPageEvidence] = []

    for page in all_pages:
        # Deterministic regex extraction
        entities = extract_entities(page)
        for item in entities.emails:
            deterministic_emails.add(item.email)
            if item.email not in email_source_map:
                email_source_map[item.email] = item.source_url

        for item in entities.linkedin_profiles:
            norm_linkedin = normalize_url(str(item.linkedin_url))
            if norm_linkedin:
                observed_linkedin_urls.add(norm_linkedin)

        # Semantic content cleaning
        cleaned = clean_page(page)
        if cleaned.content.strip():
            cleaned_pages.append(cleaned)

    llm_evidence_text = build_llm_evidence(cleaned_pages)

    return {
        "deterministic_emails": sorted(deterministic_emails),
        "email_source_map": email_source_map,
        "observed_linkedin_urls": observed_linkedin_urls,
        "cleaned_pages": cleaned_pages,
        "llm_evidence_text": llm_evidence_text,
        "current_stage": "prepare_evidence",
    }


def route_evidence_check(state: EnrichmentState) -> str:
    """
    Conditional router: evaluate if acquired evidence is sufficient for LLM extraction.
    """
    crawl_result = state.get("crawl_result")
    all_pages = state.get("all_pages", [])
    cleaned_pages = state.get("cleaned_pages", [])
    recovery_attempts = state.get("recovery_attempts", 0)
    max_recovery_attempts = state.get("max_recovery_attempts", 1)

    has_valid_homepage = crawl_result and crawl_result.homepage and crawl_result.homepage.success
    has_content = any(len(p.content.strip()) > 30 for p in cleaned_pages)

    if has_valid_homepage and has_content:
        logger.info("Graph Router: Evidence sufficient for %s -> extract_semantics", state["clean_domain"])
        return "extract_semantics"

    if recovery_attempts < max_recovery_attempts:
        logger.warning(
            "Graph Router: Evidence insufficient for %s (attempt %d/%d) -> recover_evidence",
            state["clean_domain"],
            recovery_attempts + 1,
            max_recovery_attempts,
        )
        return "recover_evidence"

    logger.warning("Graph Router: Evidence recovery exhausted for %s -> score_result", state["clean_domain"])
    return "score_result"


async def node_recover_evidence(state: EnrichmentState) -> dict[str, Any]:
    """Bounded recovery node: attempts fallback navigation on http or additional URLs."""
    clean_domain = state["clean_domain"]
    crawler = state["crawler"]
    settings = state["settings"]
    attempts = state.get("recovery_attempts", 0) + 1

    notify_stage(state, "recover_evidence", f"Attempting bounded evidence recovery (attempt {attempts})")
    logger.info("Graph: Running recovery attempt %d for %s", attempts, clean_domain)

    context = None
    try:
        context = await crawler.create_context()
        fallback_url = f"http://{clean_domain}"
        snapshot = await crawler.fetch_page(context, fallback_url)
        evidence = crawler.snapshot_to_evidence(snapshot)

        existing_pages = list(state.get("all_pages", []))
        acquired_urls = set(state.get("acquired_urls", set()))

        if evidence.success:
            existing_pages.append(evidence)
            if evidence.url:
                acquired_urls.add(normalize_url(str(evidence.url)))

        return {
            "all_pages": existing_pages,
            "acquired_urls": acquired_urls,
            "recovery_attempts": attempts,
            "current_stage": "recover_evidence",
        }
    except Exception as exc:
        logger.exception("Graph: Recovery failed for %s: %s", clean_domain, exc)
        return {
            "recovery_attempts": attempts,
            "errors": state.get("errors", []) + [f"Recovery error: {exc}"],
            "current_stage": "recover_evidence",
        }
    finally:
        if context:
            await context.close()


async def node_extract_semantics(state: EnrichmentState) -> dict[str, Any]:
    """Invoke Ollama with structured LLMCompanyExtraction schema and zero temperature."""
    clean_domain = state["clean_domain"]
    evidence_text = state.get("llm_evidence_text", "")
    deterministic_emails = state.get("deterministic_emails", [])
    settings = state["settings"]

    notify_stage(state, "extract_semantics", f"Running Ollama structured extraction ({settings.llm_model})")
    logger.info("Graph: Running semantic extraction for %s", clean_domain)

    try:
        llm_extractor = OllamaExtractor(settings)
        llm_result: LLMExtractionResult = await llm_extractor.extract(
            domain=clean_domain,
            evidence_text=evidence_text,
            deterministic_emails=deterministic_emails,
        )

        return {
            "llm_extraction": llm_result.data,
            "llm_usage": llm_result.usage,
            "current_stage": "extract_semantics",
        }
    except Exception as exc:
        logger.exception("Graph: LLM extraction error for %s: %s", clean_domain, exc)
        return {
            "llm_extraction": LLMCompanyExtraction(),
            "llm_usage": LLMUsage(),
            "errors": state.get("errors", []) + [f"LLM extraction error: {exc}"],
            "current_stage": "extract_semantics",
        }


async def node_validate_result(state: EnrichmentState) -> dict[str, Any]:
    """Validate source URLs, leadership names/roles, and filter ungrounded entities."""
    raw_extraction = state.get("llm_extraction") or LLMCompanyExtraction()
    acquired_urls = state.get("acquired_urls", set())
    observed_linkedin_urls = state.get("observed_linkedin_urls", set())
    all_pages = state.get("all_pages", [])
    deterministic_emails = state.get("deterministic_emails", [])
    email_source_map = state.get("email_source_map", {})

    notify_stage(state, "validate_result", "Validating source URLs and leadership grounding")

    validated_sources = validate_sources(raw_extraction.sources, acquired_urls)
    if not validated_sources and acquired_urls:
        for page in all_pages:
            if page.url:
                validated_sources.append(page.url)

    validated_leadership = validate_leadership(
        raw_extraction.leadership,
        acquired_urls,
        observed_linkedin_urls,
    )

    validated_overview = validate_overview(raw_extraction.company_overview)
    validated_audience = validate_target_audience(raw_extraction.target_audience)

    contact_points = [
        ContactPoint(
            email=email,
            source_url=email_source_map.get(email),
        )
        for email in deterministic_emails
    ]

    return {
        "validated_sources": validated_sources,
        "validated_leadership": validated_leadership,
        "validated_overview": validated_overview,
        "validated_audience": validated_audience,
        "contact_points": contact_points,
        "current_stage": "validate_result",
    }


async def node_score_result(state: EnrichmentState) -> dict[str, Any]:
    """Calculate deterministic heuristic confidence score and breakdown."""
    clean_domain = state["clean_domain"]
    crawl_result = state.get("crawl_result")

    notify_stage(state, "score_result", "Calculating deterministic heuristic confidence score")

    intelligence = CompanyIntelligence(
        domain=clean_domain,
        company_overview=state.get("validated_overview", ""),
        target_audience=state.get("validated_audience", ""),
        leadership=state.get("validated_leadership", []),
        contact_points=state.get("contact_points", []),
        confidence_score=0.0,
        sources=state.get("validated_sources", []),
        crawl_status="success" if not state.get("errors") else "partial",
    )

    confidence_score, breakdown = calculate_confidence_score(intelligence, crawl_result)
    intelligence.confidence_score = confidence_score

    return {
        "confidence_score": confidence_score,
        "confidence_breakdown": breakdown,
        "final_intelligence": intelligence,
        "current_stage": "score_result",
    }


async def node_finalize(state: EnrichmentState) -> dict[str, Any]:
    """Finalize CompanyIntelligence, set execution status, and log summary."""
    clean_domain = state["clean_domain"]
    intelligence = state.get("final_intelligence")

    if intelligence is None:
        intelligence = CompanyIntelligence(
            domain=clean_domain,
            company_overview="",
            target_audience="",
            leadership=[],
            contact_points=[],
            confidence_score=0.0,
            sources=[],
            crawl_status="failed",
            error_message="; ".join(state.get("errors", ["Failed to produce intelligence"])),
        )

    status = "success" if intelligence.confidence_score >= 0.50 else ("partial" if intelligence.confidence_score > 0 else "failed")
    intelligence.crawl_status = status
    if state.get("errors"):
        intelligence.error_message = "; ".join(state.get("errors", []))

    notify_stage(state, "finalize", f"Enrichment completed with status: {status.upper()} (score: {intelligence.confidence_score:.2f})")

    return {
        "final_intelligence": intelligence,
        "status": status,
        "current_stage": "finalize",
    }


# ---------------------------------------------------------------------------
# Graph Builder & Runners
# ---------------------------------------------------------------------------


def build_enrichment_graph() -> Any:
    """
    Construct the LangGraph StateGraph workflow for lead enrichment.
    """
    workflow = StateGraph(EnrichmentState)

    # Register nodes
    workflow.add_node("initialize", node_initialize)
    workflow.add_node("acquire_evidence", node_acquire_evidence)
    workflow.add_node("prepare_evidence", node_prepare_evidence)
    workflow.add_node("recover_evidence", node_recover_evidence)
    workflow.add_node("extract_semantics", node_extract_semantics)
    workflow.add_node("validate_result", node_validate_result)
    workflow.add_node("score_result", node_score_result)
    workflow.add_node("finalize", node_finalize)

    # Define edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "acquire_evidence")
    workflow.add_edge("acquire_evidence", "prepare_evidence")

    # Conditional router after prepare_evidence
    workflow.add_conditional_edges(
        "prepare_evidence",
        route_evidence_check,
        {
            "extract_semantics": "extract_semantics",
            "recover_evidence": "recover_evidence",
            "score_result": "score_result",
        },
    )

    # Recovery loops back to prepare_evidence
    workflow.add_edge("recover_evidence", "prepare_evidence")

    # Linear downstream pipeline
    workflow.add_edge("extract_semantics", "validate_result")
    workflow.add_edge("validate_result", "score_result")
    workflow.add_edge("score_result", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


async def run_graph_enrichment(
    domain: str,
    crawler: BrowserCrawler,
    settings: Settings,
    on_stage_callback: Optional[Callable[[str, str], None]] = None,
) -> tuple[CompanyIntelligence, LLMUsage, ConfidenceBreakdown]:
    """
    Execute lead enrichment for a single domain using the compiled LangGraph workflow.
    """
    graph = build_enrichment_graph()

    initial_state: EnrichmentState = {
        "domain": domain,
        "clean_domain": normalize_input_domain(domain),
        "crawler": crawler,
        "settings": settings,
        "on_stage_callback": on_stage_callback,
        "errors": [],
        "recovery_attempts": 0,
        "max_recovery_attempts": 1,
    }

    final_state = await graph.ainvoke(initial_state)

    intelligence = final_state.get(
        "final_intelligence",
        CompanyIntelligence(domain=domain, crawl_status="failed"),
    )
    usage = final_state.get("llm_usage", LLMUsage())
    breakdown = final_state.get("confidence_breakdown", ConfidenceBreakdown())

    return intelligence, usage, breakdown


async def run_graph_batch(
    domains: list[str],
    settings: Settings,
    on_stage_callback: Optional[Callable[[str, str], None]] = None,
) -> list[tuple[CompanyIntelligence, LLMUsage, ConfidenceBreakdown]]:
    """
    Execute lead enrichment across multiple domains concurrently using LangGraph.
    """
    crawler = BrowserCrawler(settings)
    await crawler.start()

    semaphore = asyncio.Semaphore(settings.max_concurrent_domains)

    async def _worker(domain: str):
        async with semaphore:
            return await run_graph_enrichment(
                domain=domain,
                crawler=crawler,
                settings=settings,
                on_stage_callback=on_stage_callback,
            )

    try:
        tasks = [_worker(d) for d in domains]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results
    finally:
        await crawler.close()
