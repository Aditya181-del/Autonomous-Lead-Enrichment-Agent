import pytest

from app.config import Settings
from app.graph import (
    EnrichmentState,
    build_enrichment_graph,
    node_finalize,
    node_initialize,
    route_evidence_check,
)
from app.models import (
    CleanedPageEvidence,
    CompanyIntelligence,
    CrawlResult,
    PageEvidence,
)


def test_build_enrichment_graph_compiles():
    """Verify that the LangGraph StateGraph compiles and contains all nodes."""
    graph = build_enrichment_graph()
    assert graph is not None

    # Inspect compiled graph nodes
    node_keys = graph.nodes.keys()
    assert "initialize" in node_keys
    assert "acquire_evidence" in node_keys
    assert "prepare_evidence" in node_keys
    assert "recover_evidence" in node_keys
    assert "extract_semantics" in node_keys
    assert "validate_result" in node_keys
    assert "score_result" in node_keys
    assert "finalize" in node_keys


def test_route_evidence_check_sufficient():
    """Route to extract_semantics when valid homepage and content exist."""
    state: EnrichmentState = {
        "clean_domain": "postman.com",
        "crawl_result": CrawlResult(
            domain="postman.com",
            homepage=PageEvidence(
                url="https://postman.com",
                status_code=200,
                success=True,
            ),
        ),
        "cleaned_pages": [
            CleanedPageEvidence(
                url="https://postman.com/about",
                content="Postman is an API platform for building and collaborating on APIs worldwide.",
            )
        ],
        "recovery_attempts": 0,
        "max_recovery_attempts": 1,
    }

    destination = route_evidence_check(state)
    assert destination == "extract_semantics"


def test_route_evidence_check_recovery():
    """Route to recover_evidence when homepage or content is missing and recovery attempts remain."""
    state: EnrichmentState = {
        "clean_domain": "incomplete-site.com",
        "crawl_result": None,
        "cleaned_pages": [],
        "recovery_attempts": 0,
        "max_recovery_attempts": 1,
    }

    destination = route_evidence_check(state)
    assert destination == "recover_evidence"


def test_route_evidence_check_exhausted():
    """Route to score_result when recovery attempts have reached the maximum bound."""
    state: EnrichmentState = {
        "clean_domain": "failed-site.com",
        "crawl_result": None,
        "cleaned_pages": [],
        "recovery_attempts": 1,
        "max_recovery_attempts": 1,
    }

    destination = route_evidence_check(state)
    assert destination == "score_result"


@pytest.mark.asyncio
async def test_node_initialize():
    """Verify domain normalization during state initialization."""
    state: EnrichmentState = {
        "domain": "https://www.supabase.com/",
    }

    result = await node_initialize(state)
    assert result["clean_domain"] == "supabase.com"
    assert result["status"] == "running"
    assert result["recovery_attempts"] == 0


@pytest.mark.asyncio
async def test_node_finalize():
    """Verify final state assembly and status assignment."""
    intelligence = CompanyIntelligence(
        domain="vapi.ai",
        company_overview="Vapi is a voice AI agent platform.",
        target_audience="Developers.",
        confidence_score=0.90,
    )

    state: EnrichmentState = {
        "clean_domain": "vapi.ai",
        "final_intelligence": intelligence,
        "errors": [],
    }

    result = await node_finalize(state)
    final_intel = result["final_intelligence"]
    assert final_intel.domain == "vapi.ai"
    assert result["status"] == "success"
    assert final_intel.crawl_status == "success"


@pytest.mark.asyncio
async def test_node_recover_evidence_success():
    """Verify bounded recovery updates all_pages, acquired_urls, and enables source validation."""
    from unittest.mock import AsyncMock, MagicMock
    from pydantic import HttpUrl
    from app.crawler import PageSnapshot
    from app.graph import node_recover_evidence, node_validate_result
    from app.models import LLMCompanyExtraction

    mock_crawler = MagicMock()
    mock_context = MagicMock()
    mock_crawler.create_context = AsyncMock(return_value=mock_context)
    mock_context.close = AsyncMock()

    mock_snapshot = PageSnapshot(
        url="http://recovered-domain.com",
        title="Recovered Homepage",
        html="<html><body>Recovered content about company</body></html>",
        text="Recovered content about company",
        links=[],
        status_code=200,
        error=None,
    )
    mock_crawler.fetch_page = AsyncMock(return_value=mock_snapshot)
    mock_crawler.snapshot_to_evidence = lambda snap: PageEvidence(
        url=snap.url,
        title=snap.title,
        content=snap.text,
        status_code=snap.status_code,
        success=True,
    )

    state: EnrichmentState = {
        "clean_domain": "recovered-domain.com",
        "crawler": mock_crawler,
        "settings": Settings(),
        "all_pages": [],
        "acquired_urls": set(),
        "recovery_attempts": 0,
        "max_recovery_attempts": 1,
    }

    recovery_result = await node_recover_evidence(state)
    assert recovery_result["recovery_attempts"] == 1
    assert len(recovery_result["all_pages"]) == 1
    assert str(recovery_result["all_pages"][0].url) == "http://recovered-domain.com/"
    assert "http://recovered-domain.com" in recovery_result["acquired_urls"]

    # Now verify validation node recognizes the recovered page URL as a source
    validation_state: EnrichmentState = {
        **state,
        **recovery_result,
        "llm_extraction": LLMCompanyExtraction(
            sources=[HttpUrl("http://recovered-domain.com")],
            company_overview="Recovered overview",
        ),
    }

    val_result = await node_validate_result(validation_state)
    assert len(val_result["validated_sources"]) == 1
    assert str(val_result["validated_sources"][0]).rstrip("/") == "http://recovered-domain.com"

