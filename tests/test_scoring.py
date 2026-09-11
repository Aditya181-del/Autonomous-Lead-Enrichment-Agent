from pydantic import HttpUrl
import pytest

from app.models import (
    CompanyIntelligence,
    ContactPoint,
    CrawlResult,
    PageEvidence,
    TeamMember,
)
from app.scoring import calculate_confidence_score


def test_confidence_score_empty():
    intelligence = CompanyIntelligence(domain="example.com")
    score, breakdown = calculate_confidence_score(intelligence)
    assert score == 0.0
    assert breakdown.total_score == 0.0
    assert breakdown.overview_score == 0.0
    assert breakdown.leadership_score == 0.0


def test_confidence_score_partial_evidence():
    crawl_result = CrawlResult(
        domain="example.com",
        homepage=PageEvidence(
            url="https://example.com",
            status_code=200,
            success=True,
        ),
        selected_url_count=2,
        pages=[
            PageEvidence(
                url="https://example.com/about",
                status_code=200,
                success=True,
            )
        ],
    )

    intelligence = CompanyIntelligence(
        domain="example.com",
        company_overview="Example Inc provides software solutions for modern cloud architecture.",
        target_audience="DevOps engineers and cloud architects.",
        sources=[HttpUrl("https://example.com/about")],
    )

    score, breakdown = calculate_confidence_score(intelligence, crawl_result)
    assert 0.40 <= score <= 0.80
    assert breakdown.overview_score == 0.20
    assert breakdown.target_audience_score == 0.15
    assert breakdown.crawl_coverage_score > 0.15


def test_confidence_score_full_evidence():
    crawl_result = CrawlResult(
        domain="example.com",
        homepage=PageEvidence(
            url="https://example.com",
            status_code=200,
            success=True,
        ),
        selected_url_count=2,
        pages=[
            PageEvidence(
                url="https://example.com/about",
                status_code=200,
                success=True,
            ),
            PageEvidence(
                url="https://example.com/contact",
                status_code=200,
                success=True,
            ),
        ],
    )

    intelligence = CompanyIntelligence(
        domain="example.com",
        company_overview="Example Inc builds developer tooling and API platforms. It empowers teams globally.",
        target_audience="API engineers, developers, and enterprise QA teams worldwide.",
        leadership=[
            TeamMember(
                name="Abhinav Asthana",
                role="CEO & Founder",
                linkedin_url=HttpUrl("https://www.linkedin.com/in/abhinav-asthana"),
                source_url=HttpUrl("https://example.com/about"),
            )
        ],
        contact_points=[
            ContactPoint(
                email="contact@example.com",
                source_url=HttpUrl("https://example.com/contact"),
            )
        ],
        sources=[
            HttpUrl("https://example.com/about"),
            HttpUrl("https://example.com/contact"),
        ],
    )

    score, breakdown = calculate_confidence_score(intelligence, crawl_result)
    assert score >= 0.90
    assert score <= 1.0
    assert breakdown.leadership_score == 0.20
    assert breakdown.contact_score == 0.10
    assert breakdown.source_attribution_score == 0.10
