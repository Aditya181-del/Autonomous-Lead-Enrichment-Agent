from pydantic import ValidationError
import pytest

from app.models import (
    CompanyIntelligence,
    ContactPoint,
    CrawlResult,
    LLMCompanyExtraction,
    LLMTeamMember,
    PageEvidence,
    TeamMember,
)


def test_company_intelligence_defaults():
    result = CompanyIntelligence(
        domain="example.com"
    )

    assert result.domain == "example.com"
    assert result.company_overview == ""
    assert result.target_audience == ""
    assert result.contact_points == []
    assert result.leadership == []
    assert result.confidence_score == 0.0
    assert result.sources == []
    assert result.crawl_status == "success"
    assert result.error_message is None


def test_llm_company_extraction_valid():
    raw_json = """
    {
        "company_overview": "Example Inc is an API testing platform. It helps engineering teams collaborate.",
        "target_audience": "Software engineers and QA teams.",
        "leadership": [
            {
                "name": "Jane Doe",
                "role": "CEO",
                "linkedin_url": "https://linkedin.com/in/jane-doe",
                "source_url": "https://example.com/about"
            }
        ],
        "sources": [
            "https://example.com/about"
        ]
    }
    """
    model = LLMCompanyExtraction.model_validate_json(raw_json)
    assert len(model.leadership) == 1
    assert model.leadership[0].name == "Jane Doe"
    assert str(model.sources[0].url).rstrip("/") == "https://example.com/about"


def test_llm_company_extraction_structured_source_objects():
    raw_json = """
    {
        "company_overview": "Supabase is an open-source Firebase alternative. It provides Postgres databases.",
        "target_audience": "Developers and engineering teams.",
        "leadership": [],
        "sources": [
            {
                "url": "https://supabase.com/about",
                "claim": "Company provides hosted Postgres databases"
            },
            {
                "url": "https://supabase.com/solutions/startups",
                "claim": "Targeted towards developers and startups"
            }
        ]
    }
    """
    model = LLMCompanyExtraction.model_validate_json(raw_json)
    assert len(model.sources) == 2
    assert str(model.sources[0].url).rstrip("/") == "https://supabase.com/about"
    assert model.sources[0].claim == "Company provides hosted Postgres databases"
    assert str(model.sources[1].url).rstrip("/") == "https://supabase.com/solutions/startups"
    assert model.sources[1].claim == "Targeted towards developers and startups"


def test_llm_company_extraction_schema_separation():
    # LLMCompanyExtraction does not require or own domain or confidence_score
    schema = LLMCompanyExtraction.model_json_schema()
    properties = schema.get("properties", {})
    assert "company_overview" in properties
    assert "target_audience" in properties
    assert "leadership" in properties
    assert "sources" in properties
    assert "domain" not in properties
    assert "confidence_score" not in properties
    assert "contact_points" not in properties


def test_page_evidence_accepts_mailto_links():
    result = PageEvidence(
        url="https://postman.com/company/contact-us",
        links=[
            "https://postman.com/about",
            "mailto:info@postman.com",
            "mailto:sales@postman.com",
        ],
    )

    assert len(result.links) == 3
    assert "mailto:info@postman.com" in result.links


def test_crawl_result_defaults():
    result = CrawlResult(
        domain="example.com"
    )

    assert result.domain == "example.com"
    assert result.homepage is None
    assert result.pages == []
    assert result.discovered_url_count == 0
    assert result.internal_candidate_count == 0
    assert result.selected_url_count == 0
    assert result.failed_url_count == 0