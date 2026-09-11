import pytest
from pydantic import ValidationError

from app.llm import OllamaExtractor
from app.models import LLMCompanyExtraction


def test_build_context_contains_required_sections():
    extractor = object.__new__(
        OllamaExtractor
    )

    context = extractor.build_context(
        domain="example.com",
        evidence_text=(
            "SOURCE 1\n"
            "URL: https://example.com/about\n"
            "CONTENT:\n"
            "Example builds software."
        ),
        deterministic_emails=[
            "info@example.com",
            "sales@example.com",
        ],
    )

    assert "TARGET DOMAIN:" in context
    assert "example.com" in context
    assert (
        "AUTHORITATIVE CONTACT EMAILS"
        in context
    )
    assert "info@example.com" in context
    assert "sales@example.com" in context
    assert (
        "<<<BEGIN UNTRUSTED FIRST-PARTY WEBSITE EVIDENCE>>>"
        in context
    )
    assert "Example builds software." in context


def test_llm_company_extraction_missing_optional_fields():
    # Only overview and target audience provided
    raw_json = '{"company_overview": "Co builds tools.", "target_audience": "Devs"}'
    parsed = LLMCompanyExtraction.model_validate_json(raw_json)
    assert parsed.company_overview == "Co builds tools."
    assert parsed.target_audience == "Devs"
    assert parsed.leadership == []
    assert parsed.sources == []


def test_llm_company_extraction_empty_json():
    parsed = LLMCompanyExtraction.model_validate_json("{}")
    assert parsed.company_overview == ""
    assert parsed.target_audience == ""
    assert parsed.leadership == []
    assert parsed.sources == []


def test_llm_company_extraction_malformed_json():
    with pytest.raises(ValueError):
        LLMCompanyExtraction.model_validate_json("{invalid json")