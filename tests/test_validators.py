from pydantic import HttpUrl
import pytest

from app.models import LLMSourceCitation, LLMTeamMember
from app.validators import (
    is_valid_person_name,
    validate_leadership,
    validate_overview,
    validate_sources,
    validate_target_audience,
)


def test_is_valid_person_name():
    assert is_valid_person_name("Abhinav Asthana")
    assert is_valid_person_name("Jane Doe")
    assert is_valid_person_name("John O'Connor")

    # Generic titles & placeholders must be rejected
    assert not is_valid_person_name("CEO")
    assert not is_valid_person_name("Founder")
    assert not is_valid_person_name("Unknown")
    assert not is_valid_person_name("N/A")
    assert not is_valid_person_name("12345")
    assert not is_valid_person_name("")
    assert not is_valid_person_name(None)
    assert not is_valid_person_name("a")


def test_validate_sources_filters_unsupported_urls():
    acquired_urls = {
        "https://postman.com/about",
        "https://postman.com/company/team",
    }

    llm_sources = [
        HttpUrl("https://postman.com/about"),
        HttpUrl("https://postman.com/about/"),  # trailing slash variant
        HttpUrl("https://fake-hallucinated-source.com/page"),
    ]

    validated = validate_sources(llm_sources, acquired_urls)
    assert len(validated) == 1
    assert str(validated[0]).rstrip("/") == "https://postman.com/about"


def test_validate_sources_with_structured_citations():
    acquired_urls = {
        "https://supabase.com/about",
        "https://supabase.com/pricing",
    }

    citations = [
        LLMSourceCitation(
            url=HttpUrl("https://supabase.com/about"),
            claim="Hosted Postgres databases",
        ),
        LLMSourceCitation(
            url=HttpUrl("https://supabase.com/pricing"),
            claim="Free tier available",
        ),
        LLMSourceCitation(
            url=HttpUrl("https://hallucinated-blog.com/post"),
            claim="Not in evidence",
        ),
    ]

    validated = validate_sources(citations, acquired_urls)
    assert len(validated) == 2
    assert str(validated[0]).rstrip("/") == "https://supabase.com/about"
    assert str(validated[1]).rstrip("/") == "https://supabase.com/pricing"


def test_validate_leadership_grounding():
    acquired_urls = {
        "https://postman.com/about",
    }
    observed_linkedin = {
        "https://www.linkedin.com/in/abhinav-asthana",
    }

    llm_leadership = [
        LLMTeamMember(
            name="Abhinav Asthana",
            role="CEO & Founder",
            linkedin_url=HttpUrl("https://www.linkedin.com/in/abhinav-asthana"),
            source_url=HttpUrl("https://postman.com/about"),
        ),
        LLMTeamMember(
            name="CEO",  # generic title as name
            role="Chief Executive Officer",
            linkedin_url=None,
            source_url=None,
        ),
        LLMTeamMember(
            name="Jane Hallucinated",
            role="CTO",
            linkedin_url=HttpUrl("https://www.linkedin.com/in/invented-profile"),
            source_url=HttpUrl("https://not-in-evidence.com/about"),
        ),
    ]

    validated = validate_leadership(
        llm_leadership,
        acquired_urls,
        observed_linkedin,
    )

    assert len(validated) == 2  # "CEO" rejected

    # Abhinav Asthana has verified name, role, linkedin, source
    assert validated[0].name == "Abhinav Asthana"
    assert validated[0].role == "CEO & Founder"
    assert str(validated[0].linkedin_url) == "https://www.linkedin.com/in/abhinav-asthana"
    assert str(validated[0].source_url) == "https://postman.com/about"

    # Jane Hallucinated has unobserved linkedin cleared and invalid source cleared
    assert validated[1].name == "Jane Hallucinated"
    assert validated[1].linkedin_url is None
    assert validated[1].source_url is None


def test_validate_overview_and_audience():
    assert validate_overview("  Postman is an API platform.  ") == "Postman is an API platform."
    assert validate_overview("") == ""
    assert validate_target_audience("  Developers and engineering teams.  ") == "Developers and engineering teams."
