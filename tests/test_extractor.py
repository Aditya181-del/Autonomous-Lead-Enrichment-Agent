from app.extractor import (
    extract_emails_from_links,
    extract_emails_from_text,
    extract_entities,
    extract_linkedin_from_links,
    is_linkedin_profile,
    normalize_email,
    normalize_linkedin_url,
)
from app.models import PageEvidence


def test_normalize_email():
    assert (
        normalize_email(
            " INFO@EXAMPLE.COM "
        )
        == "info@example.com"
    )


def test_extract_emails_from_text():
    text = """
    Contact us at info@example.com.
    Sales: sales@example.com.
    """

    result = extract_emails_from_text(
        text
    )

    assert result == [
        "info@example.com",
        "sales@example.com",
    ]


def test_extract_emails_from_links():
    links = [
        "mailto:info@example.com",
        "mailto:sales@example.com?subject=Hello",
        "https://example.com/contact",
        "mailto:INFO@EXAMPLE.COM",
    ]

    result = extract_emails_from_links(
        links
    )

    assert result == [
        "info@example.com",
        "sales@example.com",
    ]


def test_normalize_linkedin_url():
    url = (
        "https://www.linkedin.com/in/jane-doe/"
        "?trk=profile"
        "#about"
    )

    assert (
        normalize_linkedin_url(url)
        == "https://www.linkedin.com/in/jane-doe"
    )


def test_is_linkedin_profile():
    assert is_linkedin_profile(
        "https://www.linkedin.com/in/jane-doe"
    )

    assert not is_linkedin_profile(
        "https://www.linkedin.com/company/example"
    )


def test_extract_linkedin_from_links():
    links = [
        "https://www.linkedin.com/in/jane-doe/",
        "https://linkedin.com/in/john-smith?trk=abc",
        "https://www.linkedin.com/company/example/",
    ]

    result = extract_linkedin_from_links(
        links
    )

    assert result == [
        "https://linkedin.com/in/john-smith",
        "https://www.linkedin.com/in/jane-doe",
    ]


def test_extract_entities():
    page = PageEvidence(
        url="https://example.com/contact",
        content="""
            Contact sales@example.com.
            General enquiries: info@example.com.
        """,
        links=[
            "mailto:sales@example.com",
            "https://www.linkedin.com/in/jane-doe/",
        ],
        status_code=200,
        success=True,
    )

    result = extract_entities(
        page
    )

    emails = [
        item.email
        for item in result.emails
    ]

    linkedin = [
        str(item.linkedin_url)
        for item in result.linkedin_profiles
    ]

    assert emails == [
        "info@example.com",
        "sales@example.com",
    ]

    assert linkedin == [
        "https://www.linkedin.com/in/jane-doe"
    ]