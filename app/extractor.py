from __future__ import annotations

import re
from urllib.parse import unquote, urlparse, urlunparse

from app.models import (
    ExtractedEmail,
    ExtractedEntities,
    ExtractedLinkedInProfile,
    PageEvidence,
)
from app.utils import get_logger


logger = get_logger(__name__)


# Conservative email pattern.
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)


# We specifically want personal LinkedIn profiles,
# not company pages.
LINKEDIN_PROFILE_PATTERN = re.compile(
    r"^https?://(?:www\.)?linkedin\.com/in/[^/?#]+/?$",
    re.IGNORECASE,
)


def normalize_email(email: str) -> str:
    """
    Normalize a discovered email address.
    """

    return unquote(
        email.strip()
    ).lower()


def normalize_linkedin_url(url: str) -> str:
    """
    Normalize a LinkedIn profile URL.

    Removes:
    - query parameters
    - fragments
    - trailing slash
    """

    parsed = urlparse(
        url.strip()
    )

    normalized = parsed._replace(
        query="",
        fragment="",
    )

    result = urlunparse(
        normalized
    )

    return result.rstrip("/")


def is_valid_email(email: str) -> bool:
    """
    Validate whether a string resembles a usable email address.
    """

    return bool(
        EMAIL_PATTERN.fullmatch(
            email.strip()
        )
    )


def is_linkedin_profile(url: str) -> bool:
    """
    Determine whether a URL points to an individual LinkedIn profile.
    """

    normalized = normalize_linkedin_url(
        url
    )

    return bool(
        LINKEDIN_PROFILE_PATTERN.match(
            normalized
        )
    )


def extract_emails_from_text(
    text: str,
) -> list[str]:
    """
    Extract unique email addresses from text.
    """

    if not text:
        return []

    matches = EMAIL_PATTERN.findall(
        text
    )

    emails: set[str] = set()

    for match in matches:

        email = normalize_email(
            match
        )

        if is_valid_email(email):
            emails.add(email)

    return sorted(emails)


def extract_emails_from_links(
    links: list[str],
) -> list[str]:
    """
    Extract email addresses from mailto: links.
    """

    emails: set[str] = set()

    for link in links:

        if not link:
            continue

        if not link.lower().startswith(
            "mailto:"
        ):
            continue

        raw_value = link[
            len("mailto:"):
        ]

        # mailto:user@example.com?subject=Hello
        email_part = raw_value.split(
            "?",
            1,
        )[0]

        email = normalize_email(
            email_part
        )

        if is_valid_email(email):
            emails.add(email)

    return sorted(emails)


def extract_linkedin_from_links(
    links: list[str],
) -> list[str]:
    """
    Extract unique individual LinkedIn profile URLs from href values.
    """

    profiles: set[str] = set()

    for link in links:

        if not link:
            continue

        normalized = normalize_linkedin_url(
            link
        )

        if is_linkedin_profile(
            normalized
        ):
            profiles.add(
                normalized
            )

    return sorted(profiles)


def extract_entities(
    page: PageEvidence,
) -> ExtractedEntities:
    """
    Extract deterministic entities from one PageEvidence object.

    Email extraction considers both:
    - visible/rendered text
    - mailto: hrefs

    LinkedIn extraction considers:
    - href values from the rendered page
    """

    logger.info(
        "Extracting deterministic entities from %s",
        page.url,
    )

    text_emails = extract_emails_from_text(
        page.content
    )

    linked_emails = extract_emails_from_links(
        page.links
    )

    all_emails = sorted(
        set(text_emails)
        | set(linked_emails)
    )

    linkedin_profiles = (
        extract_linkedin_from_links(
            page.links
        )
    )

    email_entities = [
        ExtractedEmail(
            email=email,
            source_url=page.url,
        )
        for email in all_emails
    ]

    linkedin_entities = [
        ExtractedLinkedInProfile(
            linkedin_url=profile,
            source_url=page.url,
        )
        for profile in linkedin_profiles
    ]

    logger.info(
        "Extracted entities | emails=%d | linkedin_profiles=%d",
        len(email_entities),
        len(linkedin_entities),
    )

    return ExtractedEntities(
        source_url=page.url,
        emails=email_entities,
        linkedin_profiles=linkedin_entities,
    )