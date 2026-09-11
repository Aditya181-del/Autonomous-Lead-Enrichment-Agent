from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

from pydantic import HttpUrl

from app.discovery import normalize_url
from app.models import (
    ContactPoint,
    ExtractedEntities,
    LLMCompanyExtraction,
    LLMSourceCitation,
    LLMTeamMember,
    PageEvidence,
    TeamMember,
)
from app.utils import get_logger


logger = get_logger(__name__)


# Generic non-person names that LLMs might mistakenly output as people.
INVALID_NAME_PATTERNS = re.compile(
    r"^(ceo|cto|cfo|cpo|coo|founder|co-founder|president|vice president|vp|director|"
    r"executive|leadership|team|board|unknown|n/a|none|na|anonymous|company|admin|staff|"
    r"engineering|sales|marketing|support|contact|info|help|author|admin)$",
    re.IGNORECASE,
)


def is_valid_person_name(name: Optional[str]) -> bool:
    """
    Validate whether a name string represents a plausible personal name.

    Rejects:
    - Empty strings or whitespace
    - Generic titles used as names (e.g. 'CEO', 'Founder')
    - Placeholders ('Unknown', 'N/A')
    - Obvious single-character or numeric tokens
    """

    if not name:
        return False

    cleaned = name.strip()

    if len(cleaned) < 2:
        return False

    if INVALID_NAME_PATTERNS.match(cleaned):
        return False

    # Names should contain at least one letter and not consist entirely of numbers/symbols.
    if not any(char.isalpha() for char in cleaned):
        return False

    return True


def normalize_source_url_str(url: str | HttpUrl | LLMSourceCitation | None) -> str:
    """
    Normalize a source URL for exact comparison against evidence URLs.
    """

    if not url:
        return ""

    if isinstance(url, LLMSourceCitation):
        return normalize_url(str(url.url))

    return normalize_url(str(url))


def validate_sources(
    llm_sources: list[LLMSourceCitation | HttpUrl | str],
    acquired_urls: set[str],
) -> list[HttpUrl]:
    """
    Validate and filter LLM-emitted source URLs against actual acquired evidence URLs.

    Every source URL emitted by the LLM must match an acquired page URL
    after normalization. Unsupported/fabricated URLs are rejected.
    """

    validated: list[HttpUrl] = []
    seen: set[str] = set()

    for source in llm_sources:
        url_obj: HttpUrl
        if isinstance(source, LLMSourceCitation):
            url_obj = source.url
        elif isinstance(source, HttpUrl):
            url_obj = source
        else:
            url_obj = HttpUrl(str(source))

        normalized_str = normalize_source_url_str(url_obj)

        if not normalized_str:
            continue

        if normalized_str in seen:
            continue

        # Check if this source URL matches any of the acquired evidence URLs.
        if normalized_str in acquired_urls:
            validated.append(url_obj)
            seen.add(normalized_str)
        else:
            logger.warning(
                "Rejected unsupported LLM source URL: %s (not in acquired evidence)",
                source,
            )

    return validated


def validate_leadership(
    llm_leadership: list[LLMTeamMember],
    acquired_urls: set[str],
    observed_linkedin_urls: set[str],
) -> list[TeamMember]:
    """
    Validate leadership members extracted by the LLM.

    Enforces:
    1. Valid, non-generic person name.
    2. Source URL must be in acquired evidence (if provided).
    3. LinkedIn URL must be in the set of observed LinkedIn profiles from evidence.
       If an unobserved LinkedIn URL is provided, it is cleared to prevent hallucination.
    """

    validated_members: list[TeamMember] = []
    seen_names: set[str] = set()

    for item in llm_leadership:
        if not is_valid_person_name(item.name):
            logger.warning(
                "Rejected invalid leadership name: '%s'",
                item.name,
            )
            continue

        normalized_name_key = item.name.strip().casefold()
        if normalized_name_key in seen_names:
            continue

        # Validate source URL
        validated_source: Optional[HttpUrl] = None
        if item.source_url:
            norm_source = normalize_source_url_str(item.source_url)
            if norm_source in acquired_urls:
                validated_source = item.source_url
            else:
                logger.warning(
                    "Cleared unverified source_url for leader %s: %s",
                    item.name,
                    item.source_url,
                )

        # Validate LinkedIn URL
        validated_linkedin: Optional[HttpUrl] = None
        if item.linkedin_url:
            norm_linkedin = normalize_url(str(item.linkedin_url))
            if norm_linkedin in observed_linkedin_urls:
                validated_linkedin = item.linkedin_url
            else:
                logger.warning(
                    "Cleared unobserved LinkedIn URL for leader %s: %s",
                    item.name,
                    item.linkedin_url,
                )

        validated_members.append(
            TeamMember(
                name=item.name.strip(),
                role=item.role.strip() if item.role else None,
                linkedin_url=validated_linkedin,
                source_url=validated_source,
            )
        )
        seen_names.add(normalized_name_key)

    return validated_members


def validate_overview(overview: str) -> str:
    """
    Sanitize and validate company overview text.
    """

    if not overview:
        return ""

    return overview.strip()


def validate_target_audience(target_audience: str) -> str:
    """
    Sanitize and validate target audience / ICP text.
    """

    if not target_audience:
        return ""

    return target_audience.strip()
