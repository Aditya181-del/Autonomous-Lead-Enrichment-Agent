from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from app.models import CleanedPageEvidence, PageEvidence
from app.utils import get_logger


logger = get_logger(__name__)


# Structural elements that generally do not contribute useful
# company intelligence.
REMOVE_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "canvas",
    "template",
    "path",
}


# Conservative keyword patterns for common UI/boilerplate containers.
# We intentionally avoid removing generic "header" elements because
# headers can contain useful product/company positioning.
BOILERPLATE_PATTERN = re.compile(
    r"""
    cookie
    |consent
    |gdpr
    |privacy-banner
    |cookie-banner
    |newsletter
    |subscribe
    |popup
    |modal
    |breadcrumb
    |social-share
    """,
    re.IGNORECASE | re.VERBOSE,
)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving line boundaries.
    """

    lines = []

    for raw_line in text.splitlines():
        line = re.sub(
            r"\s+",
            " ",
            raw_line,
        ).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def deduplicate_lines(text: str) -> str:
    """
    Remove repeated lines while preserving their original order.
    """

    seen: set[str] = set()
    unique_lines: list[str] = []

    for line in text.splitlines():

        normalized = line.strip()

        if not normalized:
            continue

        # Case-insensitive duplicate comparison.
        key = normalized.casefold()

        if key in seen:
            continue

        seen.add(key)
        unique_lines.append(normalized)

    return "\n".join(unique_lines)


def _remove_hidden_elements(soup: BeautifulSoup) -> None:
    """
    Remove elements that are explicitly hidden from users.
    """

    for tag in soup.find_all(
        attrs={"aria-hidden": "true"}
    ):
        tag.decompose()

    for tag in soup.find_all(
        attrs={"hidden": True}
    ):
        tag.decompose()

    for tag in soup.find_all(
        style=True
    ):
        style = str(tag.get("style", "")).lower()

        if (
            "display:none" in style.replace(" ", "")
            or "visibility:hidden" in style.replace(" ", "")
        ):
            tag.decompose()


def _remove_unwanted_tags(soup: BeautifulSoup) -> None:
    """
    Remove known non-content HTML elements.
    """

    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()


def _remove_boilerplate_containers(
    soup: BeautifulSoup,
) -> None:
    """
    Remove elements whose id/class strongly suggests UI boilerplate.
    """

    for tag in soup.find_all(
        attrs={"id": True}
    ):
        element_id = str(tag.get("id", ""))

        if BOILERPLATE_PATTERN.search(element_id):
            tag.decompose()

    # Classes can contain multiple values, so inspect the joined
    # representation.
    for tag in soup.find_all(
        attrs={"class": True}
    ):
        class_names = " ".join(
            str(value)
            for value in tag.get("class", [])
        )

        if BOILERPLATE_PATTERN.search(class_names):
            tag.decompose()


def _remove_navigation_and_footer(
    soup: BeautifulSoup,
) -> None:
    """
    Remove explicit navigation/footer regions.

    We preserve generic header elements because they can contain
    important company positioning or product messaging.
    """

    for tag_name in (
        "nav",
        "footer",
    ):
        for tag in soup.find_all(tag_name):
            tag.decompose()


def _choose_content_root(
    soup: BeautifulSoup,
) -> Tag:
    """
    Prefer semantic main/article content when available.

    Fall back to the body when semantic containers are absent.
    """

    main = soup.find("main")

    if isinstance(main, Tag):
        return main

    article = soup.find("article")

    if isinstance(article, Tag):
        return article

    body = soup.find("body")

    if isinstance(body, Tag):
        return body

    # BeautifulSoup documents normally contain html/body, but return
    # the soup itself as a defensive fallback.
    return soup


def _extract_text(root: Tag) -> str:
    """
    Extract visible textual content from the cleaned DOM.
    """

    # Insert line boundaries around common semantic blocks so that
    # headings and paragraphs don't collapse together.
    for tag_name in (
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "li",
        "blockquote",
        "dt",
        "dd",
        "tr",
    ):
        for tag in root.find_all(tag_name):
            if tag.string:
                continue

            text = tag.get_text(
                " ",
                strip=True,
            )

            if text:
                tag.clear()
                tag.append(text)

    return root.get_text(
        separator="\n",
        strip=True,
    )


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using a simple character-based heuristic.

    This is intentionally an approximation for telemetry only.
    Actual token counts will come from the LLM provider later.
    """

    if not text:
        return 0

    return max(
        1,
        len(text) // 4,
    )


def clean_page(
    page: PageEvidence,
) -> CleanedPageEvidence:
    """
    Transform raw webpage evidence into LLM-ready text.
    """

    original_html = page.html or page.content

    original_characters = len(original_html)

    original_lines = len(
        original_html.splitlines()
    )

    logger.info(
        "Cleaning page: %s",
        page.url,
    )

    if not original_html.strip():
        logger.warning(
            "No HTML content available for %s",
            page.url,
        )

        return CleanedPageEvidence(
            url=page.url,
            title=page.title,
            content="",
            original_characters=0,
            cleaned_characters=0,
            original_lines=0,
            cleaned_lines=0,
            estimated_input_tokens=0,
            reduction_ratio=0.0,
            links=page.links,
            source_status_code=page.status_code,
        )

    soup = BeautifulSoup(
        original_html,
        "lxml",
    )

    _remove_unwanted_tags(soup)

    _remove_hidden_elements(soup)

    _remove_boilerplate_containers(soup)

    _remove_navigation_and_footer(soup)

    root = _choose_content_root(soup)

    extracted_text = _extract_text(root)

    normalized_text = normalize_whitespace(
        extracted_text
    )

    cleaned_text = deduplicate_lines(
        normalized_text
    )

    cleaned_characters = len(cleaned_text)

    cleaned_lines = len(
        cleaned_text.splitlines()
    )

    estimated_tokens = estimate_tokens(
        cleaned_text
    )

    if original_characters > 0:
        reduction_ratio = max(
            0.0,
            min(
                1.0,
                1 - (
                    cleaned_characters
                    / original_characters
                ),
            ),
        )
    else:
        reduction_ratio = 0.0

    logger.info(
        "Cleaned %s | chars=%d -> %d | reduction=%.1f%%",
        page.url,
        original_characters,
        cleaned_characters,
        reduction_ratio * 100,
    )

    return CleanedPageEvidence(
        url=page.url,
        title=page.title,
        content=cleaned_text,
        original_characters=original_characters,
        cleaned_characters=cleaned_characters,
        original_lines=original_lines,
        cleaned_lines=cleaned_lines,
        estimated_input_tokens=estimated_tokens,
        reduction_ratio=reduction_ratio,
        links=page.links,
        source_status_code=page.status_code,
    )