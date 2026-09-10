from __future__ import annotations

from urllib.parse import urlparse, urlunparse

from app.utils import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# URL relevance configuration
# ---------------------------------------------------------------------------
#
# These keywords represent page intent rather than exact page names.
# For example:
#   /about
#   /about-us
#   /about-postman
#
# can all contain the semantic signal "about".
#
# Scores are intentionally hierarchical. We use the strongest matching
# signal rather than blindly adding every keyword together.
# ---------------------------------------------------------------------------

KEYWORD_SCORES = {
    # Highest-value company intelligence pages
    "team": 100,
    "leadership": 100,
    "founder": 100,
    "founders": 100,
    "people": 95,

    # Core company information
    "about": 90,
    "company": 80,

    # Contact / commercial information
    "contact": 75,
    "pricing": 60,

    # Useful supporting information
    "customer": 45,
    "customers": 45,
    "solution": 40,
    "solutions": 40,
    "partner": 35,
    "partners": 35,

    # Lower-value pages
    "career": 20,
    "careers": 20,
    "product": 15,

    # Lowest-value informational content
    "blog": 5,
    "news": 5,
    "event": 5,
    "events": 5,
    "webinar": 5,
    "webinars": 5,
    "resource": 5,
    "resources": 5,
    "docs": 3,
    "documentation": 3,
}


# Pages that are normally not useful for our company-intelligence objective.
EXCLUDED_KEYWORDS = {
    "privacy",
    "terms",
    "cookie",
    "cookies",
    "login",
    "signup",
    "sign-in",
    "signin",
    "auth",
    "authentication",
    "download",
}


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """
    Normalize a URL for crawling.

    Current normalization:
    - removes URL fragments (#section)
    - removes a trailing slash
    - preserves query parameters

    Examples:
        https://example.com/about/#team
        -> https://example.com/about

        https://example.com/about/
        -> https://example.com/about
    """

    if not url:
        return ""

    url = url.strip()

    if not url:
        return ""

    parsed = urlparse(url)

    # We only want normal HTTP(S) URLs in the discovery pipeline.
    if parsed.scheme.lower() not in {"http", "https"}:
        return ""

    normalized = parsed._replace(
        fragment="",
    )

    normalized_url = urlunparse(normalized)

    # Keep the root URL as-is logically rather than turning it into
    # an empty string.
    if normalized_url.endswith("/") and parsed.path not in {"", "/"}:
        normalized_url = normalized_url.rstrip("/")

    elif normalized_url.endswith("/") and parsed.path == "/":
        normalized_url = normalized_url.rstrip("/")

    return normalized_url


# ---------------------------------------------------------------------------
# Hostname / domain handling
# ---------------------------------------------------------------------------


def get_hostname(url: str) -> str:
    """
    Return a normalized hostname.

    Examples:
        https://www.postman.com/about
        -> postman.com

        https://api.postman.com
        -> api.postman.com
    """

    if not url:
        return ""

    try:
        hostname = urlparse(url).hostname or ""
    except ValueError:
        return ""

    hostname = hostname.lower().strip()

    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname


def is_internal_url(
    url: str,
    root_domain: str,
) -> bool:
    """
    Determine whether a URL belongs to the target company domain.

    Examples for root_domain = "postman.com":

        postman.com              -> True
        www.postman.com          -> True
        api.postman.com          -> True
        evilpostman.com          -> False
        buildwithfern.com       -> False
    """

    url_domain = get_hostname(url)

    if not url_domain or not root_domain:
        return False

    root_domain = root_domain.lower().strip()

    if root_domain.startswith("www."):
        root_domain = root_domain[4:]

    return (
        url_domain == root_domain
        or url_domain.endswith(f".{root_domain}")
    )


# ---------------------------------------------------------------------------
# URL relevance scoring
# ---------------------------------------------------------------------------


def _tokenize_path(path: str) -> list[str]:
    """
    Convert a URL path into normalized semantic segments.

    Example:
        /company/about-postman
        ->
        ["company", "about-postman"]
    """

    return [
        segment
        for segment in path.split("/")
        if segment
    ]


def _contains_excluded_keyword(
    segments: list[str],
) -> bool:
    """
    Return True if any path segment exactly matches an excluded keyword.
    """

    return any(
        segment in EXCLUDED_KEYWORDS
        for segment in segments
    )


def _best_keyword_score(
    segments: list[str],
) -> int:
    """
    Find the strongest semantic keyword signal in the URL.

    Matching is substring-based so URLs like:

        /about-us
        /about-postman
        /contact-sales
        /leadership-team

    can still be classified correctly.
    """

    best_score = 0

    for segment in segments:
        for keyword, score in KEYWORD_SCORES.items():
            if keyword in segment:
                best_score = max(best_score, score)

    return best_score


def score_url(url: str) -> int:
    """
    Assign a relevance score based on the semantic intent of a URL.

    The most specific/high-value page intent gets the primary score.
    Generic parent paths such as /company do not overpower a more
    specific page type such as /careers or /press-media.

    Examples:

        /team
            -> 100

        /company/team
            -> 110

        /company/about-postman
            -> 100

        /company/contact-sales
            -> 80

        /company/careers
            -> 25

        /company/press-media
            -> 5
    """

    if not url:
        return 0

    parsed = urlparse(url)

    path = parsed.path.lower().strip("/")

    if not path:
        # Homepage is already crawled separately.
        return 0

    segments = [
        segment
        for segment in path.split("/")
        if segment
    ]

    if not segments:
        return 0

    # ---------------------------------------------------------------
    # Step 1: explicit exclusions
    # ---------------------------------------------------------------

    if _contains_excluded_keyword(segments):
        return -20

    # ---------------------------------------------------------------
    # Step 2: determine the strongest specific intent
    # ---------------------------------------------------------------
    #
    # "company" is intentionally excluded from this primary scoring
    # because it is a generic parent/container concept.
    #
    # Example:
    #   /company/careers
    #
    # should be primarily classified as "careers", not "company".
    # ---------------------------------------------------------------

    specific_scores = {
        "team": 100,
        "leadership": 100,
        "founder": 100,
        "founders": 100,
        "people": 95,

        "about": 90,

        "contact": 75,
        "pricing": 60,

        "customer": 45,
        "customers": 45,

        "solution": 40,
        "solutions": 40,

        "partner": 35,
        "partners": 35,

        "career": 20,
        "careers": 20,

        "product": 15,

        "blog": 5,
        "news": 5,
        "event": 5,
        "events": 5,
        "webinar": 5,
        "webinars": 5,
        "resource": 5,
        "resources": 5,
        "docs": 3,
        "documentation": 3,

        "press": 5,
        "media": 5,
    }

    best_score = 0
    matched_keyword = None

    for segment in segments:
        for keyword, score in specific_scores.items():
            if keyword in segment and score > best_score:
                best_score = score
                matched_keyword = keyword

    # No meaningful semantic signal.
    if best_score == 0:
        return 0

    # ---------------------------------------------------------------
    # Step 3: contextual bonuses
    # ---------------------------------------------------------------

    normalized_path = " ".join(segments)

    # Company/about is a particularly useful source.
    if (
        "company" in segments
        and "about" in normalized_path
    ):
        best_score += 10

    # Company/team or company/leadership is especially valuable.
    if (
        "company" in segments
        and any(
            keyword in normalized_path
            for keyword in (
                "team",
                "leadership",
                "founder",
                "founders",
                "people",
            )
        )
    ):
        best_score += 10

    # Company/contact is also a strong first-party contact source.
    if (
        "company" in segments
        and "contact" in normalized_path
    ):
        best_score += 5

    # Sales contact pages are useful for lead enrichment.
    if (
        matched_keyword == "contact"
        and "sales" in normalized_path
    ):
        best_score += 5

    # A direct /about or /team path should be slightly stronger than
    # the equivalent nested page only when the nested page gains
    # meaningful company context.
    #
    # We deliberately do not add generic "company" points.
    return best_score


# ---------------------------------------------------------------------------
# URL ranking
# ---------------------------------------------------------------------------


def rank_urls(
    urls: list[str],
    root_domain: str,
    max_urls: int = 6,
) -> list[tuple[str, int]]:
    """
    Normalize, filter, deduplicate, score, and rank URLs.

    Returns:
        A list of (url, score) tuples sorted from most relevant
        to least relevant.
    """

    if max_urls <= 0:
        return []

    candidates: set[str] = set()

    for raw_url in urls:
        normalized = normalize_url(raw_url)

        if not normalized:
            continue

        if not is_internal_url(
            normalized,
            root_domain,
        ):
            continue

        candidates.add(normalized)

    ranked = [
        (url, score_url(url))
        for url in candidates
    ]

    # Highest score first.
    # URL is used as a deterministic tie-breaker.
    ranked.sort(
        key=lambda item: (
            -item[1],
            item[0],
        )
    )

    logger.info(
        "Discovered %d URLs | %d unique internal candidates | selecting top %d",
        len(urls),
        len(ranked),
        min(max_urls, len(ranked)),
    )

    return ranked[:max_urls]