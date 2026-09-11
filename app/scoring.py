"""
Deterministic, explainable confidence scoring module.

IMPORTANT ARCHITECTURAL NOTE:
The confidence score calculated here is an evidence-based heuristic metric
reflecting the breadth, completeness, and verification of collected signals.
It is NOT a statistically calibrated probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.models import (
    CompanyIntelligence,
    CrawlResult,
)
from app.utils import get_logger


logger = get_logger(__name__)


@dataclass
class ConfidenceBreakdown:
    """
    Detailed explainability breakdown for a calculated confidence score.
    """

    crawl_coverage_score: float = 0.0
    overview_score: float = 0.0
    target_audience_score: float = 0.0
    leadership_score: float = 0.0
    contact_score: float = 0.0
    source_attribution_score: float = 0.0
    total_score: float = 0.0
    notes: list[str] = field(default_factory=list)


def calculate_confidence_score(
    intelligence: CompanyIntelligence,
    crawl_result: Optional[CrawlResult] = None,
) -> tuple[float, ConfidenceBreakdown]:
    """
    Calculate an explainable, deterministic heuristic confidence score (0.0 to 1.0)
    for an enriched company intelligence result.

    Scoring Weights (Total 1.00):
    1. Crawl & Homepage Foundation (0.25)
       - Homepage 200 OK: +0.15
       - Successful page acquisition ratio: +0.10
    2. Company Overview Grounding (0.20)
       - Overview present & non-empty: +0.15
       - Substantive overview length (>= 30 chars): +0.05
    3. Target Audience / ICP Grounding (0.15)
       - ICP present & substantive (>= 15 chars): +0.15
    4. Leadership Evidence (0.20)
       - At least one validated person: +0.10
       - Verified roles present: +0.05
       - Validated source or LinkedIn URL on leaders: +0.05
    5. Contact Information Evidence (0.10)
       - Authoritative deterministic emails found: +0.10
    6. Source Attribution & Traceability (0.10)
       - Validated source URLs present: +0.10

    Returns:
        tuple[float, ConfidenceBreakdown]:
            Normalized confidence score (0.0 <= score <= 1.0) and detailed breakdown.
    """

    breakdown = ConfidenceBreakdown()
    notes: list[str] = []

    # -----------------------------------------------------------------------
    # 1. Crawl & Homepage Foundation (Weight: 0.25)
    # -----------------------------------------------------------------------
    crawl_score = 0.0
    if crawl_result:
        if crawl_result.homepage and crawl_result.homepage.success:
            crawl_score += 0.15
            notes.append("Homepage acquired successfully (200 OK)")
        else:
            notes.append("Homepage acquisition failed or was not 200 OK")

        selected = crawl_result.selected_url_count
        successful_pages = len(crawl_result.pages)
        if selected > 0:
            page_ratio = min(1.0, successful_pages / selected)
            crawl_score += 0.10 * page_ratio
            notes.append(f"Acquired {successful_pages}/{selected} selected pages")
        elif successful_pages > 0:
            crawl_score += 0.10
    else:
        # If crawl_result is not supplied, grant baseline credit if sources exist
        if intelligence.sources:
            crawl_score = 0.15
            notes.append("Crawl metadata omitted; baseline granted based on sources")

    breakdown.crawl_coverage_score = round(crawl_score, 3)

    # -----------------------------------------------------------------------
    # 2. Company Overview Grounding (Weight: 0.20)
    # -----------------------------------------------------------------------
    overview_score = 0.0
    overview = intelligence.company_overview.strip()
    if overview:
        overview_score += 0.15
        if len(overview) >= 30:
            overview_score += 0.05
            notes.append("Overview present and substantive")
        else:
            notes.append("Overview present but short")
    else:
        notes.append("Overview missing")

    breakdown.overview_score = round(overview_score, 3)

    # -----------------------------------------------------------------------
    # 3. Target Audience / ICP Grounding (Weight: 0.15)
    # -----------------------------------------------------------------------
    audience_score = 0.0
    audience = intelligence.target_audience.strip()
    if audience:
        if len(audience) >= 15:
            audience_score += 0.15
            notes.append("Target audience / ICP substantive")
        else:
            audience_score += 0.08
            notes.append("Target audience / ICP present but short")
    else:
        notes.append("Target audience missing")

    breakdown.target_audience_score = round(audience_score, 3)

    # -----------------------------------------------------------------------
    # 4. Leadership Evidence (Weight: 0.20)
    # -----------------------------------------------------------------------
    leadership_score = 0.0
    if intelligence.leadership:
        leadership_score += 0.10
        has_roles = any(member.role for member in intelligence.leadership)
        if has_roles:
            leadership_score += 0.05

        has_grounded_links = any(
            member.linkedin_url or member.source_url
            for member in intelligence.leadership
        )
        if has_grounded_links:
            leadership_score += 0.05

        notes.append(f"Found {len(intelligence.leadership)} validated leadership members")
    else:
        notes.append("No verified leadership members found in evidence")

    breakdown.leadership_score = round(leadership_score, 3)

    # -----------------------------------------------------------------------
    # 5. Contact Evidence (Weight: 0.10)
    # -----------------------------------------------------------------------
    contact_score = 0.0
    if intelligence.contact_points:
        contact_score += 0.10
        notes.append(f"Found {len(intelligence.contact_points)} deterministic contact points")
    else:
        notes.append("No public contact emails detected")

    breakdown.contact_score = round(contact_score, 3)

    # -----------------------------------------------------------------------
    # 6. Source Attribution & Grounding (Weight: 0.10)
    # -----------------------------------------------------------------------
    source_score = 0.0
    if intelligence.sources:
        source_score += 0.10
        notes.append(f"Intelligence grounded in {len(intelligence.sources)} verified sources")
    else:
        notes.append("No verified sources attached")

    breakdown.source_attribution_score = round(source_score, 3)

    # -----------------------------------------------------------------------
    # Aggregate and normalize
    # -----------------------------------------------------------------------
    total = (
        breakdown.crawl_coverage_score
        + breakdown.overview_score
        + breakdown.target_audience_score
        + breakdown.leadership_score
        + breakdown.contact_score
        + breakdown.source_attribution_score
    )

    final_score = max(0.0, min(1.0, round(total, 2)))
    breakdown.total_score = final_score
    breakdown.notes = notes

    logger.debug(
        "Confidence calculated for %s: %.2f (crawl=%.2f, overview=%.2f, audience=%.2f, lead=%.2f, contact=%.2f, sources=%.2f)",
        intelligence.domain,
        final_score,
        breakdown.crawl_coverage_score,
        breakdown.overview_score,
        breakdown.target_audience_score,
        breakdown.leadership_score,
        breakdown.contact_score,
        breakdown.source_attribution_score,
    )

    return final_score, breakdown
