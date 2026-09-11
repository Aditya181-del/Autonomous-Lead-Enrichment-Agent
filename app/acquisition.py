from __future__ import annotations

from urllib.parse import urlparse

from playwright.async_api import BrowserContext

from app.config import Settings
from app.crawler import BrowserCrawler
from app.discovery import rank_urls
from app.models import CrawlResult, PageEvidence
from app.utils import get_logger


logger = get_logger(__name__)


def normalize_input_domain(raw_domain: str) -> str:
    """
    Clean and normalize an input company domain.

    Examples:
        'https://www.postman.com/' -> 'postman.com'
        'http://supabase.com' -> 'supabase.com'
        'vapi.ai/' -> 'vapi.ai'
    """

    if not raw_domain:
        return ""

    cleaned = raw_domain.strip().lower()

    if cleaned.startswith("http://"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("https://"):
        cleaned = cleaned[8:]

    cleaned = cleaned.rstrip("/").split("/")[0]

    if cleaned.startswith("www."):
        cleaned = cleaned[4:]

    return cleaned


class EvidenceAcquisitionService:
    """
    Orchestrates homepage discovery and selected-page acquisition.
    """

    def __init__(
        self,
        crawler: BrowserCrawler,
        settings: Settings,
    ) -> None:
        self.crawler = crawler
        self.settings = settings

    async def collect_domain(
        self,
        context: BrowserContext,
        domain: str,
    ) -> CrawlResult:
        """
        Collect homepage and relevant internal-page evidence
        for a single domain with failure isolation.
        """

        clean_domain = normalize_input_domain(domain)
        if not clean_domain:
            return CrawlResult(
                domain=domain,
                failed_url_count=1,
            )

        homepage_url = f"https://{clean_domain}"

        logger.info(
            "Starting evidence acquisition for %s (%s)",
            clean_domain,
            homepage_url,
        )

        homepage_snapshot = await self.crawler.fetch_page(
            context,
            homepage_url,
        )

        homepage_evidence = (
            self.crawler.snapshot_to_evidence(
                homepage_snapshot
            )
        )

        if not homepage_evidence.success:
            logger.warning(
                "Homepage acquisition failed for %s (status=%s, error=%s)",
                clean_domain,
                homepage_evidence.status_code,
                homepage_evidence.error,
            )

            return CrawlResult(
                domain=clean_domain,
                homepage=homepage_evidence,
                discovered_url_count=len(
                    homepage_snapshot.links
                ),
                internal_candidate_count=0,
                selected_url_count=0,
                failed_url_count=1,
            )

        root_domain = (
            urlparse(homepage_snapshot.url).hostname
            or clean_domain
        )
        if root_domain.startswith("www."):
            root_domain = root_domain[4:]

        ranked_urls = rank_urls(
            homepage_snapshot.links,
            root_domain,
            max_urls=self.settings.max_pages_per_domain,
        )

        logger.info(
            "Selected %d relevant pages for %s",
            len(ranked_urls),
            clean_domain,
        )

        pages: list[PageEvidence] = []
        failed_count = 0

        for url, score in ranked_urls:
            logger.info(
                "Acquiring page | score=%d | url=%s",
                score,
                url,
            )

            try:
                evidence = await self.crawler.fetch_evidence(
                    context,
                    url,
                )

                if evidence.success:
                    pages.append(evidence)
                else:
                    failed_count += 1
                    logger.warning(
                        "Page fetch unsuccessful: %s (status=%s, error=%s)",
                        url,
                        evidence.status_code,
                        evidence.error,
                    )
            except Exception as exc:
                failed_count += 1
                logger.exception("Unhandled error acquiring page %s: %s", url, exc)

        return CrawlResult(
            domain=clean_domain,
            homepage=homepage_evidence,
            pages=pages,
            discovered_url_count=len(
                homepage_snapshot.links
            ),
            internal_candidate_count=self._count_internal_candidates(
                homepage_snapshot.links,
                root_domain,
            ),
            selected_url_count=len(ranked_urls),
            failed_url_count=failed_count,
        )

    def _count_internal_candidates(
        self,
        urls: list[str],
        root_domain: str,
    ) -> int:
        """
        Count the number of unique internal candidates before
        top-N selection.
        """

        from app.discovery import (
            is_internal_url,
            normalize_url,
        )

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

        return len(candidates)