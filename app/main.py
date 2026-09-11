from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from typing import Optional
from urllib.parse import urlparse

from pydantic import HttpUrl

from app.acquisition import EvidenceAcquisitionService, normalize_input_domain
from app.cleaner import clean_page
from app.config import Settings, get_settings
from app.crawler import BrowserCrawler
from app.discovery import normalize_url
from app.evidence import build_llm_evidence
from app.extractor import extract_entities
from app.llm import LLMExtractionResult, LLMUsage, OllamaExtractor
from app.models import (
    CleanedPageEvidence,
    CompanyIntelligence,
    ContactPoint,
    CrawlResult,
    LLMCompanyExtraction,
    TeamMember,
)
from app.scoring import ConfidenceBreakdown, calculate_confidence_score
from app.utils import configure_logging, get_logger
from app.validators import (
    validate_leadership,
    validate_overview,
    validate_sources,
    validate_target_audience,
)


logger = get_logger(__name__)

DEFAULT_DOMAINS = [
    "postman.com",
    "supabase.com",
    "vapi.ai",
]


async def enrich_domain(
    domain: str,
    crawler: BrowserCrawler,
    settings: Settings,
) -> tuple[CompanyIntelligence, LLMUsage, ConfidenceBreakdown]:
    """
    Execute the end-to-end enrichment pipeline for a single company domain
    with full failure isolation and evidence grounding.
    """

    clean_domain = normalize_input_domain(domain)
    start_time = time.perf_counter()
    usage = LLMUsage()
    breakdown = ConfidenceBreakdown()

    logger.info(">>> Starting lead enrichment for domain: %s", clean_domain)

    context = None
    try:
        context = await crawler.create_context()
        acquisition_service = EvidenceAcquisitionService(crawler, settings)

        # -------------------------------------------------------------------
        # 1. Page Acquisition
        # -------------------------------------------------------------------
        crawl_result = await acquisition_service.collect_domain(
            context=context,
            domain=clean_domain,
        )

        # Handle total acquisition failure
        if not crawl_result.homepage or not crawl_result.homepage.success:
            logger.warning(
                "Domain crawl failed for %s | Error: %s",
                clean_domain,
                crawl_result.homepage.error if crawl_result.homepage else "No homepage evidence",
            )
            intelligence = CompanyIntelligence(
                domain=clean_domain,
                company_overview="",
                target_audience="",
                leadership=[],
                contact_points=[],
                confidence_score=0.0,
                sources=[],
                crawl_status="failed",
                error_message=crawl_result.homepage.error if crawl_result.homepage else "Homepage unreachable",
            )
            _, breakdown = calculate_confidence_score(intelligence, crawl_result)
            return intelligence, usage, breakdown

        # Aggregate all acquired pages (homepage + selected internal pages)
        all_pages = []
        if crawl_result.homepage and crawl_result.homepage.success:
            all_pages.append(crawl_result.homepage)
        all_pages.extend(crawl_result.pages)

        acquired_urls = {
            normalize_url(str(page.url))
            for page in all_pages
            if page.url
        }

        # -------------------------------------------------------------------
        # 2. Dual-Path: Deterministic Extraction & Content Cleaning
        # -------------------------------------------------------------------
        deterministic_emails: set[str] = set()
        observed_linkedin_urls: set[str] = set()
        email_source_map: dict[str, HttpUrl] = {}
        cleaned_pages: list[CleanedPageEvidence] = []

        for page in all_pages:
            # Deterministic extraction
            entities = extract_entities(page)
            for item in entities.emails:
                deterministic_emails.add(item.email)
                if item.email not in email_source_map:
                    email_source_map[item.email] = item.source_url

            for item in entities.linkedin_profiles:
                norm_linkedin = normalize_url(str(item.linkedin_url))
                if norm_linkedin:
                    observed_linkedin_urls.add(norm_linkedin)

            # Semantic content cleaning
            cleaned = clean_page(page)
            if cleaned.content.strip():
                cleaned_pages.append(cleaned)

        # -------------------------------------------------------------------
        # 3. LLM Evidence Assembly & Semantic Extraction
        # -------------------------------------------------------------------
        llm_evidence = build_llm_evidence(cleaned_pages)
        llm_extractor = OllamaExtractor(settings)

        llm_result: LLMExtractionResult = await llm_extractor.extract(
            domain=clean_domain,
            evidence_text=llm_evidence,
            deterministic_emails=sorted(deterministic_emails),
        )

        usage = llm_result.usage
        raw_extraction: LLMCompanyExtraction = llm_result.data

        # -------------------------------------------------------------------
        # 4. Application Validation
        # -------------------------------------------------------------------
        validated_sources = validate_sources(
            llm_sources=raw_extraction.sources,
            acquired_urls=acquired_urls,
        )

        # If LLM omitted sources but we acquired evidence, default to verified acquired URLs
        if not validated_sources and acquired_urls:
            for page in all_pages:
                validated_sources.append(page.url)

        validated_leadership = validate_leadership(
            llm_leadership=raw_extraction.leadership,
            acquired_urls=acquired_urls,
            observed_linkedin_urls=observed_linkedin_urls,
        )

        validated_overview = validate_overview(raw_extraction.company_overview)
        validated_audience = validate_target_audience(raw_extraction.target_audience)

        # Build contact points deterministically
        contact_points = [
            ContactPoint(
                email=email,
                source_url=email_source_map.get(email),
            )
            for email in sorted(deterministic_emails)
        ]

        # -------------------------------------------------------------------
        # 5. Final Assembly & Confidence Scoring
        # -------------------------------------------------------------------
        intelligence = CompanyIntelligence(
            domain=clean_domain,
            company_overview=validated_overview,
            target_audience=validated_audience,
            leadership=validated_leadership,
            contact_points=contact_points,
            confidence_score=0.0,  # Computed below
            sources=validated_sources,
            crawl_status="success",
            error_message=None,
        )

        confidence_score, breakdown = calculate_confidence_score(
            intelligence=intelligence,
            crawl_result=crawl_result,
        )
        intelligence.confidence_score = confidence_score

        elapsed = time.perf_counter() - start_time
        logger.info(
            "<<< Finished enrichment for %s in %.2fs | Confidence: %.2f | Status: %s",
            clean_domain,
            elapsed,
            intelligence.confidence_score,
            intelligence.crawl_status,
        )

        return intelligence, usage, breakdown

    except Exception as exc:
        logger.exception("Unhandled error enriching domain %s: %s", clean_domain, exc)
        failed_intelligence = CompanyIntelligence(
            domain=clean_domain,
            company_overview="",
            target_audience="",
            leadership=[],
            contact_points=[],
            confidence_score=0.0,
            sources=[],
            crawl_status="failed",
            error_message=str(exc),
        )
        _, breakdown = calculate_confidence_score(failed_intelligence)
        return failed_intelligence, usage, breakdown

    finally:
        if context is not None:
            await context.close()


async def enrich_batch(
    domains: list[str],
    settings: Settings,
) -> list[tuple[CompanyIntelligence, LLMUsage, ConfidenceBreakdown]]:
    """
    Enrich a batch of domains concurrently, bounded by max_concurrent_domains.
    """

    crawler = BrowserCrawler(settings)
    await crawler.start()

    semaphore = asyncio.Semaphore(settings.max_concurrent_domains)

    async def _worker(domain: str):
        async with semaphore:
            return await enrich_domain(
                domain=domain,
                crawler=crawler,
                settings=settings,
            )

    try:
        tasks = [_worker(d) for d in domains]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results
    finally:
        await crawler.close()


def save_output_json(
    results: list[tuple[CompanyIntelligence, LLMUsage, ConfidenceBreakdown]],
    filepath: str = "output/output.json",
) -> None:
    """Save enriched company intelligence as formatted JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    data = [item[0].model_dump(mode="json") for item in results]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved structured JSON output to %s", filepath)


def save_output_csv(
    results: list[tuple[CompanyIntelligence, LLMUsage, ConfidenceBreakdown]],
    filepath: str = "output/output.csv",
) -> None:
    """Save enriched company intelligence summary as CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fieldnames = [
        "domain",
        "confidence_score",
        "crawl_status",
        "company_overview",
        "target_audience",
        "leadership_count",
        "leadership_names",
        "contact_emails",
        "sources_count",
        "input_tokens",
        "output_tokens",
        "error_message",
    ]

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for intelligence, usage, _ in results:
            writer.writerow(
                {
                    "domain": intelligence.domain,
                    "confidence_score": intelligence.confidence_score,
                    "crawl_status": intelligence.crawl_status,
                    "company_overview": intelligence.company_overview,
                    "target_audience": intelligence.target_audience,
                    "leadership_count": len(intelligence.leadership),
                    "leadership_names": "; ".join(m.name for m in intelligence.leadership),
                    "contact_emails": "; ".join(c.email for c in intelligence.contact_points),
                    "sources_count": len(intelligence.sources),
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "error_message": intelligence.error_message or "",
                }
            )

    logger.info("Saved summary CSV output to %s", filepath)


def print_summary_table(
    results: list[tuple[CompanyIntelligence, LLMUsage, ConfidenceBreakdown]],
    total_elapsed: float,
) -> None:
    """Print an explainable console summary table of batch enrichment results."""

    print("\n" + "=" * 90)
    print("AUTONOMOUS LEAD ENRICHMENT AGENT — EXECUTION SUMMARY")
    print("=" * 90)

    total_input_tokens = 0
    total_output_tokens = 0
    successful_count = 0

    for intelligence, usage, breakdown in results:
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens
        if intelligence.crawl_status == "success":
            successful_count += 1

        print(f"\n[DOMAIN]: {intelligence.domain.upper()}")
        print(f"  Status:           {intelligence.crawl_status.upper()}")
        print(f"  Confidence Score: {intelligence.confidence_score:.2f} / 1.00")
        print(f"  Company Overview: {intelligence.company_overview or '(none)'}")
        print(f"  Target Audience:  {intelligence.target_audience or '(none)'}")

        # Leadership
        if intelligence.leadership:
            print("  Leadership:")
            for m in intelligence.leadership:
                role_str = f" ({m.role})" if m.role else ""
                li_str = f" [LinkedIn: {m.linkedin_url}]" if m.linkedin_url else ""
                print(f"    - {m.name}{role_str}{li_str}")
        else:
            print("  Leadership:       (None verified in evidence)")

        # Contact Points
        if intelligence.contact_points:
            emails = ", ".join(c.email for c in intelligence.contact_points)
            print(f"  Contact Emails:   {emails}")
        else:
            print("  Contact Emails:   (None detected)")

        # Sources
        print(f"  Verified Sources: {len(intelligence.sources)} URLs")
        for s in intelligence.sources[:3]:
            print(f"    * {s}")
        if len(intelligence.sources) > 3:
            print(f"    * ... and {len(intelligence.sources) - 3} more")

        # Telemetry
        print(f"  LLM Telemetry:    In={usage.input_tokens:,} tokens | Out={usage.output_tokens:,} tokens | Latency={usage.latency_seconds:.2f}s")
        if intelligence.error_message:
            print(f"  Error:            {intelligence.error_message}")

    print("\n" + "-" * 90)
    print("BATCH METRICS:")
    print(f"  Total Domains:        {len(results)}")
    print(f"  Successful:           {successful_count}")
    print(f"  Failed:               {len(results) - successful_count}")
    print(f"  Total Tokens Used:    {total_input_tokens + total_output_tokens:,} (Input: {total_input_tokens:,} | Output: {total_output_tokens:,})")
    print(f"  Total Batch Runtime:  {total_elapsed:.2f}s")
    print("=" * 90 + "\n")


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Lead Enrichment Agent"
    )
    parser.add_argument(
        "domains",
        nargs="*",
        help="Company domains to enrich (e.g. postman.com supabase.com vapi.ai)",
    )
    parser.add_argument(
        "--json-output",
        default="output/output.json",
        help="Path for output JSON file (default: output/output.json)",
    )
    parser.add_argument(
        "--csv-output",
        default="output/output.csv",
        help="Path for output CSV file (default: output/output.csv)",
    )

    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    domains = args.domains if args.domains else DEFAULT_DOMAINS
    logger.info("Initializing enrichment batch for %d domains: %s", len(domains), domains)

    batch_start = time.perf_counter()
    results = await enrich_batch(domains, settings)
    batch_elapsed = time.perf_counter() - batch_start

    save_output_json(results, args.json_output)
    save_output_csv(results, args.csv_output)

    print_summary_table(results, batch_elapsed)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()