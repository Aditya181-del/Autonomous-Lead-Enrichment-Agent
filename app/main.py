import asyncio
from urllib.parse import urlparse

from app.config import get_settings
from app.crawler import BrowserCrawler
from app.discovery import rank_urls
from app.utils import configure_logging, get_logger


logger = get_logger(__name__)


async def run() -> None:
    settings = get_settings()

    configure_logging(settings.log_level)

    logger.info("Autonomous Lead Enrichment Agent")
    logger.info("Environment: %s", settings.app_env)
    logger.info("LLM provider: %s", settings.llm_provider)
    logger.info("Max pages/domain: %s", settings.max_pages_per_domain)

    crawler = BrowserCrawler(settings)

    await crawler.start()

    context = await crawler.create_context()

    try:
        snapshot = await crawler.fetch_page(
            context,
            "https://postman.com",
        )

        print("\n" + "=" * 60)
        print("HOMEPAGE SNAPSHOT")
        print("=" * 60)

        print(f"URL: {snapshot.url}")
        print(f"Title: {snapshot.title}")
        print(f"Status: {snapshot.status_code}")
        print(f"HTML characters: {len(snapshot.html):,}")
        print(f"Text characters: {len(snapshot.text):,}")
        print(f"Links found: {len(snapshot.links):,}")

        root_domain = urlparse(snapshot.url).hostname or ""

        ranked_urls = rank_urls(
            snapshot.links,
            root_domain,
            max_urls=settings.max_pages_per_domain,
        )

        print("\n" + "=" * 60)
        print("RELEVANT PAGES")
        print("=" * 60)

        for url, score in ranked_urls:
            print(f"{score:>3} | {url}")

    finally:
        await context.close()
        await crawler.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()