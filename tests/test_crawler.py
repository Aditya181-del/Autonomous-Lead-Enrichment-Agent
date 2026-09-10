import pytest

from app.config import Settings
from app.crawler import BrowserCrawler


@pytest.mark.asyncio
async def test_postman_homepage():
    settings = Settings(
        request_timeout_seconds=30,
    )

    crawler = BrowserCrawler(settings)

    await crawler.start()

    context = await crawler.create_context()

    try:
        snapshot = await crawler.fetch_page(
            context,
            "https://postman.com",
        )

        assert snapshot.url
        assert snapshot.title
        assert snapshot.text
        assert isinstance(snapshot.links, list)

    finally:
        await context.close()
        await crawler.close()   