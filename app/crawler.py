from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from playwright.async_api import Browser, BrowserContext, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from app.config import Settings
from app.utils import get_logger


logger = get_logger(__name__)


@dataclass
class PageSnapshot:
    """
    Represents the information captured from a webpage.
    """

    url: str
    title: str
    html: str
    text: str
    links: list[str]
    status_code: Optional[int]


class BrowserCrawler:
    """
    Browser-based website crawler powered by Playwright.

    Responsibilities:
    - launch Chromium
    - create isolated browser contexts
    - navigate to pages
    - capture basic page information
    - cleanly release browser resources
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._playwright = None
        self._browser: Optional[Browser] = None

    async def start(self) -> None:
        """Start Playwright and launch Chromium."""

        logger.info("Starting Playwright")

        self._playwright = await async_playwright().start()

        self._browser = await self._playwright.chromium.launch(
            headless=True
        )

        logger.info("Chromium browser started")

    async def create_context(self) -> BrowserContext:
        """Create a new isolated browser context."""

        if self._browser is None:
            raise RuntimeError(
                "BrowserCrawler has not been started. "
                "Call start() before creating a context."
            )

        context = await self._browser.new_context()

        context.set_default_timeout(
            self.settings.request_timeout_seconds * 1000
        )

        return context

    async def fetch_page(
        self,
        context: BrowserContext,
        url: str,
    ) -> PageSnapshot:
        """
        Navigate to a URL and capture basic page information.
        """

        page: Page = await context.new_page()

        try:
            logger.info("Navigating to %s", url)

            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self.settings.request_timeout_seconds * 1000,
            )

            # DOMContentLoaded tells us the document has been parsed.
            # Modern sites may continue rendering afterward, so we give
            # the page a small opportunity to settle without assuming
            # that networkidle means "everything is done".
            title = await page.title()

            html = await page.content()

            text = await page.locator("body").inner_text()

            links = await page.locator("a[href]").evaluate_all(
                """
                anchors => anchors.map(anchor => anchor.href)
                """
            )

            status_code = response.status if response else None

            logger.info(
                "Fetched %s | status=%s | title=%s",
                url,
                status_code,
                title,
            )

            return PageSnapshot(
                url=page.url,
                title=title,
                html=html,
                text=text,
                links=links,
                status_code=status_code,
            )

        except PlaywrightTimeoutError:
            logger.warning("Timeout while loading %s", url)

            return PageSnapshot(
                url=url,
                title="",
                html="",
                text="",
                status_code=None,
            )

        except Exception:
            logger.exception("Unexpected error while loading %s", url)

            return PageSnapshot(
                url=url,
                title="",
                html="",
                text="",
                status_code=None,
            )

        finally:
            await page.close()

    async def close(self) -> None:
        """Close browser resources cleanly."""

        logger.info("Closing browser")

        if self._browser is not None:
            await self._browser.close()
            self._browser = None

        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None