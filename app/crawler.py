import asyncio
from dataclasses import dataclass
from typing import Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError,
)
from playwright.async_api import async_playwright

from app.config import Settings
from app.models import PageEvidence
from app.utils import get_logger


logger = get_logger(__name__)


# Modern standard browser user-agent to ensure fair compatibility
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class PageSnapshot:
    """
    Raw browser observation for a single page.
    """

    url: str
    title: str
    html: str
    text: str
    links: list[str]
    status_code: Optional[int]
    error: Optional[str] = None


class BrowserCrawler:
    """
    Browser-based crawler powered by Playwright.

    Responsibilities:
    - launch Chromium in headless mode
    - create isolated browser contexts
    - navigate to pages with bounded retries on transient errors
    - capture page title, text, rendered DOM, and links
    - convert browser observations into PageEvidence
    - release resources cleanly
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._playwright = None
        self._browser: Optional[Browser] = None

    async def start(self) -> None:
        """
        Start Playwright and launch Chromium.
        """

        logger.info("Starting Playwright Chromium")

        self._playwright = await async_playwright().start()

        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        logger.info("Chromium browser started successfully")

    async def create_context(self) -> BrowserContext:
        """
        Create a new isolated browser context with custom user agent.
        """

        if self._browser is None:
            raise RuntimeError(
                "BrowserCrawler has not been started. "
                "Call start() before creating a context."
            )

        context = await self._browser.new_context(
            user_agent=DEFAULT_USER_AGENT,
            viewport={"width": 1280, "height": 800},
            ignore_https_errors=True,
        )

        context.set_default_timeout(
            self.settings.request_timeout_seconds * 1000
        )

        return context

    async def _fetch_attempt(
        self,
        context: BrowserContext,
        url: str,
    ) -> tuple[PageSnapshot, bool]:
        """
        Perform a single navigation attempt.
        Returns (PageSnapshot, is_transient_error).
        """

        page: Optional[Page] = None
        try:
            page = await context.new_page()
            logger.info("Navigating to %s", url)

            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self.settings.request_timeout_seconds * 1000,
            )

            # Brief settling period for dynamic rendering
            await page.wait_for_timeout(1000)

            title = await page.title()
            html = await page.content()

            try:
                body = page.locator("body")
                text = await body.inner_text()
            except Exception:
                text = ""

            try:
                links = await page.locator("a[href]").evaluate_all(
                    """anchors => anchors.map(a => a.href).filter(Boolean)"""
                )
            except Exception:
                links = []

            status_code = response.status if response else None

            logger.info(
                "Fetched %s | status=%s | title=%s | links=%d",
                url,
                status_code,
                title[:40] if title else "(no title)",
                len(links),
            )

            # Check if status code represents a permanent client error (e.g., 404, 403, 410)
            is_permanent = status_code is not None and (
                status_code in {400, 401, 403, 404, 405, 410, 451}
            )

            return PageSnapshot(
                url=page.url or url,
                title=title or "",
                html=html or "",
                text=text or "",
                links=links or [],
                status_code=status_code,
                error=None if (status_code and 200 <= status_code < 400) else f"HTTP status {status_code}",
            ), not is_permanent

        except PlaywrightTimeoutError as exc:
            logger.warning("Timeout loading %s: %s", url, exc)
            return PageSnapshot(
                url=url,
                title="",
                html="",
                text="",
                links=[],
                status_code=None,
                error=f"TimeoutError: {exc}",
            ), True  # Timeouts are transient and retryable

        except PlaywrightError as exc:
            logger.warning("Playwright error loading %s: %s", url, exc)
            return PageSnapshot(
                url=url,
                title="",
                html="",
                text="",
                links=[],
                status_code=None,
                error=f"PlaywrightError: {exc}",
            ), True

        except Exception as exc:
            logger.exception("Unexpected error loading %s: %s", url, exc)
            return PageSnapshot(
                url=url,
                title="",
                html="",
                text="",
                links=[],
                status_code=None,
                error=f"Exception: {exc}",
            ), False

        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception:
                    pass

    async def fetch_page(
        self,
        context: BrowserContext,
        url: str,
    ) -> PageSnapshot:
        """
        Navigate to a URL and capture browser-level page data with bounded retries
        for transient failures.
        """

        max_attempts = max(1, self.settings.max_retries + 1)
        last_snapshot: Optional[PageSnapshot] = None

        for attempt in range(1, max_attempts + 1):
            snapshot, is_transient = await self._fetch_attempt(context, url)

            # Success condition: status 200-399 and has some text/html
            if snapshot.status_code is not None and 200 <= snapshot.status_code < 400:
                return snapshot

            last_snapshot = snapshot

            # Do not retry permanent errors or on final attempt
            if not is_transient or attempt >= max_attempts:
                break

            backoff_seconds = 1.0 * attempt
            logger.info(
                "Transient error on %s (attempt %d/%d). Retrying in %.1fs...",
                url,
                attempt,
                max_attempts,
                backoff_seconds,
            )
            await asyncio.sleep(backoff_seconds)

        return last_snapshot or PageSnapshot(
            url=url,
            title="",
            html="",
            text="",
            links=[],
            status_code=None,
            error="Fetch failed with no snapshot produced",
        )

    def snapshot_to_evidence(
        self,
        snapshot: PageSnapshot,
    ) -> PageEvidence:
        """
        Convert a browser PageSnapshot into an application-level
        PageEvidence object.
        """

        success = (
            snapshot.status_code is not None
            and 200 <= snapshot.status_code < 400
            and bool(snapshot.text.strip())
        )

        return PageEvidence(
            url=snapshot.url,
            title=snapshot.title,
            html=snapshot.html,
            content=snapshot.text,
            status_code=snapshot.status_code,
            links=snapshot.links,
            success=success,
            error=snapshot.error,
        )

    async def fetch_evidence(
        self,
        context: BrowserContext,
        url: str,
    ) -> PageEvidence:
        """
        Fetch a URL and return structured page evidence.
        """

        snapshot = await self.fetch_page(
            context,
            url,
        )

        return self.snapshot_to_evidence(
            snapshot
        )

    async def close(self) -> None:
        """
        Close browser resources cleanly with complete exception safety.
        """

        logger.info("Closing browser")

        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception as exc:
                logger.debug("Error closing browser: %s", exc)
            finally:
                self._browser = None

        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as exc:
                logger.debug("Error stopping playwright: %s", exc)
            finally:
                self._playwright = None