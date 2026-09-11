import pytest

from app.acquisition import normalize_input_domain
from app.config import Settings
from app.crawler import BrowserCrawler, PageSnapshot


def test_snapshot_to_evidence_success():
    crawler = BrowserCrawler(Settings())
    snapshot = PageSnapshot(
        url="https://example.com/about",
        title="About Us",
        html="<html><body><h1>About Us</h1></body></html>",
        text="About Us",
        links=["https://example.com/team"],
        status_code=200,
        error=None,
    )

    evidence = crawler.snapshot_to_evidence(snapshot)
    assert evidence.success is True
    assert evidence.status_code == 200
    assert evidence.title == "About Us"
    assert evidence.content == "About Us"
    assert evidence.error is None


def test_snapshot_to_evidence_404_failure():
    crawler = BrowserCrawler(Settings())
    snapshot = PageSnapshot(
        url="https://example.com/not-found",
        title="404 Not Found",
        html="<html><body>Not Found</body></html>",
        text="Not Found",
        status_code=404,
        links=[],
        error="HTTP status 404",
    )

    evidence = crawler.snapshot_to_evidence(snapshot)
    assert evidence.success is False
    assert evidence.status_code == 404
    assert evidence.error == "HTTP status 404"


def test_snapshot_to_evidence_empty_text_failure():
    crawler = BrowserCrawler(Settings())
    snapshot = PageSnapshot(
        url="https://example.com/empty",
        title="",
        html="<html><body></body></html>",
        text="",
        status_code=200,
        links=[],
        error=None,
    )

    evidence = crawler.snapshot_to_evidence(snapshot)
    assert evidence.success is False


def test_normalize_input_domain():
    assert normalize_input_domain("https://www.postman.com/") == "postman.com"
    assert normalize_input_domain("http://supabase.com/docs") == "supabase.com"
    assert normalize_input_domain("vapi.ai/") == "vapi.ai"
    assert normalize_input_domain("postman.com") == "postman.com"
    assert normalize_input_domain("") == ""   