from app.discovery import (
    get_hostname,
    is_internal_url,
    normalize_url,
    rank_urls,
    score_url,
)


def test_normalize_url_removes_fragment():
    url = "https://www.postman.com/team/#members"

    assert normalize_url(url) == "https://www.postman.com/team"


def test_hostname_removes_www():
    assert (
        get_hostname("https://www.postman.com/about")
        == "postman.com"
    )


def test_internal_url():
    assert is_internal_url(
        "https://www.postman.com/about",
        "postman.com",
    )


def test_external_url():
    assert not is_internal_url(
        "https://buildwithfern.com/",
        "postman.com",
    )


def test_evil_domain_not_internal():
    assert not is_internal_url(
        "https://evilpostman.com/",
        "postman.com",
    )


def test_team_scores_higher_than_blog():
    assert score_url(
        "https://postman.com/team"
    ) > score_url(
        "https://postman.com/blog"
    )


def test_rank_urls():
    urls = [
        "https://postman.com/blog",
        "https://postman.com/team",
        "https://postman.com/about",
        "https://buildwithfern.com/",
        "https://postman.com/contact",
    ]

    ranked = rank_urls(
        urls,
        "postman.com",
        max_urls=3,
    )

    selected_urls = [url for url, _ in ranked]

    assert len(selected_urls) == 3
    assert "https://postman.com/team" in selected_urls
    assert "https://postman.com/about" in selected_urls
    assert "https://buildwithfern.com" not in selected_urls


def test_company_careers_does_not_get_company_score():
    assert score_url(
        "https://postman.com/company/careers"
    ) < score_url(
        "https://postman.com/company/about-postman"
    )


def test_press_media_is_low_priority():
    assert score_url(
        "https://postman.com/company/press-media"
    ) < score_url(
        "https://postman.com/pricing"
    )


def test_company_contact_is_high_priority():
    assert score_url(
        "https://postman.com/company/contact-us"
    ) > score_url(
        "https://postman.com/company/careers"
    )


def test_company_about_is_highest_priority():
    assert score_url(
        "https://postman.com/company/about-postman"
    ) > score_url(
        "https://postman.com/company/careers"
    )