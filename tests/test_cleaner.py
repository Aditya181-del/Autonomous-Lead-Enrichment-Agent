from app.cleaner import (
    clean_page,
    deduplicate_lines,
    estimate_tokens,
    normalize_whitespace,
)
from app.models import PageEvidence


def test_normalize_whitespace():
    text = """
    Hello       world

    This     is    a test.
    """

    result = normalize_whitespace(text)

    assert result == (
        "Hello world\n"
        "This is a test."
    )


def test_deduplicate_lines():
    text = """About Us
About Us
Our company
Contact
Contact
"""

    result = deduplicate_lines(text)

    assert result == (
        "About Us\n"
        "Our company\n"
        "Contact"
    )


def test_estimate_tokens():
    text = "a" * 400

    assert estimate_tokens(text) == 100


def test_clean_page_removes_noise():
    html = """
    <html>
        <head>
            <style>
                body { color: red; }
            </style>
        </head>

        <body>

            <nav>
                Home
                Product
                Contact
            </nav>

            <main>

                <h1>About Example Company</h1>

                <p>
                    Example Company builds software for modern teams.
                </p>

                <div class="cookie-banner">
                    Accept cookies
                </div>

                <p>
                    We serve developers and businesses worldwide.
                </p>

                <script>
                    console.log("noise");
                </script>

            </main>

            <footer>
                Privacy
                Terms
            </footer>

        </body>
    </html>
    """

    page = PageEvidence(
        url="https://example.com/about",
        title="About Example",
        html=html,
        content="",
        status_code=200,
        links=[
            "https://example.com/team",
        ],
        success=True,
    )

    cleaned = clean_page(page)

    assert "Home" not in cleaned.content
    assert "Accept cookies" not in cleaned.content
    assert "Privacy" not in cleaned.content
    assert "console.log" not in cleaned.content

    assert (
        "About Example Company"
        in cleaned.content
    )

    assert (
        "modern teams"
        in cleaned.content
    )

    assert cleaned.cleaned_characters > 0
    assert cleaned.original_characters > 0

    assert (
        cleaned.reduction_ratio > 0
    )