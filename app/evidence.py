from __future__ import annotations

from app.models import CleanedPageEvidence


def build_llm_evidence(
    pages: list[CleanedPageEvidence],
) -> str:
    """
    Assemble cleaned first-party page evidence into a compact,
    source-aware LLM context.
    """

    sections: list[str] = []

    for index, page in enumerate(
        pages,
        start=1,
    ):

        sections.append(
            "\n".join(
                [
                    f"SOURCE {index}",
                    f"URL: {page.url}",
                    f"TITLE: {page.title}",
                    "CONTENT:",
                    page.content,
                ]
            )
        )

    return "\n\n".join(
        sections
    )