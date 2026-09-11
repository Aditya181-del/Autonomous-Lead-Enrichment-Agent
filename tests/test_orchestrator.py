import json
import os
import tempfile
import pytest
from pydantic import HttpUrl

from app.main import (
    DEFAULT_DOMAINS,
    save_output_csv,
    save_output_json,
)
from app.models import CompanyIntelligence, ContactPoint, TeamMember
from app.llm import LLMUsage
from app.scoring import ConfidenceBreakdown


def test_default_domains():
    assert "postman.com" in DEFAULT_DOMAINS
    assert "supabase.com" in DEFAULT_DOMAINS
    assert "vapi.ai" in DEFAULT_DOMAINS


def test_save_output_json_and_csv():
    results = [
        (
            CompanyIntelligence(
                domain="example.com",
                company_overview="Example Inc provides developer tooling.",
                target_audience="Software developers.",
                leadership=[
                    TeamMember(
                        name="Jane Doe",
                        role="CEO",
                        linkedin_url=HttpUrl("https://www.linkedin.com/in/jane-doe"),
                        source_url=HttpUrl("https://example.com/about"),
                    )
                ],
                contact_points=[
                    ContactPoint(
                        email="contact@example.com",
                        source_url=HttpUrl("https://example.com/contact"),
                    )
                ],
                confidence_score=0.88,
                sources=[HttpUrl("https://example.com/about")],
                crawl_status="success",
            ),
            LLMUsage(
                model="nemotron-3-super:cloud",
                input_tokens=150,
                output_tokens=60,
                total_tokens=210,
                latency_seconds=1.25,
            ),
            ConfidenceBreakdown(total_score=0.88),
        ),
        (
            CompanyIntelligence(
                domain="failed-domain.com",
                company_overview="",
                target_audience="",
                leadership=[],
                contact_points=[],
                confidence_score=0.0,
                sources=[],
                crawl_status="failed",
                error_message="DNS lookup failed",
            ),
            LLMUsage(),
            ConfidenceBreakdown(),
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_output.json")
        csv_path = os.path.join(tmpdir, "test_output.csv")

        save_output_json(results, json_path)
        save_output_csv(results, csv_path)

        assert os.path.exists(json_path)
        assert os.path.exists(csv_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 2
            assert data[0]["domain"] == "example.com"
            assert data[0]["crawl_status"] == "success"
            assert data[1]["domain"] == "failed-domain.com"
            assert data[1]["crawl_status"] == "failed"

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 3  # Header + 2 data rows
            assert "domain,confidence_score" in lines[0]
            assert "example.com,0.88" in lines[1]
            assert "failed-domain.com,0.0" in lines[2]
