from app.models import CompanyIntelligence


def test_company_intelligence_defaults():
    result = CompanyIntelligence(domain="example.com")

    assert result.domain == "example.com"
    assert result.company_overview == ""
    assert result.target_audience == ""
    assert result.contact_points == []
    assert result.leadership == []
    assert result.confidence_score == 0.0