from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class ContactPoint(BaseModel):
    email: str
    source_url: Optional[HttpUrl] = None


class TeamMember(BaseModel):
    name: str
    role: Optional[str] = None
    linkedin_url: Optional[HttpUrl] = None
    source_url: Optional[HttpUrl] = None


class CompanyIntelligence(BaseModel):
    domain: str

    company_overview: str = ""

    target_audience: str = ""

    contact_points: list[ContactPoint] = Field(default_factory=list)

    leadership: list[TeamMember] = Field(default_factory=list)

    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    sources: list[HttpUrl] = Field(default_factory=list)
    
class PageEvidence(BaseModel):
    url: HttpUrl
    title: str = ""
    content: str = ""
    status_code: Optional[int] = None