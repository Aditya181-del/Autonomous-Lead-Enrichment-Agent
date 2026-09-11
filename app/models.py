from typing import Any, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class ContactPoint(BaseModel):
    """A public company contact point."""

    email: str
    source_url: Optional[HttpUrl] = None


class TeamMember(BaseModel):
    """A leadership/team member discovered from company evidence."""

    name: str
    role: Optional[str] = None
    linkedin_url: Optional[HttpUrl] = None
    source_url: Optional[HttpUrl] = None


class LLMTeamMember(BaseModel):
    """
    Leadership/team member extracted semantically by the LLM.
    """

    name: str = Field(
        description="Full name of the leader or key executive."
    )
    role: Optional[str] = Field(
        default=None,
        description="Job title or executive role."
    )
    linkedin_url: Optional[HttpUrl] = Field(
        default=None,
        description="Personal LinkedIn profile URL if explicitly cited in evidence."
    )
    source_url: Optional[HttpUrl] = Field(
        default=None,
        description="URL of the evidence page mentioning this person."
    )

    @field_validator("linkedin_url", "source_url", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return None
        return v


class LLMSourceCitation(BaseModel):
    """
    Evidence citation emitted by the LLM linking a URL to an extracted claim.
    """

    url: HttpUrl = Field(
        description="Evidence URL supporting the extracted semantic facts."
    )
    claim: Optional[str] = Field(
        default=None,
        description="Specific fact or claim supported by this source URL."
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string_or_dict(cls, data: Any) -> Any:
        if isinstance(data, (str, HttpUrl)):
            return {"url": data}
        return data


class LLMCompanyExtraction(BaseModel):
    """
    Dedicated LLM semantic extraction output contract.

    This schema is strictly limited to semantic fields that the LLM is
    responsible for extracting from evidence. It intentionally excludes
    application-owned fields such as domain, deterministic contact points,
    and confidence scores.
    """

    company_overview: str = Field(
        default="",
        description=(
            "Exactly two concise sentences describing what "
            "the company does, grounded strictly in the evidence."
        ),
    )

    target_audience: str = Field(
        default="",
        description=(
            "Primary ideal customer profile (ICP) or target audience, "
            "based only on the supplied evidence."
        ),
    )

    leadership: list[LLMTeamMember] = Field(
        default_factory=list,
        description=(
            "Founders, executives, or key leaders explicitly "
            "identified in the evidence."
        ),
    )

    sources: list[LLMSourceCitation] = Field(
        default_factory=list,
        description="Evidence URLs and claims supporting the extracted semantic facts.",
    )

    @field_validator("sources", mode="before")
    @classmethod
    def filter_empty_sources(cls, v: Any) -> Any:
        if isinstance(v, list):
            cleaned = []
            for item in v:
                if not item:
                    continue
                if isinstance(item, str) and not item.strip():
                    continue
                cleaned.append(item)
            return cleaned
        return v



class CompanyIntelligence(BaseModel):
    """
    Final structured company intelligence produced by the enrichment
    pipeline after deterministic extraction, validation, and scoring.
    """

    domain: str

    company_overview: str = Field(
        default="",
        description=(
            "Exactly two concise sentences describing what "
            "the company does."
        ),
    )

    target_audience: str = Field(
        default="",
        description=(
            "Primary ideal customer profile (ICP), based only "
            "on the supplied evidence."
        ),
    )

    leadership: list[TeamMember] = Field(
        default_factory=list,
        description=(
            "Key founders, executives, or team members explicitly "
            "supported by the evidence."
        ),
    )

    contact_points: list[ContactPoint] = Field(
        default_factory=list,
        description=(
            "Public generic company contact emails discovered deterministically."
        ),
    )

    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Evidence-based heuristic confidence score between 0.0 and 1.0."
        ),
    )

    sources: list[HttpUrl] = Field(
        default_factory=list,
        description="Validated URLs supporting the extracted information.",
    )

    crawl_status: str = Field(
        default="success",
        description="Overall crawl and enrichment status: 'success', 'partial', or 'failed'.",
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error details if enrichment encountered failure.",
    )

class PageEvidence(BaseModel):
    """
    Evidence collected from one webpage.

    Raw HTML is retained for deterministic entity extraction and
    downstream preprocessing.
    """

    url: HttpUrl
    title: str = ""
    html: str = ""
    content: str = ""
    status_code: Optional[int] = None

    # href values may contain https://, mailto:, tel:, etc.
    links: list[str] = Field(
        default_factory=list
    )

    success: bool = False
    error: Optional[str] = None


class CleanedPageEvidence(BaseModel):
    """
    LLM-ready representation after content preprocessing.
    """

    url: HttpUrl
    title: str = ""
    content: str = ""

    original_characters: int = 0
    cleaned_characters: int = 0

    original_lines: int = 0
    cleaned_lines: int = 0

    estimated_input_tokens: int = 0

    reduction_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    links: list[str] = Field(
        default_factory=list
    )

    source_status_code: Optional[int] = None


class CrawlResult(BaseModel):
    """
    Complete evidence-acquisition result for one company domain.
    """

    domain: str

    homepage: Optional[PageEvidence] = None

    pages: list[PageEvidence] = Field(
        default_factory=list
    )

    discovered_url_count: int = 0

    internal_candidate_count: int = 0

    selected_url_count: int = 0

    failed_url_count: int = 0


class ExtractedEmail(BaseModel):
    """
    Email address discovered deterministically from a webpage.
    """

    email: str
    source_url: HttpUrl


class ExtractedLinkedInProfile(BaseModel):
    """
    Individual LinkedIn profile URL discovered deterministically.
    """

    linkedin_url: HttpUrl
    source_url: HttpUrl


class ExtractedEntities(BaseModel):
    """
    Deterministic entities extracted from one webpage.
    """

    source_url: HttpUrl

    emails: list[ExtractedEmail] = Field(
        default_factory=list
    )

    linkedin_profiles: list[ExtractedLinkedInProfile] = Field(
        default_factory=list
    )