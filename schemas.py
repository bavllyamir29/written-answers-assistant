from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GradeWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic: float = Field(default=0.5, ge=0, le=1)
    keywords: float = Field(default=0.4, ge=0, le=1)
    completeness: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def weights_must_sum_to_one(self) -> "GradeWeights":
        total = self.semantic + self.keywords + self.completeness
        if abs(total - 1.0) > 1e-6:
            raise ValueError("semantic, keywords, and completeness weights must sum to 1.")
        return self


Keyword = Annotated[str, Field(min_length=1, max_length=100)]


class GradeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    model_answer: str = Field(min_length=1, max_length=10000)
    student_answer: str = Field(min_length=1, max_length=10000)
    max_score: float = Field(gt=0, le=1000)
    must_have_keywords: list[Keyword] | None = Field(default=None, max_length=100)
    extract_keywords: bool = False
    allow_fuzzy_matching: bool = True
    fuzzy_threshold: float = Field(default=0.88, ge=0.5, le=1)
    normalize_ta_marbuta: bool = False
    weights: GradeWeights = Field(default_factory=GradeWeights)
    score_decimals: int = Field(default=2, ge=0, le=4)

    @field_validator("must_have_keywords")
    @classmethod
    def keywords_must_not_be_blank(cls, keywords: list[str] | None) -> list[str] | None:
        if keywords is not None and any(not keyword.strip() for keyword in keywords):
            raise ValueError("must_have_keywords cannot contain blank terms.")
        return keywords


class GradeResponse(BaseModel):
    score: float
    max_score: float
    final_ratio: float
    base_final_ratio: float
    semantic_similarity: float
    keyword_match_ratio: float
    completeness_ratio: float
    matched_keywords: list[str]
    missing_keywords: list[str]
    feedback_ar: str
    review_flags: list[str]
    contradiction_cap_applied: bool
