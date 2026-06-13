"""
Pydantic response/request models for the Dwarpala REST API.

These exist so OpenAPI (`/docs`) renders accurate schemas. The endpoint logic
builds plain dicts from the pipeline result and returns them; FastAPI validates
and documents them against these models.
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class QualityBlock(BaseModel):
    is_acceptable: bool
    blur_score: float
    brightness: float
    face_width: int
    face_height: int
    landmark_confidence: float
    issues: list = Field(default_factory=list)


class VerifyResponse(BaseModel):
    request_id: str
    verdict: str = Field(..., description="ACCEPT | REJECT | MANUAL_REVIEW")
    match_score: float
    liveness_score: Optional[float] = None
    liveness_breakdown: Dict[str, Optional[float]] = Field(default_factory=dict)
    signal_status: Dict[str, str] = Field(default_factory=dict)
    quality: Dict[str, Optional[QualityBlock]] = Field(default_factory=dict)
    explanation: str
    latency_ms: float


class LivenessResponse(BaseModel):
    request_id: str
    verdict: str = Field(..., description="LIVE | SPOOF")
    is_live: bool
    liveness_score: Optional[float] = None
    liveness_breakdown: Dict[str, Optional[float]] = Field(default_factory=dict)
    signal_status: Dict[str, str] = Field(default_factory=dict)
    quality: Dict[str, Optional[QualityBlock]] = Field(default_factory=dict)
    explanation: str
    latency_ms: float


class MatchResponse(BaseModel):
    request_id: str
    verdict: str = Field(..., description="MATCH | NO_MATCH | MANUAL_REVIEW")
    is_match: bool
    match_score: float
    match_confidence: str
    match_needs_review: bool
    quality: Dict[str, Optional[QualityBlock]] = Field(default_factory=dict)
    explanation: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok | loading | error")
    models_loaded: bool
    model_versions: Dict[str, str] = Field(default_factory=dict)
    uptime_seconds: float


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorBody
