from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ConsensusSettings(BaseModel):
    flat: int = Field(80, ge=0, le=100)
    trend: int = Field(65, ge=0, le=100)


class PredictRequest(BaseModel):
    state: List[float]
    model_paths: List[str]
    consensus: Optional[ConsensusSettings] = None
    symbol: Optional[str] = None


class SinglePrediction(BaseModel):
    model_path: str
    action: str
    confidence: float
    q_values: Optional[List[float]] = None


class PredictResponse(BaseModel):
    success: bool
    decision: str
    votes: Dict[str, int]
    threshold_used: int
    predictions: List[SinglePrediction]
    error: Optional[str] = None
