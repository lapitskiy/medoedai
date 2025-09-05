from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import List
import os

from models import PredictRequest, PredictResponse, SinglePrediction
from ensemble_predictor import EnsemblePredictor

app = FastAPI(title="MedoedAI Serving", version="0.1.0")

predictor = EnsemblePredictor(model_base_dir=os.environ.get("MODEL_BASE_DIR", "/workspace/models"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_ensemble", response_model=PredictResponse)
async def predict_ensemble(req: PredictRequest):
    try:
        if not req.model_paths:
            return JSONResponse(status_code=400, content={"success": False, "error": "model_paths is empty"})
        state = req.state
        models = req.model_paths
        consensus_flat = 80
        consensus_trend = 65
        if req.consensus:
            consensus_flat = int(req.consensus.flat)
            consensus_trend = int(req.consensus.trend)

        # В базовой версии используем единый порог — flat; позже можно детектировать режим рынка
        threshold_pct = consensus_flat

        predictions: List[SinglePrediction] = []
        labels: List[str] = []
        for mp in models:
            action, confidence, q_values = predictor.predict_single(mp, state)
            predictions.append(SinglePrediction(model_path=mp, action=action, confidence=confidence, q_values=q_values))
            labels.append(action)

        decision, votes, threshold_votes = predictor.vote(labels, threshold_pct)
        return PredictResponse(
            success=True,
            decision=decision,
            votes=votes,
            threshold_used=threshold_votes,
            predictions=predictions
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
