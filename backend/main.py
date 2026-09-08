

import logging
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from data_processing import build_database, add_product_by_barcode
from recommendation import RecommendationEngine, ProductNotFoundError as RecoNotFoundError
from comparison import ComparisonEngine, ProductNotFoundError as CompareNotFoundError, NoComparableMatchError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")

app = FastAPI(title="India vs UK Food Health Comparator", version="1.0")


_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
_allowed_origins = ["*"] if _origins_env.strip() == "*" else [o.strip() for o in _origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_reco_engine: Optional[RecommendationEngine] = None
_compare_engine: Optional[ComparisonEngine] = None


def _load_engines() -> None:
    global _reco_engine, _compare_engine
    try:
        _reco_engine = RecommendationEngine()
        logger.info("Recommendation engine loaded.")
    except FileNotFoundError:
        _reco_engine = None
        logger.warning("No dataset found yet for recommendations — call POST /api/build.")

    try:
        _compare_engine = ComparisonEngine()
        logger.info("Comparison engine loaded.")
    except FileNotFoundError:
        _compare_engine = None
        logger.warning("No dataset found yet for comparisons — call POST /api/build.")


@app.on_event("startup")
def startup() -> None:
    _load_engines()


def _require_reco() -> RecommendationEngine:
    if _reco_engine is None:
        raise HTTPException(503, "Dataset not built yet. POST /api/build first.")
    return _reco_engine


def _require_compare() -> ComparisonEngine:
    if _compare_engine is None:
        raise HTTPException(503, "Dataset not built yet. POST /api/build first.")
    return _compare_engine


def _clean(obj):
 
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj



@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "recommendation_ready": _reco_engine is not None,
        "comparison_ready": _compare_engine is not None,
    }


@app.post("/api/build")
def build():
    
    summary = build_database()
    _load_engines()
    return summary


@app.post("/api/add-product")
def add_product(barcode: str):
   
    try:
        result = add_product_by_barcode(barcode)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    _load_engines()  
    return result


@app.get("/api/search")
def search(q: str, country: Optional[str] = None, limit: int = 10):
    engine = _require_compare()
    results = engine.search_product(q, country=country, limit=limit)
    return _clean(results.to_dict(orient="records"))


@app.get("/api/recommend/{product_id}")
def recommend(product_id: str, top_k: int = 1):
    engine = _require_reco()
    try:
        df = engine.recommend(product_id, top_k=top_k)
    except RecoNotFoundError as e:
        raise HTTPException(404, str(e))
    return _clean(df.to_dict(orient="records"))


@app.post("/api/compare")
async def compare(
    input: Optional[str] = Form(None),
    product_id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):

    engine = _require_compare()
    if not input and not image and not product_id:
        raise HTTPException(400, "Provide 'product_id', 'input' (barcode/name), or an 'image' file.")

    try:
        if product_id:
            result = engine.compare_by_product_id(product_id)
        elif image is not None:
            image_bytes = await image.read()
            result = engine.compare(image_bytes)
        else:
            result = engine.compare(input)
    except (CompareNotFoundError, NoComparableMatchError) as e:
        raise HTTPException(404, str(e))

    return _clean(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
