"""
recommendation.py
------------------
Backend recommendation service. Loads the product database + pre-computed
embeddings ONCE (at process/server startup) and serves fast lookups.

Usage from a web framework (FastAPI/Flask):

    from recommendation import RecommendationEngine

    engine = RecommendationEngine()          # load once at startup

    @app.get("/recommend/{product_id}")
    def recommend(product_id: str):
        return engine.recommend(product_id).to_dict(orient="records")

Do NOT instantiate RecommendationEngine per-request — it loads a
SentenceTransformer-sized embedding matrix and should be a singleton.
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import util

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("recommendation")

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.getcwd(), "data"))


class ProductNotFoundError(Exception):
    pass


class RecommendationEngine:
    """Loads product.csv + embeddings.pt once and answers recommendation queries."""

    def __init__(self, data_dir: str = DATA_DIR, initial_min_similarity: float = 0.40):
        self.data_dir = data_dir
        self.initial_min_similarity = initial_min_similarity

        csv_path = os.path.join(data_dir, "product.csv")
        emb_path = os.path.join(data_dir, "embeddings.pt")
        if not os.path.exists(csv_path) or not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"Missing '{csv_path}' or '{emb_path}'. Run data_processing.build_database() first."
            )

        logger.info("Loading product database and embeddings from %s ...", data_dir)
        self.df = pd.read_csv(csv_path)
        for col in ["product_name", "brand", "category", "country"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("")

        # weights_only=True is the safe/forward-compatible way to load a plain tensor
        self.embeddings = torch.load(emb_path, weights_only=True)

        if len(self.df) != self.embeddings.shape[0]:
            raise ValueError(
                f"product.csv rows ({len(self.df)}) and embeddings ({self.embeddings.shape[0]}) "
                "are out of sync — rebuild the database."
            )

        # Precompute once: which rows are Indian products
        self._is_indian_mask = (
            self.df["product_id"].str.startswith("IN_").to_numpy()
            | self.df["country"].str.lower().str.contains("india", na=False).to_numpy()
        )
        # id -> row index, for O(1) lookup instead of scanning the df each call
        self._id_to_idx = {pid: i for i, pid in enumerate(self.df["product_id"])}

        logger.info("Loaded %d products.", len(self.df))

    def get_product(self, product_id: str) -> Optional[pd.Series]:
        idx = self._id_to_idx.get(product_id)
        return None if idx is None else self.df.iloc[idx]

    def recommend(self, target_product_id: str, top_k: int = 1) -> pd.DataFrame:
        """Returns the top_k healthier Indian alternatives for target_product_id.
        Raises ProductNotFoundError if the id doesn't exist."""
        target_idx = self._id_to_idx.get(target_product_id)
        if target_idx is None:
            raise ProductNotFoundError(f"Product ID '{target_product_id}' not found.")

        target_item = self.df.iloc[target_idx]
        target_health_score = target_item["overall_health_score"]
        target_embedding = self.embeddings[target_idx]

        similarity_scores = util.cos_sim(target_embedding, self.embeddings)[0].cpu().numpy()

        health_mask = self.df["overall_health_score"].to_numpy() > target_health_score
        identity_mask = self.df["product_id"].to_numpy() != target_product_id

        current_threshold = self.initial_min_similarity
        valid_indices = np.array([], dtype=int)
        while current_threshold >= 0.05:
            combined_mask = (
                (similarity_scores >= current_threshold)
                & health_mask & identity_mask & self._is_indian_mask
            )
            valid_indices = np.where(combined_mask)[0]
            if len(valid_indices) > 0:
                break
            current_threshold -= 0.05

        if len(valid_indices) == 0:
            return pd.DataFrame(columns=[
                "recommended_product_id", "product_name", "brand", "country",
                "similarity_score", "health_score", "nutrition_score",
                "additive_score", "ingredient_score", "health_boost",
            ])

        rows = []
        for idx in valid_indices:
            item = self.df.iloc[idx]
            rows.append({
                "recommended_product_id": item["product_id"],
                "product_name": item["product_name"],
                "brand": item["brand"],
                "country": item["country"],
                "similarity_score": round(float(similarity_scores[idx]), 4),
                "health_score": item["overall_health_score"],
                "nutrition_score": item.get("nutrition_score"),
                "additive_score": item.get("additive_score"),
                "ingredient_score": item.get("ingredient_score"),
                "health_boost": round(item["overall_health_score"] - target_health_score, 2),
            })

        results_df = pd.DataFrame(rows)
        # Rank by SIMILARITY first (genuinely comparable products), then by
        # health improvement as the tiebreaker. Sorting by health_boost first
        # would surface the globally healthiest products for every query
        # regardless of how similar they actually are to the target.
        return results_df.sort_values(
            by=["similarity_score", "health_boost"], ascending=[False, False]
        ).head(top_k).reset_index(drop=True)


if __name__ == "__main__":
    # Quick manual smoke test
    engine = RecommendationEngine()
    sample_id = engine.df.iloc[0]["product_id"]
    print(f"Target: {sample_id}")
    print(engine.recommend(sample_id))
