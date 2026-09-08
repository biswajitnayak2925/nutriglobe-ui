"""
comparison.py
-------------
Backend service that takes USER INPUT — a BARCODE, a product NAME, or an
IMAGE of the product (photo of packaging / barcode) — and tells the user
which version (India vs UK) is healthier and by how much. Uses the
pre-computed comparison.csv (India<->UK matched pairs) built by
data_processing.build_database(), with a live embedding-based fallback if
no pre-computed match exists.

Usage from a web framework (single unified endpoint):

    from comparison import ComparisonEngine

    comparer = ComparisonEngine()   # load once at startup

    @app.post("/api/compare")
    def compare(payload: dict):
        # payload["input"] can be a barcode string, a name string, or
        # (for image uploads) raw bytes / a saved temp file path
        return comparer.compare(payload["input"])

Or, explicitly by type:
    comparer.compare_by_barcode("8901058851937")
    comparer.compare_by_query("Maggi noodles")
    comparer.compare_by_image(uploaded_file.file)   # barcode-in-photo or OCR

Do NOT instantiate ComparisonEngine per-request — load once, reuse.
"""

import os
import re
import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from sentence_transformers import util

from input_resolver import resolve_input, InputResolutionError, ImageInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("comparison")

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.getcwd(), "data"))


class ProductNotFoundError(Exception):
    pass


class NoComparableMatchError(Exception):
    pass


class ComparisonEngine:
    """Loads product.csv, comparison.csv and embeddings.pt once, and answers
    'which version is healthier, and by how much' queries."""

    def __init__(self, data_dir: str = DATA_DIR, fallback_min_similarity: float = 0.30):
        self.data_dir = data_dir
        self.fallback_min_similarity = fallback_min_similarity

        product_path = os.path.join(data_dir, "product.csv")
        comparison_path = os.path.join(data_dir, "comparison.csv")
        embeddings_path = os.path.join(data_dir, "embeddings.pt")

        for path in (product_path, comparison_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing '{path}'. Run data_processing.build_database() first.")

        logger.info("Loading product + comparison database from %s ...", data_dir)
        self.product_df = pd.read_csv(product_path)
        for col in ["product_name", "brand", "category", "country"]:
            if col in self.product_df.columns:
                self.product_df[col] = self.product_df[col].fillna("")

        self.comparison_df = pd.read_csv(comparison_path)

        # Embeddings are optional but enable on-the-fly matching when no
        # pre-computed pair exists in comparison.csv
        self.embeddings = None
        if os.path.exists(embeddings_path):
            self.embeddings = torch.load(embeddings_path, weights_only=True)
            if len(self.product_df) != self.embeddings.shape[0]:
                logger.warning("product.csv/embeddings row mismatch — disabling embedding fallback.")
                self.embeddings = None

        self._id_to_idx = {pid: i for i, pid in enumerate(self.product_df["product_id"])}
        # barcode -> product_id, for exact-match lookups (barcode scan / typed barcode)
        self._barcode_to_id = {
            str(bc): pid for bc, pid in zip(self.product_df["barcode"], self.product_df["product_id"])
        }
        self._is_indian_mask = (
            self.product_df["product_id"].str.startswith("IN_").to_numpy()
            | self.product_df["country"].str.lower().str.contains("india", na=False).to_numpy()
        )
        self._is_uk_mask = ~self._is_indian_mask

        logger.info("Loaded %d products, %d pre-computed comparisons.", len(self.product_df), len(self.comparison_df))

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def _get_row(self, product_id: str) -> pd.Series:
        idx = self._id_to_idx.get(product_id)
        if idx is None:
            raise ProductNotFoundError(f"Product ID '{product_id}' not found.")
        return self.product_df.iloc[idx]

    def search_product(self, query: str, country: Optional[str] = None, limit: int = 10) -> pd.DataFrame:
        """Word-based search over brand + product_name — matches products
        containing ALL significant words from the query, in any order. This
        is more forgiving than a single continuous substring match, which
        fails easily on noisy input (e.g. OCR text like 'Image of Kit Kat
        Hazelnut Crunch 2-Pack' won't exactly match a dataset entry like
        'Kitkat Hazelnut', but word-overlap still finds it).
        country: optionally restrict to 'india' or 'united kingdom'."""
        q = (query or "").strip().lower()
        if not q:
            return self.product_df.iloc[0:0]

        # Strip common noise words that show up in OCR captions/filenames
        # but never in real product names.
        stopwords = {
            "image", "photo", "picture", "of", "the", "a", "an", "pack",
            "package", "packet", "product", "label", "front", "back", "showing",
        }
        words = [w for w in re.findall(r"[a-z0-9]+", q) if w not in stopwords and len(w) > 1]
        if not words:
            words = re.findall(r"[a-z0-9]+", q)  # fall back to original if everything got stripped
        if not words:
            return self.product_df.iloc[0:0]

        haystack = (self.product_df["brand"] + " " + self.product_df["product_name"]).str.lower()
        mask = pd.Series(True, index=self.product_df.index)
        for w in words:
            mask &= haystack.str.contains(re.escape(w), na=False)

        if country:
            mask &= self.product_df["country"].str.lower().str.contains(country.lower(), na=False)

        results = self.product_df[mask]

        # If requiring every word matched nothing, relax to "most words" —
        # rank by how many query words each candidate actually contains.
        if results.empty and len(words) > 1:
            scores = pd.Series(0, index=self.product_df.index)
            for w in words:
                scores += haystack.str.contains(re.escape(w), na=False).astype(int)
            candidate_mask = scores > 0
            if country:
                candidate_mask &= self.product_df["country"].str.lower().str.contains(country.lower(), na=False)
            results = self.product_df[candidate_mask].assign(_match_score=scores[candidate_mask]) \
                .sort_values("_match_score", ascending=False).drop(columns="_match_score")

        return results.head(limit).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Core comparison logic
    # ------------------------------------------------------------------
    def _build_result(self, india_row: pd.Series, uk_row: pd.Series, similarity: Optional[float]) -> dict:
        india_score = float(india_row["overall_health_score"])
        uk_score = float(uk_row["overall_health_score"])
        diff = round(india_score - uk_score, 2)  # positive => India healthier

        if india_score == uk_score:
            winner, margin_pct = "Tie", 0.0
        elif india_score > uk_score:
            winner = "India"
            margin_pct = round((india_score - uk_score) / max(uk_score, 1e-6) * 100, 1)
        else:
            winner = "United Kingdom"
            margin_pct = round((uk_score - india_score) / max(india_score, 1e-6) * 100, 1)

        return {
            "india_product": {
                "product_id": india_row["product_id"],
                "name": india_row["product_name"],
                "brand": india_row["brand"],
                "health_score": india_score,
                "nutrition_score": india_row.get("nutrition_score"),
                "additive_score": india_row.get("additive_score"),
                "ingredient_score": india_row.get("ingredient_score"),
                "source_url": india_row.get("source_url"),
            },
            "uk_product": {
                "product_id": uk_row["product_id"],
                "name": uk_row["product_name"],
                "brand": uk_row["brand"],
                "health_score": uk_score,
                "nutrition_score": uk_row.get("nutrition_score"),
                "additive_score": uk_row.get("additive_score"),
                "ingredient_score": uk_row.get("ingredient_score"),
                "source_url": uk_row.get("source_url"),
            },
            "healthier_country": winner,
            "score_difference": abs(diff),          # absolute point gap, 0-100 scale
            "percentage_better": margin_pct,          # % better than the loser's score
            "match_similarity": round(similarity, 4) if similarity is not None else None,
        }

    def compare_by_ids(self, india_product_id: str, uk_product_id: str) -> dict:
        """Directly compare two explicit product IDs (any pairing), regardless
        of whether they appear together in comparison.csv."""
        india_row = self._get_row(india_product_id)
        uk_row = self._get_row(uk_product_id)

        similarity = None
        if self.embeddings is not None:
            idx_a, idx_b = self._id_to_idx[india_product_id], self._id_to_idx[uk_product_id]
            similarity = float(util.cos_sim(self.embeddings[idx_a], self.embeddings[idx_b])[0][0].item())

        return self._build_result(india_row, uk_row, similarity)

    def search_by_barcode(self, barcode: str) -> Optional[pd.Series]:
        """Exact-match lookup by barcode. Returns None if not in the local dataset."""
        product_id = self._barcode_to_id.get(str(barcode).strip())
        return None if product_id is None else self._get_row(product_id)

    def compare_by_query(self, query: str) -> dict:
        """User-typed product NAME. Finds the best-matching product (in
        either country), then resolves its healthier/worse counterpart."""
        candidates = self.search_product(query, limit=1)
        if candidates.empty:
            raise ProductNotFoundError(f"No product found matching '{query}'.")
        return self._find_and_compare(candidates.iloc[0])

    def compare_by_product_id(self, product_id: str) -> dict:
        """Compare a SPECIFIC, already-known product_id (e.g. one the user
        clicked from a search results list) — no re-searching/ambiguity,
        unlike compare_by_query which can match a different near-duplicate
        product with the same name."""
        anchor = self._get_row(product_id)
        return self._find_and_compare(anchor)

    def compare_by_barcode(self, barcode: str) -> dict:
        """User-typed or scanned BARCODE. Exact match against the local
        dataset, then resolves its healthier/worse counterpart."""
        anchor = self.search_by_barcode(barcode)
        if anchor is None:
            raise ProductNotFoundError(
                f"No product with barcode '{barcode}' in the local dataset. "
                "It may exist on Open Food Facts but hasn't been fetched yet."
            )
        return self._find_and_compare(anchor)

    def compare(self, user_input: Union[str, ImageInput]) -> dict:
        """Universal entry point: pass a barcode string, a product name
        string, or an image (path / bytes / file-like / PIL.Image) and this
        routes to the right lookup automatically (barcode -> exact match,
        image -> barcode-in-photo or OCR'd name, text -> name search)."""
        try:
            resolved = resolve_input(user_input)
        except InputResolutionError as e:
            raise NoComparableMatchError(str(e))

        if resolved["type"] == "barcode":
            return self.compare_by_barcode(resolved["value"])
        return self.compare_by_query(resolved["value"])

    # kept as an explicit alias for readability in route handlers
    compare_by_image = compare

    def _find_and_compare(self, anchor: pd.Series) -> dict:
        """Shared logic: given an anchor product row, find its healthier/worse
        counterpart in the other country — first from the pre-computed
        comparison.csv, falling back to a live embedding search."""
        anchor_id = anchor["product_id"]
        is_indian = bool(self._is_indian_mask[self._id_to_idx[anchor_id]])

        # 1) Try the pre-computed comparison table first (fast, curated)
        if is_indian:
            match = self.comparison_df[self.comparison_df["india_product_id"] == anchor_id]
            if not match.empty:
                uk_id = match.iloc[0]["foreign_product_id"]
                similarity = match.iloc[0]["embedding_similarity_score"]
                return self._build_result(self._get_row(anchor_id), self._get_row(uk_id), similarity)
        else:
            match = self.comparison_df[self.comparison_df["foreign_product_id"] == anchor_id]
            if not match.empty:
                india_id = match.iloc[0]["india_product_id"]
                similarity = match.iloc[0]["embedding_similarity_score"]
                return self._build_result(self._get_row(india_id), self._get_row(anchor_id), similarity)

        # 2) Fallback: live nearest-neighbour search across the other country
        if self.embeddings is None:
            raise NoComparableMatchError(
                f"No pre-computed match for '{anchor['product_name']}' and no embeddings available for a live search."
            )

        anchor_idx = self._id_to_idx[anchor_id]
        opposite_mask = self._is_uk_mask if is_indian else self._is_indian_mask
        scores = util.cos_sim(self.embeddings[anchor_idx], self.embeddings)[0].cpu().numpy()
        scores[~opposite_mask] = -1.0  # exclude same-country + self

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < self.fallback_min_similarity:
            raise NoComparableMatchError(
                f"No sufficiently similar product found in the other country for '{anchor['product_name']}'."
            )

        best_row = self.product_df.iloc[best_idx]
        if is_indian:
            return self._build_result(anchor, best_row, best_score)
        return self._build_result(best_row, anchor, best_score)


if __name__ == "__main__":
    comparer = ComparisonEngine()
    if len(comparer.comparison_df) > 0:
        sample_query = comparer.comparison_df.iloc[0]["india_product_name"]
        print(f"Query: {sample_query}")
        print(comparer.compare_by_query(sample_query))
