"""
data_processing.py
-------------------
Backend module that builds the product/ingredient/nutrition/additive database
from Open Food Facts and pre-computes sentence embeddings for the
recommendation engine.

Usage (as a one-off / scheduled job, e.g. via cron or a "refresh" endpoint):

    from data_processing import build_database
    build_database()

This is intended to be run OFFLINE (periodically), not on every web request.
The recommendation engine (recommendation.py) only ever reads the CSV +
embeddings.pt files this produces.
"""

import os
import re
import time
import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests
import torch
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_processing")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(2)

# ==========================================
# CONFIG (env-var driven so it works outside Kaggle)
# ==========================================
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.getcwd(), "data"))
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

HEADERS = {
    "User-Agent": (
        "SIH-UK-India-Comparison/6.1 (research/student project; contact:"
        " student@example.com)"
    )
}
SEARCH_URL = "https://world.openfoodfacts.org/api/v2/search"

TARGET_CATEGORIES = [
    "en:snacks", "en:beverages", "en:biscuits", "en:breakfast-cereals",
    "en:chocolates", "en:sweet-snacks", "en:sauces", "en:instant-noodles",
    "en:dairy", "en:groceries",
    "en:carbonated-drinks", "en:sodas", "en:colas", "en:fruit-juices",
    "en:energy-drinks",
    # Added: broader coverage across common grocery aisles
    "en:spreads", "en:condiments", "en:frozen-foods", "en:ready-meals",
    "en:bakery-products", "en:breads", "en:pastries", "en:teas", "en:coffees",
    "en:spices", "en:pastas", "en:rices", "en:pickles", "en:jams",
    "en:yogurts", "en:cheeses", "en:ice-creams", "en:cereal-bars",
    "en:baby-foods", "en:canned-foods", "en:soups", "en:vinegars",
]

# Well-known brands sold in BOTH India and the UK — fetching by brand
# catches products that Open Food Facts miscategorized or left uncategorized
# (this is how the Coca-Cola gap was found), and improves match quality
# since same-brand products embed/compare more meaningfully.
BRAND_TARGETS = [
    "coca-cola", "pepsi", "nestle", "cadbury", "britannia", "parle",
    "kelloggs", "heinz", "walkers", "mcvities", "unilever", "itc",
    "amul", "haldirams", "maggi", "lays", "kraft-heinz", "mars",
    "ferrero", "danone", "lipton", "red-bull", "sprite", "fanta",
]

FIELDS = ",".join([
    "code", "product_name", "brands", "categories", "categories_tags",
    "ingredients_text", "ingredients", "nutriments", "additives_tags", "url",
])

MAX_PAGES_PER_CATEGORY = int(os.environ.get("MAX_PAGES_PER_CATEGORY", 5))
PAGE_SIZE = int(os.environ.get("PAGE_SIZE", 1000))  # OFF's documented max page size
# Open Food Facts enforces a HARD limit of 10 search requests/minute/IP.
# 6.5s keeps us safely under that (~9.2 req/min) — do not lower this without
# reading https://openfoodfacts.github.io/documentation/docs/Product-Opener/api/
SLEEP_BETWEEN_REQUESTS = float(os.environ.get("SLEEP_BETWEEN_REQUESTS", 6.5))
SIMILARITY_THRESHOLD = 0.45

UPF_KEYWORDS = [
    "palm oil", "high fructose", "invert sugar", "hydrogenated",
    "maltodextrin", "dextrose", "glucose syrup", "flavour enhancer",
    "flavor enhancer", "modified starch", "emulsifier", "artificial",
]
BENEFICIAL_KEYWORDS = [
    "whole grain", "whole wheat", "oat", "almond", "walnut",
    "pulse", "lentil", "chickpea", "fruit", "vegetable",
]
ADDITIVE_HAZARD_MAP = {
    "E102": {"score": 25, "risk": "High", "name": "Tartrazine"},
    "E110": {"score": 25, "risk": "High", "name": "Sunset Yellow"},
    "E122": {"score": 25, "risk": "High", "name": "Azorubine"},
    "E124": {"score": 25, "risk": "High", "name": "Ponceau 4R"},
    "E129": {"score": 25, "risk": "High", "name": "Allura Red AC"},
    "E133": {"score": 75, "risk": "Medium", "name": "Brilliant Blue FCF"},
    "E211": {"score": 25, "risk": "High", "name": "Sodium Benzoate"},
    "E250": {"score": 25, "risk": "High", "name": "Sodium Nitrite"},
    "E320": {"score": 25, "risk": "High", "name": "BHA"},
    "E330": {"score": 100, "risk": "Low", "name": "Citric Acid"},
    "E322": {"score": 100, "risk": "Low", "name": "Lecithin"},
    "E412": {"score": 100, "risk": "Low", "name": "Guar Gum"},
    "E415": {"score": 100, "risk": "Low", "name": "Xanthan Gum"},
}
# matches "E330", "e330i", "en:e330-citric-acid" style tags -> base E-number
_E_CODE_RE = re.compile(r"E(\d{3,4})", re.IGNORECASE)


# ==========================================
# SCORING HELPERS
# ==========================================
def calculate_uk_npm_score(nutriments: dict) -> float:
    """Custom 0-100 health index INSPIRED BY the UK Nutrient Profiling Model.
    NOTE: this is a simplified/rescaled heuristic, not FSA's official
    NPM-to-traffic-light output — label it as such in the UI."""
    if not nutriments:
        return 50.0
    energy_kj = nutriments.get("energy-kj_100g") or (nutriments.get("energy-kcal_100g", 0) * 4.184)
    sat_fat = nutriments.get("saturated-fat_100g", 0) or 0
    sugars = nutriments.get("sugars_100g", 0) or 0
    sodium_mg = (nutriments.get("sodium_100g") or 0) * 1000
    protein = nutriments.get("proteins_100g", 0) or 0
    fibre = nutriments.get("fiber_100g", 0) or 0

    a_energy = pd.cut([energy_kj], bins=[-np.inf, 335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350, np.inf], labels=range(11))[0]
    a_sat_fat = pd.cut([sat_fat], bins=[-np.inf, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf], labels=range(11))[0]
    a_sugars = pd.cut([sugars], bins=[-np.inf, 4.5, 9, 13.5, 18, 22.5, 27, 31.5, 36, 40.5, 45, np.inf], labels=range(11))[0]
    a_sodium = pd.cut([sodium_mg], bins=[-np.inf, 90, 180, 270, 360, 450, 540, 630, 720, 810, 900, np.inf], labels=range(11))[0]
    c_protein = pd.cut([protein], bins=[-np.inf, 1.6, 3.2, 4.8, 6.4, 8.0, np.inf], labels=range(6))[0]
    c_fibre = pd.cut([fibre], bins=[-np.inf, 0.9, 1.9, 2.8, 3.7, 4.7, np.inf], labels=range(6))[0]

    a_points = int(a_energy) + int(a_sat_fat) + int(a_sugars) + int(a_sodium)
    c_points = int(c_protein) + int(c_fibre)
    raw_npm = a_points - c_points if a_points < 11 else a_points - int(c_fibre)
    return round(max(0, min(100, 100 - ((raw_npm + 15) / 55.0) * 100)), 2)


def score_ingredient(name: Optional[str], rank: int) -> tuple:
    if not name:
        return 50.0, "Neutral"
    name_lower = name.lower()
    weight = 1.0 / np.sqrt(max(1, rank))
    if any(k in name_lower for k in UPF_KEYWORDS):
        return round(max(0, 20 + (15 * (1 - weight))), 2), "Ultra-Processed"
    elif any(k in name_lower for k in BENEFICIAL_KEYWORDS):
        return round(min(100, 75 + (25 * weight)), 2), "Beneficial"
    return 75.0, "Neutral"


def score_additive(tag: str) -> tuple:
    """Robust to suffixed/prefixed OFF tags like 'en:e330i' or 'en:e330-citric-acid'."""
    match = _E_CODE_RE.search(tag or "")
    code = f"E{match.group(1)}" if match else (tag or "UNKNOWN").replace("en:", "").upper()
    info = ADDITIVE_HAZARD_MAP.get(code, {"score": 100, "risk": "Low / Unclassified", "name": code})
    return code, info["name"], info["score"], info["risk"]


# ==========================================
# FETCHING
# ==========================================
def fetch_category_products(category_tag: str, country_tag: str, session: requests.Session) -> list:
    category_items = []
    for page in range(1, MAX_PAGES_PER_CATEGORY + 1):
        params = {
            "categories_tags": category_tag,
            "countries_tags_en": country_tag,
            "fields": FIELDS,
            "page_size": PAGE_SIZE,
            "page": page,
        }
        try:
            resp = session.get(SEARCH_URL, params=params, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                logger.warning("Rate limited on %s/%s page %s — backing off 30s.", category_tag, country_tag, page)
                time.sleep(30)
                resp = session.get(SEARCH_URL, params=params, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                products = resp.json().get("products", [])
                valid = [p for p in products if p.get("code") and (p.get("ingredients") or p.get("nutriments"))]
                category_items.extend(valid)
                if len(products) < PAGE_SIZE:
                    break
            else:
                logger.warning("Non-200 (%s) for %s/%s page %s", resp.status_code, category_tag, country_tag, page)
                break
            time.sleep(SLEEP_BETWEEN_REQUESTS)
        except requests.RequestException as e:
            logger.error("Request failed for %s (%s) page %s: %s", category_tag, country_tag, page, e)
            break
    return category_items


def fetch_brand_products(brand_tag: str, country_tag: str, session: requests.Session) -> list:
    """Same as fetch_category_products but filters by brand instead of
    category — catches products Open Food Facts miscategorized or left
    without a matching category tag."""
    brand_items = []
    for page in range(1, MAX_PAGES_PER_CATEGORY + 1):
        params = {
            "brands_tags": brand_tag,
            "countries_tags_en": country_tag,
            "fields": FIELDS,
            "page_size": PAGE_SIZE,
            "page": page,
        }
        try:
            resp = session.get(SEARCH_URL, params=params, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                logger.warning("Rate limited on brand %s/%s page %s — backing off 30s.", brand_tag, country_tag, page)
                time.sleep(30)
                resp = session.get(SEARCH_URL, params=params, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                products = resp.json().get("products", [])
                valid = [p for p in products if p.get("code") and (p.get("ingredients") or p.get("nutriments"))]
                brand_items.extend(valid)
                if len(products) < PAGE_SIZE:
                    break
            else:
                logger.warning("Non-200 (%s) for brand %s/%s page %s", resp.status_code, brand_tag, country_tag, page)
                break
            time.sleep(SLEEP_BETWEEN_REQUESTS)
        except requests.RequestException as e:
            logger.error("Request failed for brand %s (%s) page %s: %s", brand_tag, country_tag, page, e)
            break
    return brand_items


# ==========================================
# MAIN BUILD FUNCTION
# ==========================================
def build_database(out_dir: str = DATA_DIR, model_name: str = EMBEDDING_MODEL_NAME) -> dict:
    """Fetches OFF data for India & UK, scores every product, computes
    embeddings once, and writes CSVs + embeddings.pt to out_dir.

    Returns a small summary dict (counts), useful for a "rebuild" API endpoint.
    """
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Step 1: Querying Open Food Facts (India & UK)...")
    with requests.Session() as session:
        india_dict = {p["code"]: p for cat in TARGET_CATEGORIES for p in fetch_category_products(cat, "india", session)}
        uk_dict = {p["code"]: p for cat in TARGET_CATEGORIES for p in fetch_category_products(cat, "united-kingdom", session)}

        logger.info("Step 1b: Querying by brand (fills category-tagging gaps)...")
        for brand in BRAND_TARGETS:
            for p in fetch_brand_products(brand, "india", session):
                india_dict.setdefault(p["code"], p)
            for p in fetch_brand_products(brand, "united-kingdom", session):
                uk_dict.setdefault(p["code"], p)

    logger.info("Fetched Products: India=%d | UK=%d", len(india_dict), len(uk_dict))

    product_rows, ingredient_rows, nutrition_rows, additive_rows = [], [], [], []
    processed_cache = set()
    counters = {"ing_id": 1, "nut_id": 1, "add_id": 1}

    def process_product(p: dict, country: str, prefix: str):
        code = p["code"]
        product_id = f"{prefix}_{code}"
        if product_id in processed_cache:
            return
        processed_cache.add(product_id)

        npm_score = calculate_uk_npm_score(p.get("nutriments") or {})

        ing_scores = []
        for rank, ing in enumerate(p.get("ingredients") or [], start=1):
            score, cls = score_ingredient(ing.get("text"), rank)
            ing_scores.append(score)
            ingredient_rows.append({
                "ingredient_id": counters["ing_id"], "product_id": product_id,
                "ingredient_name": ing.get("text"), "sequence_rank": rank,
                "classification": cls, "ingredient_score": score,
            })
            counters["ing_id"] += 1
        avg_ing = float(np.mean(ing_scores)) if ing_scores else 70.0

        nutriments = p.get("nutriments") or {}
        nutrition_rows.append({
            "nutrition_id": counters["nut_id"], "product_id": product_id, "basis": "per_100g",
            "energy_kcal": nutriments.get("energy-kcal_100g"), "protein_g": nutriments.get("proteins_100g"),
            "fat_g": nutriments.get("fat_100g"), "saturated_fat_g": nutriments.get("saturated-fat_100g"),
            "sugars_g": nutriments.get("sugars_100g"), "fiber_g": nutriments.get("fiber_100g"),
            "sodium_mg": (nutriments.get("sodium_100g") or 0) * 1000, "uk_npm_nutrition_score": npm_score,
        })
        counters["nut_id"] += 1

        add_scores = []
        for tag in (p.get("additives_tags") or []):
            acode, aname, ascore, arisk = score_additive(tag)
            add_scores.append(ascore)
            additive_rows.append({
                "additive_id": counters["add_id"], "product_id": product_id, "additive_code": acode,
                "additive_name": aname, "risk_level": arisk, "additive_safety_score": ascore,
            })
            counters["add_id"] += 1

        overall = (
            round((0.50 * npm_score) + (0.30 * np.mean(add_scores)) + (0.20 * avg_ing), 2)
            if add_scores else round((0.70 * npm_score) + (0.30 * avg_ing), 2)
        )
        additive_score = round(float(np.mean(add_scores)), 2) if add_scores else None
        product_rows.append({
            "product_id": product_id, "barcode": code, "brand": p.get("brands") or "Generic",
            "product_name": p.get("product_name") or "Unknown Product", "category": p.get("categories") or "",
            "country": country, "overall_health_score": overall,
            "nutrition_score": npm_score, "additive_score": additive_score, "ingredient_score": round(avg_ing, 2),
            "source_url": p.get("url") or f"https://world.openfoodfacts.org/product/{code}",
        })

    for p in india_dict.values():
        process_product(p, "India", "IN")
    for p in uk_dict.values():
        process_product(p, "United Kingdom", "UK")

    product_df = pd.DataFrame(product_rows)
    if product_df.empty:
        raise RuntimeError("No products fetched from Open Food Facts — check network/API availability.")

    india_mask = product_df["country"] == "India"
    uk_mask = product_df["country"] == "United Kingdom"

    # ==========================================
    # EMBEDDINGS (computed ONCE, then sliced — no duplicate encoding)
    # ==========================================
    logger.info("Step 2: Generating embeddings...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    all_texts = (
        product_df["brand"].fillna("") + " " +
        product_df["product_name"].fillna("") + " " +
        product_df["category"].fillna("")
    ).tolist()
    all_embeddings = model.encode(all_texts, batch_size=32, convert_to_tensor=True, show_progress_bar=False, device=device)

    india_idx = np.where(india_mask.to_numpy())[0]
    uk_idx = np.where(uk_mask.to_numpy())[0]
    india_emb = all_embeddings[india_idx]
    uk_emb = all_embeddings[uk_idx]

    india_df = product_df.iloc[india_idx].reset_index(drop=True)
    uk_df = product_df.iloc[uk_idx].reset_index(drop=True)

    similarity_matrix = util.cos_sim(india_emb, uk_emb)
    comparison_rows = []
    comp_id = 1
    for row_pos in range(india_df.shape[0]):
        scores = similarity_matrix[row_pos]
        best_match_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_match_idx].item())
        if best_score >= SIMILARITY_THRESHOLD:
            in_row = india_df.iloc[row_pos]
            uk_row = uk_df.iloc[best_match_idx]
            comparison_rows.append({
                "comparison_id": comp_id,
                "india_product_id": in_row["product_id"], "india_product_name": in_row["product_name"],
                "foreign_product_id": uk_row["product_id"], "foreign_product_name": uk_row["product_name"],
                "embedding_similarity_score": round(best_score, 4),
            })
            comp_id += 1

    comparison_df = pd.DataFrame(comparison_rows)

    # ==========================================
    # SAVE
    # ==========================================
    logger.info("Step 3: Saving files to %s", out_dir)
    product_df.to_csv(os.path.join(out_dir, "product.csv"), index=False)
    pd.DataFrame(ingredient_rows).to_csv(os.path.join(out_dir, "ingredient.csv"), index=False)
    pd.DataFrame(nutrition_rows).to_csv(os.path.join(out_dir, "nutrition.csv"), index=False)
    pd.DataFrame(additive_rows).to_csv(os.path.join(out_dir, "additives.csv"), index=False)
    comparison_df.to_csv(os.path.join(out_dir, "comparison.csv"), index=False)
    torch.save(all_embeddings.cpu(), os.path.join(out_dir, "embeddings.pt"))

    summary = {
        "products": len(product_df), "india_products": len(india_df), "uk_products": len(uk_df),
        "comparisons": len(comparison_df), "out_dir": out_dir,
    }
    logger.info("Done: %s", summary)
    return summary


if __name__ == "__main__":
    build_database()


# ==========================================
# ADD A SINGLE MISSING PRODUCT (no full rebuild needed)
# ==========================================
SINGLE_PRODUCT_URL = "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
_model_singleton = None


def _get_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Reuses one loaded model across calls instead of reloading it every time."""
    global _model_singleton
    if _model_singleton is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_singleton = SentenceTransformer(model_name, device=device)
    return _model_singleton


def _append_csv(path: str, rows: list) -> None:
    df_new = pd.DataFrame(rows)
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        pd.concat([df_existing, df_new], ignore_index=True).to_csv(path, index=False)
    else:
        df_new.to_csv(path, index=False)


def _next_id(path: str, id_col: str) -> int:
    if not os.path.exists(path):
        return 1
    df = pd.read_csv(path)
    return int(df[id_col].max()) + 1 if len(df) else 1


def add_product_by_barcode(barcode: str, out_dir: str = DATA_DIR) -> dict:
    """Fetches ONE product by barcode from Open Food Facts and appends it to
    the existing dataset (product/ingredient/nutrition/additives CSVs +
    embeddings.pt) — without re-running the full category-based build.

    Use this to backfill well-known products that the category search
    missed (e.g. a specific Coca-Cola barcode).

    Raises FileNotFoundError if build_database() hasn't been run yet,
    ValueError if the barcode doesn't exist on OFF or isn't tagged as sold
    in India/UK.
    """
    product_path = os.path.join(out_dir, "product.csv")
    emb_path = os.path.join(out_dir, "embeddings.pt")
    if not os.path.exists(product_path) or not os.path.exists(emb_path):
        raise FileNotFoundError("No existing dataset found — run build_database() first.")

    resp = requests.get(SINGLE_PRODUCT_URL.format(barcode=barcode), headers=HEADERS, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != 1:
        raise ValueError(f"Product with barcode '{barcode}' not found on Open Food Facts.")

    p = payload["product"]
    p["code"] = p.get("code", barcode)

    countries_tags = [c.lower() for c in (p.get("countries_tags") or [])]
    if "en:india" in countries_tags:
        country, prefix = "India", "IN"
    elif "en:united-kingdom" in countries_tags:
        country, prefix = "United Kingdom", "UK"
    else:
        raise ValueError(
            f"Product '{barcode}' isn't tagged as sold in India or the UK on Open Food "
            "Facts, so it can't be classified into this dataset."
        )

    product_id = f"{prefix}_{p['code']}"
    product_df = pd.read_csv(product_path)
    if product_id in product_df["product_id"].values:
        return {"status": "already_exists", "product_id": product_id}

    # --- score it exactly like build_database() does ---
    npm_score = calculate_uk_npm_score(p.get("nutriments") or {})

    ingredient_rows, ing_scores = [], []
    ing_id = _next_id(os.path.join(out_dir, "ingredient.csv"), "ingredient_id")
    for rank, ing in enumerate(p.get("ingredients") or [], start=1):
        score, cls = score_ingredient(ing.get("text"), rank)
        ing_scores.append(score)
        ingredient_rows.append({
            "ingredient_id": ing_id, "product_id": product_id, "ingredient_name": ing.get("text"),
            "sequence_rank": rank, "classification": cls, "ingredient_score": score,
        })
        ing_id += 1
    avg_ing = float(np.mean(ing_scores)) if ing_scores else 70.0

    nutriments = p.get("nutriments") or {}
    nutrition_row = {
        "nutrition_id": _next_id(os.path.join(out_dir, "nutrition.csv"), "nutrition_id"),
        "product_id": product_id, "basis": "per_100g",
        "energy_kcal": nutriments.get("energy-kcal_100g"), "protein_g": nutriments.get("proteins_100g"),
        "fat_g": nutriments.get("fat_100g"), "saturated_fat_g": nutriments.get("saturated-fat_100g"),
        "sugars_g": nutriments.get("sugars_100g"), "fiber_g": nutriments.get("fiber_100g"),
        "sodium_mg": (nutriments.get("sodium_100g") or 0) * 1000, "uk_npm_nutrition_score": npm_score,
    }

    additive_rows, add_scores = [], []
    add_id = _next_id(os.path.join(out_dir, "additives.csv"), "additive_id")
    for tag in (p.get("additives_tags") or []):
        acode, aname, ascore, arisk = score_additive(tag)
        add_scores.append(ascore)
        additive_rows.append({
            "additive_id": add_id, "product_id": product_id, "additive_code": acode,
            "additive_name": aname, "risk_level": arisk, "additive_safety_score": ascore,
        })
        add_id += 1

    overall = (
        round((0.50 * npm_score) + (0.30 * np.mean(add_scores)) + (0.20 * avg_ing), 2)
        if add_scores else round((0.70 * npm_score) + (0.30 * avg_ing), 2)
    )
    additive_score = round(float(np.mean(add_scores)), 2) if add_scores else None
    product_row = {
        "product_id": product_id, "barcode": p["code"], "brand": p.get("brands") or "Generic",
        "product_name": p.get("product_name") or "Unknown Product", "category": p.get("categories") or "",
        "country": country, "overall_health_score": overall,
        "nutrition_score": npm_score, "additive_score": additive_score, "ingredient_score": round(avg_ing, 2),
        "source_url": p.get("url") or f"https://world.openfoodfacts.org/product/{p['code']}",
    }

    # --- append to CSVs ---
    pd.concat([product_df, pd.DataFrame([product_row])], ignore_index=True).to_csv(product_path, index=False)
    if ingredient_rows:
        _append_csv(os.path.join(out_dir, "ingredient.csv"), ingredient_rows)
    _append_csv(os.path.join(out_dir, "nutrition.csv"), [nutrition_row])
    if additive_rows:
        _append_csv(os.path.join(out_dir, "additives.csv"), additive_rows)

    # --- append its embedding ---
    embeddings = torch.load(emb_path, weights_only=True)
    model = _get_model()
    text = f"{product_row['brand']} {product_row['product_name']} {product_row['category']}"
    new_emb = model.encode([text], convert_to_tensor=True).cpu()
    torch.save(torch.cat([embeddings, new_emb], dim=0), emb_path)

    logger.info("Added product %s (%s).", product_id, product_row["product_name"])
    return {
        "status": "added", "product_id": product_id,
        "product_name": product_row["product_name"], "overall_health_score": overall,
    }
