
```
backend/
  main.py              FastAPI app (routes)
  data_processing.py   Fetch + score OFF data, build embeddings (offline job)
  recommendation.py    RecommendationEngine (healthier Indian alternatives)
  comparison.py        ComparisonEngine (India vs UK, by name/barcode/image)
  input_resolver.py    Barcode/OCR image resolution helper
  requirements.txt
frontend/
  index.html
  style.css
  app.js
```

 1. Backend setup
```bash
cd backend
python -m venv venv && source venv/bin/activate   
pip install -r requirements.txt

# System deps for image-based comparison (optional):
#   Ubuntu/Debian: sudo apt-get install libzbar0 tesseract-ocr
#   macOS:         brew install zbar tesseract

export DATA_DIR=./data
uvicorn main:app --reload --port 8000
```

2. Build the dataset (run once, before using the app)
```bash
curl -X POST http://localhost:8000/api/build
```
This fetches product data from Open Food Facts and computes embeddings —
takes a few minutes. Re-run it any time you want to refresh the data.
Check readiness anytime with:
```bash
curl http://localhost:8000/api/health
```

3. Frontend
No build step — just open `frontend/index.html` directly in a browser,
or serve it:
```bash
cd frontend
python -m http.server 5500
```
Then visit `http://localhost:5500`. `app.js` points at
`API_BASE = "http://localhost:8000"` — change this if your backend runs
elsewhere (e.g. after deploying).

API summary
| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | check if dataset is loaded |
| POST | `/api/build` | (re)build dataset from Open Food Facts |
| GET | `/api/search?q=...` | search products by name |
| GET | `/api/recommend/{product_id}` | top-3 healthier Indian alternatives |
| POST | `/api/compare` | form field `input` (barcode/name) OR file `image` |
