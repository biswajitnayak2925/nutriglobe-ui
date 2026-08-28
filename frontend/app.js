/* ==========================================================================
   NutriGlobe — Frontend Application Controller (app.js)
   Wired to the real backend: GET /api/health, GET /api/search,
   POST /api/compare (multipart: product_id | input | image),
   GET /api/recommend/{product_id}
   ========================================================================== */

const CONFIG = {
  API_BASE_URL: 'http://127.0.0.1:8000/api'
};

// DOM Element Selectors
const elements = {
  backendStatus: document.getElementById('backend-status'),
  errorBanner: document.getElementById('error-banner'),

  // Search Controls
  productSearchInput: document.getElementById('product-search'),
  searchBtn: document.getElementById('search-btn'),
  searchResults: document.getElementById('search-results'),

  // Barcode Controls
  barcodeInput: document.getElementById('barcode-input'),
  barcodeBtn: document.getElementById('barcode-btn'),

  // Photo Controls
  fileInput: document.getElementById('file-input'),
  uploadBtn: document.getElementById('upload-btn'),

  // Dynamic Outputs
  productTitle: document.getElementById('product-title'),
  compareGrid: document.getElementById('compare-grid'),
  verdictContainer: document.getElementById('verdict-container'),
  recommendationsContainer: document.getElementById('recommendations-container')
};

// Initialize App Listeners
document.addEventListener('DOMContentLoaded', () => {
  checkBackendStatus();
  attachEventListeners();
});

function attachEventListeners() {
  elements.searchBtn.addEventListener('click', handleProductSearch);
  elements.productSearchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleProductSearch();
  });

  elements.barcodeBtn.addEventListener('click', handleBarcodeSearch);
  elements.barcodeInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleBarcodeSearch();
  });

  elements.uploadBtn.addEventListener('click', handlePhotoUpload);
}

/* ==========================================================================
   API Communication Handlers
   ========================================================================== */

async function checkBackendStatus() {
  try {
    const res = await fetch(`${CONFIG.API_BASE_URL}/health`);
    if (!res.ok) throw new Error('unreachable');
    const data = await res.json();

    if (data.recommendation_ready && data.comparison_ready) {
      elements.backendStatus.textContent = 'Backend ready ✓';
      elements.backendStatus.className = 'status ok';
    } else {
      elements.backendStatus.textContent = 'Backend up, dataset not built yet';
      elements.backendStatus.className = 'status error';
    }
  } catch (err) {
    elements.backendStatus.textContent = 'Cannot reach backend';
    elements.backendStatus.className = 'status error';
  }
}

// 1. Text search
async function handleProductSearch() {
  const query = elements.productSearchInput.value.trim();
  if (!query) return showError('Please enter a product name to search.');

  clearError();
  setLoadingState(true);

  try {
    const res = await fetch(`${CONFIG.API_BASE_URL}/search?q=${encodeURIComponent(query)}&limit=10`);
    const data = await res.json();

    if (!res.ok) throw new Error(extractErrorMessage(data, 'Product search failed.'));

    renderSearchResults(data); // /api/search returns a plain array
  } catch (err) {
    showError(err.message);
  } finally {
    setLoadingState(false);
  }
}

// 2. Barcode compare
async function handleBarcodeSearch() {
  const barcode = elements.barcodeInput.value.trim();
  if (!barcode) return showError('Please enter a valid barcode.');

  clearError();
  setLoadingState(true);
  try {
    await performCompare({ input: barcode });
  } catch (err) {
    showError(err.message);
  } finally {
    setLoadingState(false);
  }
}

// 3. Photo compare
async function handlePhotoUpload() {
  const file = elements.fileInput.files[0];
  if (!file) return showError('Please select a photo file to upload.');

  clearError();
  setLoadingState(true);
  try {
    await performCompare({ image: file });
  } catch (err) {
    showError(err.message);
  } finally {
    setLoadingState(false);
  }
}

// Unified comparison call — backend takes ONE of product_id / input / image
// as multipart form fields on POST /api/compare.
async function performCompare({ productId, input, image }) {
  const formData = new FormData();
  if (productId) formData.append('product_id', productId);
  if (input) formData.append('input', input);
  if (image) formData.append('image', image);

  const res = await fetch(`${CONFIG.API_BASE_URL}/compare`, { method: 'POST', body: formData });
  const data = await res.json();

  if (!res.ok) throw new Error(extractErrorMessage(data, 'Comparison failed.'));

  elements.productTitle.textContent = `${data.india_product.name} vs ${data.uk_product.name}`;
  renderComparison(data);

  // Auto-fetch a healthier Indian alternative to populate the "swaps" section
  fetchRecommendation(data.india_product.product_id);
}

async function fetchRecommendation(indiaProductId) {
  try {
    const res = await fetch(`${CONFIG.API_BASE_URL}/recommend/${encodeURIComponent(indiaProductId)}?top_k=1`);
    if (!res.ok) {
      elements.recommendationsContainer.classList.add('hidden');
      return;
    }
    const results = await res.json();
    renderRecommendation(results);
  } catch (err) {
    elements.recommendationsContainer.classList.add('hidden');
  }
}

/* ==========================================================================
   DOM Rendering & UI Updates
   ========================================================================== */

function renderSearchResults(items) {
  elements.searchResults.innerHTML = '';

  if (!items || !items.length) {
    elements.searchResults.innerHTML = '<div class="result-item" style="padding: 12px; color: var(--muted);"><span>No matching products found.</span></div>';
    return;
  }

  items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'result-item';

    div.innerHTML = `
      <div>
        <h4 style="margin:0 0 4px;">${item.product_name}</h4>
        <span style="color:var(--muted); font-size:12px;">
          ${item.brand} · ${item.country} · Overall ${fmt(item.overall_health_score)}
        </span>
      </div>
      <button class="select-btn">Compare</button>
    `;

    div.querySelector('.select-btn').addEventListener('click', async () => {
      clearError();
      setLoadingState(true);
      try {
        await performCompare({ productId: item.product_id });
      } catch (err) {
        showError(err.message);
      } finally {
        setLoadingState(false);
      }
    });

    elements.searchResults.appendChild(div);
  });
}

function renderComparison(data) {
  elements.compareGrid.innerHTML = '';

  const indiaWins = data.healthier_country === 'India';
  const ukWins = data.healthier_country === 'United Kingdom';

  elements.compareGrid.appendChild(buildProductCard('🇮🇳', data.india_product, indiaWins));
  elements.compareGrid.appendChild(buildProductCard('🇬🇧', data.uk_product, ukWins));
  elements.compareGrid.appendChild(buildScoreCard(data));

  renderVerdict(data);

  const analysisSection = document.getElementById('analysis-section');
  if (analysisSection) analysisSection.scrollIntoView({ behavior: 'smooth' });
}

function buildProductCard(flag, product, isWinner) {
  const card = document.createElement('article');
  card.className = 'product-card';
  card.innerHTML = `
    <div class="card-head">
      <div class="region"><span class="flag">${flag}</span> ${product.brand || ''}</div>
      <span class="card-meta">${isWinner ? 'Healthier pick' : ''}</span>
    </div>
    <h3 class="product-name" style="margin-top:10px; font-weight:800;">${product.name}</h3>
    <p style="color:var(--muted); font-size:12px; margin-bottom:12px;">Overall score: ${fmt(product.health_score)}/100</p>
    <div class="ingredient-list">
      <span class="pill">Nutrition: ${fmt(product.nutrition_score)}</span>
      <span class="pill">Additive: ${fmt(product.additive_score)}</span>
      <span class="pill">Ingredient: ${fmt(product.ingredient_score)}</span>
    </div>
  `;
  return card;
}

function buildScoreCard(data) {
  const card = document.createElement('article');
  card.className = 'score-card';
  const isTie = data.healthier_country === 'Tie';
  card.innerHTML = `
    <div class="score-label">${isTie ? 'Result' : data.healthier_country + ' wins'}</div>
    <div class="score-number">${isTie ? '=' : fmt(data.score_difference)}<small style="font-size:16px;">${isTie ? '' : ' pts'}</small></div>
    <div class="score-note">${isTie ? 'Both products score equally' : fmt(data.percentage_better) + '% healthier'}</div>
  `;
  return card;
}

function renderVerdict(data) {
  let text;
  if (data.healthier_country === 'Tie') {
    text = 'Both products score equally on overall health.';
  } else {
    text = `${data.healthier_country} version is healthier by ${fmt(data.score_difference)} points (${fmt(data.percentage_better)}% better).`;
  }
  if (data.match_similarity != null) {
    text += ` Match confidence: ${(data.match_similarity * 100).toFixed(1)}%.`;
  }
  elements.verdictContainer.textContent = text;
  elements.verdictContainer.classList.remove('hidden');
}

function renderRecommendation(results) {
  elements.recommendationsContainer.innerHTML = '';

  if (!results || results.length === 0) {
    elements.recommendationsContainer.classList.add('hidden');
    return;
  }

  const r = results[0];
  const card = document.createElement('article');
  card.className = 'product-card';
  card.innerHTML = `
    <div class="card-head">
      <div class="region"><span class="flag">🔁</span> Healthier Indian alternative</div>
    </div>
    <h3 class="product-name" style="margin-top:10px; font-weight:800;">${r.product_name}</h3>
    <p style="color:var(--muted); font-size:12px; margin-bottom:12px;">
      ${r.brand} · Overall ${fmt(r.health_score)}/100 (+${fmt(r.health_boost)})
    </p>
    <div class="ingredient-list">
      <span class="pill">Nutrition: ${fmt(r.nutrition_score)}</span>
      <span class="pill">Additive: ${fmt(r.additive_score)}</span>
      <span class="pill">Ingredient: ${fmt(r.ingredient_score)}</span>
    </div>
  `;
  elements.recommendationsContainer.appendChild(card);
  elements.recommendationsContainer.classList.remove('hidden');
}

/* ==========================================================================
   Helper Utilities
   ========================================================================== */

function fmt(value) {
  return (value === null || value === undefined || Number.isNaN(value)) ? '—' : value;
}

function extractErrorMessage(data, fallback) {
  if (!data) return fallback;
  if (typeof data.detail === 'string') return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map(d => `${(d.loc || []).join('.')}: ${d.msg}`).join(', ');
  if (data.message) return data.message;
  return fallback;
}

function setLoadingState(isLoading) {
  const buttons = [elements.searchBtn, elements.barcodeBtn, elements.uploadBtn];
  buttons.forEach(btn => {
    btn.disabled = isLoading;
    btn.style.opacity = isLoading ? '0.6' : '1';
  });
}

function showError(msg) {
  elements.errorBanner.textContent = msg;
  elements.errorBanner.classList.remove('hidden');
}

function clearError() {
  elements.errorBanner.textContent = '';
  elements.errorBanner.classList.add('hidden');
}
