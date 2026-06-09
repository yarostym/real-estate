# RealtyIQ — Real Estate Analytics

ML-powered real estate price analysis tool. Train models on sold data, find undervalued and overvalued listings, tune feature combinations, and predict prices.

---

## 🎬 Demo

[![RealtyIQ Demo](https://img.youtube.com/vi/uLpc7SxfMpY/maxresdefault.jpg)](https://www.youtube.com/watch?v=uLpc7SxfMpY)

**[▶ Watch demo on YouTube](https://www.youtube.com/watch?v=uLpc7SxfMpY)**

---

## Quick Start

```bash
pip install flask pandas numpy scikit-learn
python app.py
# Open http://localhost:5000
```

### Auto-load data on startup

Drop CSV files into these folders — the app loads them automatically:

```
RealtyIQ/
├── app.py
├── templates/index.html
├── data/
│   ├── sold/          ← sold properties (weight ×3, for training)
│   │   └── sold_2024.csv
│   └── ads/           ← active listings (weight ×1, for scoring)
│       └── listings_2024.csv
└── saved_models/
```

---

## Tabs

### Correlations
Explore relationships between numeric columns. Focus column defaults to `sold_price` or `price`. Use the global filter bar to narrow to a property type or city.

### Prediction
Train a model and predict prices.

**Workflow:**
1. Select target column — `sold_price` for sold data, `price` for listings
2. Check feature columns in the sidebar
3. Choose algorithm → **Train Model**
4. Use **Train / Eval split** to train on sold and evaluate on listings
5. Click **Analyse Feature Impact** to see per-feature ablation results

**Metrics:**
- **R² (5-fold CV)** — variance explained (same scale as Tuning)
- **MAE** — average dollar error
- **MAPE** — average % error · < 5% Excellent · < 10% Good · < 15% Fair

**Leakage protection:** when target = `sold_price`, the app auto-removes features that reconstruct listing price (`ppsf` + `floor_area_sqft`, `price`, `price_discount_pct`).

### Category Impact
Median price by category value (district, property type, etc.). Target defaults to `sold_price` or `price`.

### Segment Models
Train one model per category value (e.g. per city). Compare accuracy across segments and predict using the right model.

### Undervalued 🔍
Find listings priced below the model's fair-value estimate.

**Workflow:**
1. Train model on sold data (Prediction tab)
2. Switch to Undervalued, set score dataset to your ads file
3. Click **Find Undervalued**

Each result shows:
- **Gap** — how much below fair value ($ and %)
- **Confidence** — ✅ High / 〜 Medium / ⚠️ Low (RF tree agreement + data coverage)
- **±%** — prediction spread across 200 trees
- **view comps** — comparable sold listings

### Overvalued 📈
Same logic as Undervalued but for overpriced listings. Includes Min confidence filter.

### Tuning ⚙️
Automated grid search for the best feature combination and algorithm.

**3-pass process:**
1. **Pass 1** — RF n=30, all combinations (fast ranking)
2. **Pass 2** — RF n=200, top combos within gap% of best (accurate)
3. **Pass 3** — chosen algorithms (GB / Linear / Stacking) on top 15%

**Algorithm selection:** choose which to test in Pass 3. RF always runs in Pass 1+2.

**✨ Smart Select** — Boruta-style feature ranking using shadow features. Auto-selects real/maybe features above the statistical noise floor.

**Feature frequency chart** — live bar chart showing how often each feature appears in top results.

---

## File Manager

| Control | Effect |
|---------|--------|
| on / off | Include / exclude from all analysis |
| ×1/×2/×3/×5 | Sample weight in training |
| ✕ | Remove from session |

Recommended: **💰 Sold** weight ×3 for training, **📋 Listings** weight ×1 for scoring.

---

## Privacy Mode 🔒

Click **🔒 Privacy: OFF** in the header. State persists across browser sessions.

| Field | Privacy mode |
|-------|-------------|
| Street address | Address #7 |
| City | City 3 |
| District | District 12 |
| URLs, MLS, agent | HIDDEN |
| Postal code, dates | HIDDEN |
| Predicted price | Rounded to nearest $100 |
| MAE in predict | Hidden |

Aliases are stable within a session — same city always maps to the same alias.

---

## Auto-Derived Features

| Feature | Source |
|---------|--------|
| `age` | `year_built` |
| `lot_ratio` | lot / floor area |
| `room_density` | rooms / floor area |
| `bath_ratio` | bathrooms / rooms |
| `ppsf` | price / sqft |
| `n_parking_spaces` | parsed from `parking_spaces` text |
| `n_covered_parking` | parsed from `parking_spaces` text |
| `has_garage`, `has_rv_parking` | parsed from `parking_spaces` |
| `n_fireplaces`, `has_gas_fireplace` | parsed from `fireplace` |
| `n_levels` | `levels` |
| `log_dom` | log(days on market + 1) |
| `dist_van_centre` | GPS distance from Vancouver (km) |
| `has_strata` | `maintenance_fee` > 0 |
| `price_discount_pct` | (sold − asking) / asking % |

---

## Algorithms

| Algorithm | Best for |
|-----------|---------|
| 🌲 Random Forest | Default. Robust, fast, reliable feature importances |
| 🚀 Gradient Boosting | Small datasets, often slightly better R² |
| 📈 Linear Regression | Baseline; interpretable |
| 🔗 Stacking | RF + GB → meta; often best but slowest |

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload primary CSV |
| `/api/upload_extra` | POST | Add extra CSV (type, weight) |
| `/api/upload_meta` | GET | Column metadata |
| `/api/files` | GET | List loaded files |
| `/api/files/toggle` | POST | Include/exclude file |
| `/api/files/set_weight` | POST | Change file weight |
| `/api/files/remove` | POST | Remove file |
| `/api/train` | POST | Train model |
| `/api/predict` | POST | Predict price |
| `/api/tune_batch` | POST | Evaluate feature combo batch |
| `/api/smart_feature_select` | POST | Boruta feature ranking |
| `/api/feature_analysis` | POST | Per-feature ablation |
| `/api/undervalued` | POST | Find underpriced listings |
| `/api/overvalued` | POST | Find overpriced listings |
| `/api/comps` | POST | Comparable sold listings |
| `/api/similar_listings` | POST | Similar listings by feature distance |
| `/api/correlation` | POST | Pairwise correlations |
| `/api/category_impact` | POST | Price by category value |
| `/api/segment_train` | POST | Per-segment model training |
| `/api/toggle_demo` | POST | Toggle privacy mode |
| `/api/models` | GET/POST | Save/load/delete models |

---

## Requirements

```
flask
pandas
numpy
scikit-learn
```

Python 3.9+. No database — state is in-memory per session.