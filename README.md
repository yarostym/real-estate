# RealtyIQ — Real Estate Analytics Platform

A local web application for real estate price analysis, model training, and undervalued listing detection. Upload a CSV of property listings and get instant insights: correlation analysis, price prediction, category impact, segment models, and an automated model tuning engine.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)

---

## Features

**Correlations** — Heatmap and ranked correlation pairs for any numeric column. Focus-column filter, threshold control, and one-click propagation of correlated features to the Prediction tab.

**Prediction** — Train Random Forest, Gradient Boosting, or Linear Regression on any feature subset. Shows R² (5-fold CV and single split), MAE, feature importance bars, and a summary of active filters and selected features. Save and reload trained models between sessions.

**Category Impact** — Mean and median price breakdown per category value. Supports multi-value comma-separated fields (e.g. `parking_features`, `interior_features`).

**Segment Models** — Train separate models per segment (one per city or property type) and compare accuracy side by side.

**Undervalued Listings** — Score every listing against the trained model to surface properties priced below predicted value. Filter by city, confidence level, minimum undervaluation %, and property specs. Each listing shows comparable listings (`view comps`) from the same district.

**Model Tuning** — Exhaustive grid search over all feature subset combinations using a 3-phase strategy:
- Phase 1: fast RF screening (`n_estimators=30`) on all combinations
- Phase 2: full RF (`n=200`) on top 40%
- Phase 3: Gradient Boosting + Linear on top 20%

Supports distributed search across multiple machines and configurable CPU thread count.

**Saved Models** — Persist trained models to disk and reload without retraining.

---

## Quick Start

```bash
git clone https://github.com/yarostym/real-estate.git
cd real-estate
```

### Windows (PowerShell)

```powershell
py -m venv venv
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## Requirements

```
flask
pandas
numpy
scikit-learn
```

A full `requirements.txt` is included in the repository.

---

## Input Data Format

Upload any CSV where each row is a property listing and one column holds the target price. The app auto-detects numeric vs categorical columns, null rates, and the city/district hierarchy for cascading form dropdowns.

**Typical columns for BC real estate:**

| Column | Type | Notes |
|---|---|---|
| `price` | numeric | Target variable |
| `floor_area_sqft` | numeric | Key predictor |
| `bedrooms`, `bathrooms`, `rooms` | numeric | |
| `year_built` | numeric | |
| `district`, `city` | categorical | High-cardinality; target-encoded |
| `property_type` | categorical | e.g. `House for sale` |
| `parking_features`, `interior_features` | multi-value | Comma-separated; one-hot encoded |
| `summary_url` | text | Rendered as clickable link |

---

## Global Filter

The collapsible filter bar (top of every tab) restricts all analysis to a data subset — for example `property_type: House for sale`. The active filter persists in `localStorage` and is restored automatically on the next page load. All training, tuning, and undervalued scoring respect the active filter.

---

## Model Tuning

### Single Machine

1. Go to the **Tuning** tab
2. Select candidate features, set Min/Max feature count, and adjust CPU threads
3. Click **Find Best Model**
4. Results appear live; click **Use & Train** on any row to apply immediately

### Distributed (Multiple Machines)

Run the same CSV and filter on each machine, then set:

- **Total machines**: e.g. `3`
- **This machine #**: `1`, `2`, or `3` on each machine

Each machine tests a non-overlapping subset (`worker_id::n_workers` stride). When all finish, paste the job IDs from other machines into the **Merge** panel to combine and rank results.

### CPU Threads

Set **CPU threads** to `-1` (all cores, default) or a specific number to limit usage.

---

## Saved Models

After training, click **💾** next to Train Model. Models are saved to `saved_models/` as `.pkl` files and survive app restarts. Load any model from the Saved Models panel to run predictions without retraining.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload CSV; returns column metadata |
| `POST` | `/api/set_global_filter` | Apply data filter |
| `POST` | `/api/train` | Train model |
| `POST` | `/api/predict` | Predict price for input values |
| `POST` | `/api/correlation` | Correlation heatmap data |
| `POST` | `/api/category_impact` | Price breakdown by category |
| `POST` | `/api/undervalued` | Score listings vs model |
| `POST` | `/api/comps` | Comparable listings for a property |
| `POST` | `/api/similar_listings` | Find similar listings |
| `POST` | `/api/tune` | Start async grid search |
| `GET` | `/api/tune_status` | Poll tuning progress |
| `POST` | `/api/tune_stop` | Stop a running tuning job |
| `POST` | `/api/tune_merge` | Merge results from multiple workers |
| `POST` | `/api/tune_estimate` | Estimate test count and runtime |
| `GET` | `/api/models` | List saved models |
| `POST` | `/api/models/save` | Save current model to disk |
| `POST` | `/api/models/load` | Load a saved model into memory |
| `POST` | `/api/models/delete` | Delete a saved model |
| `POST` | `/api/models/rename` | Rename a saved model |
| `GET` | `/api/version` | App version |

---

## Project Structure

```
real-estate/
├── app.py                 # Flask backend — all routes and ML logic
├── templates/
│   └── index.html         # Single-page frontend (vanilla JS, no build step)
├── saved_models/          # Persisted model files (.pkl), auto-created on first save
├── requirements.txt
└── README.md
```

---

## Notes on R² Scores

The app reports two R² values after training:

- **R² — 5-fold CV** (green): cross-validated on the full dataset. Matches Tuning tab results. Use this as the reliable accuracy estimate.
- **R² — single split**: trained on 80%, tested on 20%. Varies by random split; shown for reference only.

Tuning scores may appear 2–5% higher than Prediction CV due to target encoding of high-cardinality columns (`district`, `city`) during the fast screening phase. This is expected — use Prediction CV R² as the ground-truth estimate.

---

## License

Proprietary / All rights reserved