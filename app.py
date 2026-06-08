from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error
import io
import re
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*delayed.*Parallel.*', category=UserWarning)
# Also set PYTHONWARNINGS env var so child threads/processes inherit it
import os; os.environ.setdefault('PYTHONWARNINGS', 'ignore')

app = Flask(__name__)

uploaded_data = {}
_excluded_sources = set()  # sources excluded via file manager

# ── Auto-load from data/ads and data/sold on startup ─────────────────────────
def _autoload_data_folders():
    import glob as _glob, datetime as _dt
    base = _os.path.dirname(_os.path.abspath(__file__))
    ads_dir  = _os.path.join(base, 'data', 'ads')
    sold_dir = _os.path.join(base, 'data', 'sold')
    frames = []
    for folder, ftype, weight in [(ads_dir,'listings',1.0),(sold_dir,'sold',3.0)]:
        if not _os.path.isdir(folder): continue
        for fpath in sorted(_glob.glob(_os.path.join(folder,'*.csv'))):
            try:
                with open(fpath,'rb') as f: raw = f.read()
                try:    df_f = pd.read_csv(io.BytesIO(raw), encoding='utf-8', sep=None, engine='python')
                except: df_f = pd.read_csv(io.BytesIO(raw), encoding='latin-1', sep=None, engine='python')
                df_f = clean_dataframe(df_f)
                df_f = add_derived_features(df_f)
                df_f['_source']        = f'{ftype}/{_os.path.basename(fpath)}'
                df_f['_file_type']     = ftype
                df_f['_sample_weight'] = weight
                frames.append(df_f)
            except Exception as e:
                print(f'[autoload] skipped {fpath}: {e}')
    if frames:
        merged = pd.concat(frames, ignore_index=True, sort=False)
        uploaded_data['default'] = merged
        print(f'[autoload] loaded {len(merged)} rows from {len(frames)} files')



def add_derived_features(df):
    """Auto-engineer features that improve model quality."""
    import datetime as _dt
    cy = _dt.datetime.now().year
    if 'year_built' in df.columns and 'age' not in df.columns:
        yb = pd.to_numeric(df['year_built'], errors='coerce')
        df['age'] = (cy - yb).clip(0, 200)
    if 'lot_size_sqft' in df.columns and 'floor_area_sqft' in df.columns and 'lot_ratio' not in df.columns:
        fa = pd.to_numeric(df['floor_area_sqft'], errors='coerce').replace(0, np.nan)
        df['lot_ratio'] = (pd.to_numeric(df['lot_size_sqft'], errors='coerce') / fa).clip(0, 100)
    if 'rooms' in df.columns and 'floor_area_sqft' in df.columns and 'room_density' not in df.columns:
        fa = pd.to_numeric(df['floor_area_sqft'], errors='coerce').replace(0, np.nan)
        df['room_density'] = (pd.to_numeric(df['rooms'], errors='coerce') / fa * 100).clip(0, 10)
    if 'bathrooms' in df.columns and 'rooms' in df.columns and 'bath_ratio' not in df.columns:
        rm = pd.to_numeric(df['rooms'], errors='coerce').replace(0, np.nan)
        df['bath_ratio'] = (pd.to_numeric(df['bathrooms'], errors='coerce') / rm).clip(0, 2)
    for nc, sc in [('has_view','view'),('has_fireplace','fireplace_features'),
                   ('has_cooling','cooling_features'),('has_parking_feat','parking_features'),
                   ('has_appliances','appliances'),('has_interior','interior_features')]:
        if sc in df.columns and nc not in df.columns:
            df[nc] = df[sc].fillna('').astype(str).str.strip().str.len().gt(2).astype(float)
    return df

def clean_dataframe(df):
    """Coerce columns to numeric when >= 80% of values look numeric.
    Strips common unit suffixes (ft, sqft, m, km, kg, lb, %) before parsing."""
    unit_pattern = r'\s*(ft\.?|sqft|sq\.?\s*ft\.?|m2?|km|kg|lb|lbs|%|acres?|ha)\s*$'
    for col in df.columns:
        s = df[col].astype(str).str.strip()
        # Strip unit suffixes then commas/spaces
        s_clean = s.str.replace(unit_pattern, '', regex=True, case=False)
        s_clean = s_clean.str.replace(',', '.').str.replace(' ', '')
        converted = pd.to_numeric(s_clean, errors='coerce')
        # Use non-null rows as denominator — sparse columns (many NaN) should still convert
        non_null = (df[col].notna() & (df[col].astype(str).str.strip().isin(['', 'nan']) == False)).sum()
        if non_null > 0 and converted.notna().sum() / non_null >= 0.8:
            df[col] = converted
    return df

_INTERNAL_COLS = {'_source', '_file_type', '_sample_weight'}

def get_numeric_columns(df):
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in _INTERNAL_COLS]

def get_categorical_columns(df):
    return [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in _INTERNAL_COLS]

def correlation_strength(val):
    if val >= 0.9: return 'very strong'
    if val >= 0.7: return 'strong'
    if val >= 0.5: return 'moderate'
    if val >= 0.3: return 'weak'
    return 'very weak'

@app.route('/')
def index():
    return render_template('index.html')


CITY_GROUPS = {
    "Vancouver":       ["Vancouver","Burnaby","Richmond","New Westminster","North Vancouver","West Vancouver","Coquitlam","Port Coquitlam","Port Moody","Pitt Meadows","Maple Ridge","Delta","Surrey","Langley","White Rock","Tsawwassen","Mission","Abbotsford","Ladner","Anmore","Lions Bay"],
    "Burnaby":         ["Burnaby","Vancouver","Richmond","New Westminster","Coquitlam","North Vancouver"],
    "Richmond":        ["Richmond","Vancouver","Burnaby","Delta","Tsawwassen","Ladner"],
    "Surrey":          ["Surrey","Langley","White Rock","Delta","Abbotsford","Tsawwassen","Maple Ridge","Mission"],
    "Langley":         ["Langley","Surrey","Abbotsford","Maple Ridge","Mission"],
    "North Vancouver": ["North Vancouver","West Vancouver","Vancouver","Burnaby","Squamish"],
    "West Vancouver":  ["West Vancouver","North Vancouver","Vancouver","Squamish"],
    "Coquitlam":       ["Coquitlam","Port Coquitlam","Port Moody","Burnaby","Maple Ridge","Pitt Meadows"],
    "New Westminster": ["New Westminster","Burnaby","Coquitlam","Surrey","Vancouver"],
    "Abbotsford":      ["Abbotsford","Chilliwack","Mission","Langley","Surrey"],
    "Chilliwack":      ["Chilliwack","Abbotsford","Mission","Agassiz","Harrison Hot Springs","Hope"],
    "Mission":         ["Mission","Abbotsford","Maple Ridge","Chilliwack"],
    "Maple Ridge":     ["Maple Ridge","Pitt Meadows","Mission","Coquitlam"],
    "Delta":           ["Delta","Surrey","Richmond","Tsawwassen","Ladner"],
    "White Rock":      ["White Rock","Surrey","Delta","Tsawwassen"],
    "Nanaimo":         ["Nanaimo","Lantzville","Nanoose Bay","Parksville","Ladysmith","Chemainus","Duncan"],
    "Courtenay":       ["Courtenay","Comox","Campbell River","Fanny Bay","Black Creek","Cumberland","Royston"],
    "Duncan":          ["Duncan","Mill Bay","Lake Cowichan","Chemainus","Ladysmith","Cowichan Bay","Crofton"],
    "Parksville":      ["Parksville","Qualicum Beach","Errington","Lantzville","Nanoose Bay"],
    "Campbell River":  ["Campbell River","Gold River","Sayward","Black Creek","Courtenay"],
    "Prince George":   ["Prince George","Vanderhoof","Fort St. James","Fraser Lake","Burns Lake"],
    "Sechelt":         ["Sechelt","Gibsons","Garden Bay","Halfmoon Bay","Pender Harbour"],
    "Squamish":        ["Squamish","Whistler","Pemberton","Britannia Beach"],
    "Whistler":        ["Whistler","Squamish","Pemberton"],
}

def get_nearby_cities(city):
    return CITY_GROUPS.get(city, [city])

# ── Upload ────────────────────────────────────────────────────────────────────
@app.route('/api/toggle_demo', methods=['POST'])
def toggle_demo():
    uploaded_data['demo_mode'] = not uploaded_data.get('demo_mode', False)
    return jsonify({'demo_mode': uploaded_data['demo_mode']})

def mask_row(row):
    if not uploaded_data.get('demo_mode'): return row
    masked = dict(row)
    for k in list(masked.keys()):
        kl = k.lower()
        if any(x in kl for x in ['street','address','url','link','mls','agent','brokerage','phone','email','tour']):
            if isinstance(masked[k], str) and masked[k]:
                masked[k] = 'HIDDEN'
    return masked

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        content = file.read()
        df = None
        for encoding in ['utf-8', 'utf-8-sig', 'cp1251', 'latin-1']:
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=encoding, sep=None, engine='python')
                break
            except Exception:
                continue
        if df is None:
            return jsonify({'error': 'Could not parse CSV file'}), 400

        df = clean_dataframe(df)
        df = add_derived_features(df)
        uploaded_data['default'] = df

        # Detect city column heuristically
        city_col = next((c for c in df.columns if c.lower() in ('city','місто','town','municipality')), None)
        cities = sorted(df[city_col].dropna().unique().tolist()) if city_col else []
        uploaded_data['city_col']     = city_col
        uploaded_data['global_filters'] = {}  # reset filters on new upload

        null_rates  = {col: round(df[col].isnull().sum() / max(len(df),1), 3)
                       for col in df.columns}
        cat_n_unique = {col: int(df[col].dropna().nunique())
                        for col in get_categorical_columns(df)}

        # Build city -> districts mapping for cascading dropdown in Predict form
        city_to_districts = {}
        city_col_name = city_col or 'city'
        if city_col_name in df.columns and 'district' in df.columns:
            for _, row in df.dropna(subset=['district']).iterrows():
                c = str(row.get(city_col_name, '') or '').strip()
                d = str(row['district']).strip()
                if c and d and c != d:
                    if c not in city_to_districts:
                        city_to_districts[c] = set()
                    city_to_districts[c].add(d)
            city_to_districts = {k: sorted(v) for k, v in sorted(city_to_districts.items())}

        # Filterable categoricals: skip ID/URL/address cols, keep <=50 unique after split
        skip_filter = ['real_mls','mls','street','address','summary_url','virtual_tour',
                       'agency','brokerage','property_taxes','tax_year','lot_frontage_ft',
                       'listing_agent','photo_count','photos']
        cat_filter_cols = {}
        for col in get_categorical_columns(df):
            if any(s in col.lower() for s in skip_filter):
                continue
            # Split comma-separated values to get unique options
            all_vals = sorted(set(
                v.strip() for row in df[col].fillna('').astype(str)
                for v in row.split(',')
                if v.strip() and v.strip().lower() not in ('nan', 'unknown', '')
            ))
            # Higher limit for location cols, normal limit for others
            limit = 120 if col in ('district','city') else 50
            if 2 <= len(all_vals) <= limit:
                cat_filter_cols[col] = all_vals

        return jsonify({
            'success':           True,
            'rows':              len(df),
            'columns':           len(df.columns),
            'numeric_columns':   get_numeric_columns(df),
            'categorical_columns': get_categorical_columns(df),
            'all_columns':       df.columns.tolist(),
            'city_col':          city_col,
            'cities':            cities,
            'null_rates':        null_rates,
            'cat_n_unique':      cat_n_unique,
            'cat_filter_cols':   cat_filter_cols,
            'city_to_districts': city_to_districts,
            'preview':           df.head(5).fillna('').to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': f'File read error: {str(e)}'}), 400

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_working_df():
    """Return df filtered by global_filters dict and excluded sources."""
    df = uploaded_data.get("default")
    if df is None:
        return None
    # Exclude toggled-off source files
    if _excluded_sources and '_source' in df.columns:
        df = df[~df['_source'].isin(_excluded_sources)].reset_index(drop=True)
    filters = uploaded_data.get('global_filters', {})
    for col, val in filters.items():
        if not val or col not in df.columns:
            continue
        # Support both single string and list
        if isinstance(val, list):
            df = df[df[col].astype(str).isin([str(v) for v in val])].reset_index(drop=True)
        else:
            df = df[df[col].astype(str) == str(val)].reset_index(drop=True)
    return df

@app.route('/api/set_city_filter', methods=['POST'])
def set_city_filter():
    # Legacy endpoint - redirect to global filter using city key
    data = request.json or {}
    city = data.get('city')
    filters = {'city': city} if city else {}
    uploaded_data['global_filters'] = filters
    df = get_working_df()
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400
    return jsonify({'success': True, 'rows': len(df), 'city': city or 'All cities'})

@app.route('/api/set_global_filter', methods=['POST'])
def set_global_filter():
    data    = request.json or {}
    filters = data.get('filters', {})   # {col: value, ...}
    # Remove empty values
    filters = {k: v for k, v in filters.items() if v}
    uploaded_data['global_filters'] = filters
    df = get_working_df()
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400
    return jsonify({'success': True, 'rows': len(df), 'filters': filters})


# ── Correlation ───────────────────────────────────────────────────────────────
@app.route('/api/correlation', methods=['POST'])
def get_correlation():
    df = get_working_df()
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400

    data         = request.json or {}
    threshold    = float(data.get('threshold', 0.3))
    focus_col    = data.get('focus_column')        # e.g. "price" — filter relative to this column
    all_numeric  = get_numeric_columns(df)
    requested    = data.get('columns', [])
    numeric_cols = [c for c in requested if c in all_numeric] if requested else all_numeric
    if len(numeric_cols) < 2:
        return jsonify({'error': 'Select at least 2 numeric columns'}), 400

    # Ensure focus column is included even if not in requested list
    if focus_col and focus_col in all_numeric and focus_col not in numeric_cols:
        numeric_cols = [focus_col] + numeric_cols

    corr_matrix = df[numeric_cols].corr().round(4)

    significant = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold and not np.isnan(val):
                # When focus_col set: only pairs that include the focus column
                if focus_col and focus_col not in (numeric_cols[i], numeric_cols[j]):
                    continue
                significant.append({
                    'col1':        numeric_cols[i],
                    'col2':        numeric_cols[j],
                    'correlation': round(float(val), 4),
                    'strength':    correlation_strength(abs(val)),
                    'direction':   'positive' if val > 0 else 'negative'
                })
    significant.sort(key=lambda x: abs(x['correlation']), reverse=True)

    # Build matrix only for columns that are relevant
    # When focus_col set: include focus_col + all columns correlated with it
    if focus_col:
        relevant = set([focus_col])
        for s in significant:
            relevant.add(s['col1']); relevant.add(s['col2'])
        matrix_cols = [c for c in numeric_cols if c in relevant]
    else:
        matrix_cols = numeric_cols

    matrix_data = []
    for i, c1 in enumerate(matrix_cols):
        for j, c2 in enumerate(matrix_cols):
            val = corr_matrix.loc[c1, c2] if c1 in corr_matrix.index and c2 in corr_matrix.columns else 0
            matrix_data.append({
                'x': j, 'y': i,
                'value': round(float(val), 4) if not np.isnan(val) else 0,
                'col1': c1, 'col2': c2
            })

    return jsonify({
        'correlation_matrix':    corr_matrix.fillna(0).to_dict(),
        'matrix_data':           matrix_data,
        'columns':               matrix_cols,
        'focus_column':          focus_col,
        'significant_correlations': significant,
        'total_significant':     len(significant)
    })

# ── Target Encoding ──────────────────────────────────────────────────────────
def target_encode(df, cat_col, target_col, stats=None):
    """
    Encode categorical column using median/mean target value per category.
    If stats=None, compute from df (training). Otherwise use provided stats (inference).
    Returns (encoded_series, stats_dict).
    stats_dict keys: 'median', 'mean', 'ppsf' (if sqft available), 'global_median'
    """
    if stats is None:
        global_med = float(df[target_col].median())
        by_cat = df.groupby(cat_col)[target_col].agg(['median','mean'])
        stats = {
            'global_median': global_med,
            'median': by_cat['median'].to_dict(),
            'mean':   by_cat['mean'].to_dict(),
        }
        # price-per-sqft if sqft available
        if 'sqft' in df.columns and pd.api.types.is_numeric_dtype(df['sqft']):
            ppsf = df.groupby(cat_col).apply(
                lambda g: (g[target_col] / g['sqft'].replace(0, np.nan)).median()
            )
            stats['ppsf'] = ppsf.to_dict()

    med_map  = stats['median']
    ppsf_map = stats.get('ppsf', {})
    fallback = stats['global_median']

    enc_med  = df[cat_col].map(med_map).fillna(fallback).astype(float)
    enc_ppsf = df[cat_col].map(ppsf_map).fillna(fallback / df.get('sqft', pd.Series([1]*len(df))).replace(0,1)).astype(float) if ppsf_map else None

    return enc_med, enc_ppsf, stats



def is_multi_value_col(series, delimiter=',', min_rows_with_delim=3):
    """Returns True if this column has multiple values separated by delimiter in >= min_rows_with_delim rows."""
    if pd.api.types.is_numeric_dtype(series):
        return False
    count = series.dropna().astype(str).str.contains(re.escape(delimiter)).sum()
    return count >= min_rows_with_delim


def ohe_multi_value(series, delimiter=',', existing_values=None):
    """
    One-hot encode a comma-separated column.
    Returns (X_ohe_array, value_list).
    If existing_values provided (inference), use those - else derive from data.
    """
    import re as _re
    cleaned = series.fillna('').astype(str)
    if existing_values is None:
        all_vals = sorted(set(
            v.strip() for row in cleaned for v in row.split(delimiter)
            if v.strip() and v.strip().lower() not in ('nan', 'unknown', '')
        ))
    else:
        all_vals = existing_values
    rows = []
    for row in cleaned:
        present = set(v.strip() for v in row.split(delimiter))
        rows.append([1.0 if v in present else 0.0 for v in all_vals])
    return np.array(rows, dtype=float), all_vals

# ── Train ─────────────────────────────────────────────────────────────────────
@app.route('/api/train', methods=['POST'])
def train_model():
    df = get_working_df()
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400

    data         = request.json or {}
    target_col   = data.get('target_column')
    feature_cols = data.get('feature_columns', [])
    model_type   = data.get('model_type', 'random_forest')

    if not target_col:
        return jsonify({'error': 'No target column selected'}), 400
    # Strip target_col and internal columns from features
    _internal = {'_source', '_file_type', '_sample_weight'}
    feature_cols = [c for c in feature_cols if c != target_col and c not in _internal]
    if not feature_cols:
        return jsonify({'error': 'No feature columns selected'}), 400

    try:
        # Drop only rows where TARGET is null — fill missing features via imputation
        df_model = df[feature_cols + [target_col]].copy()
        df_model = df_model.dropna(subset=[target_col]).reset_index(drop=True)
        df_model[target_col] = pd.to_numeric(df_model[target_col], errors='coerce')
        df_model = df_model.dropna(subset=[target_col]).reset_index(drop=True)

        if len(df_model) < 5:
            return jsonify({'error': 'Not enough data (minimum 5 rows)'}), 400

        # Remove outliers BEFORE encoding (same as tuning) so target_encode uses clean data
        remove_outliers  = data.get('remove_outliers', True)
        outliers_removed = 0
        if remove_outliers and len(df_model) >= 20:
            _y_pre = df_model[target_col].values.astype(float)
            _q1, _q3 = np.percentile(_y_pre, 25), np.percentile(_y_pre, 75)
            _iqr = _q3 - _q1
            _mask = (_y_pre >= _q1 - 3*_iqr) & (_y_pre <= _q3 + 3*_iqr)
            outliers_removed = int((~_mask).sum())
            df_model = df_model[_mask].reset_index(drop=True)
            if len(df_model) < 5:
                return jsonify({'error': f'After removing {outliers_removed} outliers, not enough rows remain'}), 400

        # Encode features — build X as a plain numpy array column by column
        encoders      = {}
        feature_types = {}
        X_cols        = []   # list of 1-D arrays, one per feature

        for col in feature_cols:
            col_data = df_model[col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            col_data = col_data.reset_index(drop=True)

            if not pd.api.types.is_numeric_dtype(col_data):
                col_data = col_data.fillna('Unknown')
                n_unique = col_data.nunique()

                # Multi-value comma-separated → OHE (e.g. "Elevator, Storage")
                if is_multi_value_col(col_data):
                    ohe_arr, ohe_vals = ohe_multi_value(col_data)
                    encoders[col]      = {'type': 'multi_ohe', 'values': ohe_vals}
                    feature_types[col] = 'multi_ohe'
                    for k in range(ohe_arr.shape[1]):
                        X_cols.append(ohe_arr[:, k])
                # High-cardinality single-value: target encoding
                elif n_unique > 15:
                    enc_med, enc_ppsf, te_stats = target_encode(
                        df_model.assign(**{col: col_data}), col, target_col
                    )
                    encoders[col]      = {'type': 'target_encoded', 'stats': te_stats, 'n_unique': n_unique}
                    feature_types[col] = 'target_encoded'
                    X_cols.append(enc_med.values)
                    if enc_ppsf is not None:
                        X_cols.append(enc_ppsf.values)
                        encoders[col]['has_ppsf'] = True
                    else:
                        encoders[col]['has_ppsf'] = False
                else:
                    # Low-cardinality single-value: label-encode
                    le       = LabelEncoder()
                    encoded  = le.fit_transform(col_data.astype(str).values).astype(float)
                    encoders[col]      = {'type': 'categorical', 'classes': le.classes_.tolist()}
                    feature_types[col] = 'categorical'
                    X_cols.append(encoded)
            else:
                # Numeric: coerce + impute median (robust to outliers)
                numeric = pd.to_numeric(col_data, errors='coerce')
                fill_val = numeric.median() if numeric.notna().any() else 0.0
                numeric = numeric.fillna(fill_val).values.astype(float)
                encoders[col] = {
                    'type': 'numeric',
                    'min':  float(numeric.min()),
                    'max':  float(numeric.max()),
                    'mean': float(numeric.mean())
                }
                feature_types[col] = 'numeric'
                X_cols.append(numeric)

        X = np.column_stack(X_cols)                              # shape (n_samples, n_features)
        y = pd.to_numeric(df_model[target_col], errors='coerce').values.astype(float)

        # Drop rows where target couldn't be parsed
        valid = ~np.isnan(y)
        X, y  = X[valid], y[valid]

        if len(y) < 5:
            return jsonify({'error': 'Not enough valid rows after parsing target column'}), 400

        # Outliers already removed before encoding above

        # Log-transform target if all values positive (reduces skew, improves R²)
        use_log = data.get('use_log_target', True) and bool(np.all(y > 0))
        if use_log:
            y_fit = np.log1p(y)
        else:
            y_fit = y

        # Choose model
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
        elif model_type == 'stacking':
            from sklearn.ensemble import StackingRegressor
            model = StackingRegressor(estimators=[('rf',RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=-1)),('gb',GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,max_depth=3,random_state=42))],final_estimator=LinearRegression(),cv=5)
        else:
            model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)

        # Small datasets → cross-validation; large → train/test split
        if len(y_fit) < 30:
            n_splits  = min(5, len(y_fit))
            cv        = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y_fit, cv=cv, scoring='r2')
            y_pred_fit = cross_val_predict(model, X, y_fit, cv=cv)
            model.fit(X, y_fit)
            y_pred = np.expm1(y_pred_fit) if use_log else y_pred_fit
            y_test = np.expm1(y_fit) if use_log else y_fit
            r2     = float(np.mean(cv_scores))
            mae    = float(mean_absolute_error(y_test, y_pred))
            train_n, test_n = len(y_fit), len(y_fit)
            r2_cv, mae_cv = r2, mae  # already CV for small datasets
        else:
            X_train, X_test, y_tr, y_te = train_test_split(X, y_fit, test_size=0.2, random_state=42)
            model.fit(X_train, y_tr)
            y_pred_fit = model.predict(X_test)
            y_pred = np.expm1(y_pred_fit) if use_log else y_pred_fit
            y_test = np.expm1(y_te)       if use_log else y_te
            r2     = float(r2_score(y_test, y_pred))
            mae    = float(mean_absolute_error(y_test, y_pred))
            train_n = len(X_train)
            test_n  = len(X_test)
            # 5-fold CV on full data — same method as Tuning tab
            try:
                cv_m   = build_model(model_type)
                cv_spl = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_p   = cross_val_predict(cv_m, X, y_fit, cv=cv_spl)
                cv_p   = np.expm1(cv_p) if use_log else cv_p
                y_all  = np.expm1(y_fit) if use_log else y_fit
                r2_cv  = float(r2_score(y_all, cv_p))
                mae_cv = float(mean_absolute_error(y_all, cv_p))
            except Exception:
                r2_cv = mae_cv = None

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'estimators_'):
            # Stacking: average importances from base estimators
            # sklearn 1.8+: estimators_ is a flat list of fitted estimators
            est_list = model.estimators_
            imps = [e.feature_importances_ for e in est_list
                    if hasattr(e, 'feature_importances_')]
            importances = np.mean(imps, axis=0) if imps else np.ones(len(feature_cols)) / len(feature_cols)
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            s = importances.sum() or 1
            importances = importances / s
        else:
            importances = np.ones(len(feature_cols)) / len(feature_cols)

        feature_importance = sorted(
            [{'feature': col, 'importance': round(float(imp), 4)}
             for col, imp in zip(feature_cols, importances)],
            key=lambda x: x['importance'], reverse=True
        )

        uploaded_data['default_model'] = {
            'model':            model,
            'model_type':       model_type,
            'feature_cols':     feature_cols,
            'target_col':       target_col,
            'encoders':         encoders,
            'feature_types':    feature_types,
            'use_log':          use_log,
            '_te_stats':        {k: v['stats'] for k, v in encoders.items() if v.get('type') == 'target_encoded'},
            'r2_score':         round(r2, 4),
            'r2_cv':            round(r2_cv, 4) if r2_cv is not None else None,
            'mae_cv':           round(mae_cv, 2) if mae_cv is not None else None,
            'train_size':       train_n,
            'test_size':        test_n,
            'outliers_removed': outliers_removed,
        }
        uploaded_data['n_rows'] = train_n + test_n

        sample         = min(50, len(y_test))
        actual_vs_pred = [
            {'actual': round(float(a), 2), 'predicted': round(float(p), 2)}
            for a, p in zip(y_test[:sample], y_pred[:sample])
        ]

        uploaded_data['default_model']['feature_importance'] = feature_importance
        uploaded_data['n_rows'] = train_n + test_n
        return jsonify({
            'success':             True,
            'r2_score':            round(r2, 4),
            'mae':                 round(mae, 2),
            'r2_cv':               round(r2_cv, 4) if r2_cv is not None else None,
            'mae_cv':              round(mae_cv, 2) if mae_cv is not None else None,
            'train_size':          train_n,
            'test_size':           test_n,
            'outliers_removed':    outliers_removed,
            'feature_importance':  feature_importance,
            'actual_vs_predicted': actual_vs_pred,
            'encoders':            {k: {
                                       'type': v['type'],
                                       'classes': (v['classes'] if v['type'] == 'categorical'
                                                   else sorted(v['stats']['median'].keys()) if v['type'] == 'target_encoded'
                                                   else v.get('values', [])),
                                       'mean': v.get('mean'),
                                       'min':  v.get('min'),
                                       'max':  v.get('max'),
                                   } for k, v in encoders.items()},
            'feature_types':       feature_types,
            'model_type':          model_type
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # Find the most useful line from traceback
        lines = [l.strip() for l in tb.split('\n') if l.strip() and 'File' not in l and 'Traceback' not in l]
        detail = lines[-2] if len(lines) >= 2 else str(e)
        return jsonify({'error': f'Training error: {str(e)}', 'detail': detail, 'trace': tb}), 400

# ── Predict ───────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    model_data = uploaded_data.get('default_model')
    if not model_data:
        return jsonify({'error': 'Model not trained yet'}), 400

    data         = request.json or {}
    input_values = data.get('values', {})

    try:
        model        = model_data['model']
        feature_cols = model_data['feature_cols']
        encoders     = model_data['encoders']

        row = []
        for col in feature_cols:
            val = input_values.get(col)
            enc = encoders[col]

            if enc['type'] == 'multi_ohe':
                # One-hot: one value per token in the comma-separated input
                val_str    = str(val) if val is not None else ''
                present    = set(v.strip() for v in val_str.split(','))
                for v in enc['values']:
                    row.append(1.0 if v in present else 0.0)
                continue  # already appended all sub-features, skip outer append
            elif enc['type'] == 'target_encoded':
                stats   = enc['stats']
                val_str = str(val) if val is not None else ''
                # median encoded value
                med_val = stats['median'].get(val_str, stats['global_median'])
                row.append(float(med_val))
                # ppsf encoded value
                if enc.get('has_ppsf') and 'ppsf' in stats:
                    sqft_val = float(input_values.get('sqft') or 1000)
                    ppsf_val = stats['ppsf'].get(val_str, stats['global_median'] / max(sqft_val, 1))
                    row.append(float(ppsf_val))
            elif enc['type'] == 'categorical':
                if val is None or val == '':
                    val = enc['classes'][0]
                classes = enc['classes']
                val_str = str(val)
                row.append(float(classes.index(val_str)) if val_str in classes else 0.0)
            else:
                if val is None or val == '':
                    val = enc.get('mean', 0)
                row.append(float(val))

        prediction = model.predict([row])[0]
        if model_data.get('use_log'):
            prediction = np.expm1(prediction)
        return jsonify({
            'prediction': round(float(prediction), 2),
            'target_col': model_data['target_col']
        })
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 400

# ── Scatter ───────────────────────────────────────────────────────────────────
@app.route('/api/scatter', methods=['POST'])
def get_scatter():
    df = get_working_df()
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400

    data = request.json or {}
    col1 = data.get('col1')
    col2 = data.get('col2')
    if not col1 or not col2:
        return jsonify({'error': 'Specify both columns'}), 400

    subset = df[[col1, col2]].dropna()
    sample = subset.sample(min(200, len(subset)), random_state=42)

    return jsonify({
        'points': [{'x': float(r[col1]), 'y': float(r[col2])} for _, r in sample.iterrows()],
        'col1': col1, 'col2': col2
    })


# ── Category Impact ───────────────────────────────────────────────────────────
@app.route('/api/category_impact', methods=['POST'])
def category_impact():
    df = get_working_df()
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400

    data       = request.json or {}
    target_col = data.get('target_column')

    if not target_col or target_col not in df.columns:
        return jsonify({'error': 'Invalid target column'}), 400
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return jsonify({'error': 'Target column must be numeric'}), 400

    include_cols = data.get('include_columns')   # None = all
    cat_cols = get_categorical_columns(df)
    cat_cols = [c for c in cat_cols if c != target_col]
    if include_cols:
        cat_cols = [c for c in cat_cols if c in include_cols]

    results = []
    overall_mean = float(df[target_col].mean())
    overall_median = float(df[target_col].median())

    delimiter = data.get('delimiter', '')   # '' = no splitting

    for col in cat_cols:
        sub = df[[col, target_col]].dropna().copy()
        if len(sub) < 2:
            continue

        if delimiter:
            # Explode multi-value cells: "Elevator, Storage" → two rows
            sub[col] = sub[col].astype(str).str.split(delimiter)
            sub = sub.explode(col)
            sub[col] = sub[col].str.strip()
            sub = sub[sub[col] != '']

        groups_dict = {}   # value → list of prices
        for _, row in sub.iterrows():
            val = str(row[col]).strip()
            if val not in groups_dict:
                groups_dict[val] = []
            groups_dict[val].append(float(row[target_col]))

        groups = []
        for val, prices_list in groups_dict.items():
            prices = pd.Series(prices_list)
            groups.append({
                'value':    val,
                'mean':     round(float(prices.mean()), 0),
                'median':   round(float(prices.median()), 0),
                'count':    int(len(prices)),
                'min':      round(float(prices.min()), 0),
                'max':      round(float(prices.max()), 0),
                'pct_diff': round((float(prices.mean()) - overall_mean) / overall_mean * 100, 1)
            })
        groups = [g for g in groups if g['count'] >= 5]
        groups.sort(key=lambda x: x['mean'], reverse=True)
        results.append({
            'column':       col,
            'groups':       groups,
            'n_categories': len(groups),
            'split_by':     delimiter if delimiter else None
        })

    # sort columns by variance of group means (most impactful first)
    def col_variance(r):
        means = [g['mean'] for g in r['groups']]
        return float(np.std(means)) if len(means) > 1 else 0
    results.sort(key=col_variance, reverse=True)

    return jsonify({
        'results':        results,
        'overall_mean':   round(overall_mean, 0),
        'overall_median': round(overall_median, 0),
        'target_col':     target_col
    })

# ── Segment Models ────────────────────────────────────────────────────────────
def build_model(model_type):
    if model_type == 'linear':
        return LinearRegression()
    elif model_type == 'gradient_boosting':
        return GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    elif model_type == 'stacking':
        from sklearn.ensemble import StackingRegressor
        return StackingRegressor(
            estimators=[('rf', RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)),
                        ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42))],
            final_estimator=LinearRegression(), cv=5)
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)

def encode_and_fit(df_seg, feature_cols, target_col, model_type):
    """Encode features, fit model, return result dict or None if not enough data."""
    df_seg = df_seg[feature_cols + [target_col]].dropna().reset_index(drop=True)
    if len(df_seg) < 3:
        return None

    encoders      = {}
    feature_types = {}
    X_cols        = []
    for col in feature_cols:
        # Always extract as a 1-D Series — guards against duplicate column names
        col_data = df_seg[col]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        col_data = col_data.reset_index(drop=True)

        if not pd.api.types.is_numeric_dtype(col_data):
            col_data = col_data.fillna('Unknown')
            if is_multi_value_col(col_data):
                ohe_arr, ohe_vals = ohe_multi_value(col_data)
                encoders[col]      = {'type': 'multi_ohe', 'values': ohe_vals}
                feature_types[col] = 'multi_ohe'
                for k in range(ohe_arr.shape[1]):
                    X_cols.append(ohe_arr[:, k])
            else:
                le      = LabelEncoder()
                encoded = le.fit_transform(col_data.astype(str).values).astype(float)
                encoders[col]      = {'type': 'categorical', 'classes': le.classes_.tolist()}
                feature_types[col] = 'categorical'
                X_cols.append(encoded)
        else:
            num = pd.to_numeric(col_data, errors='coerce')
            fill_val = num.median() if num.notna().any() else 0.0
            num = num.fillna(fill_val).values.astype(float)
            encoders[col]      = {'type': 'numeric', 'min': float(num.min()),
                                  'max': float(num.max()), 'mean': float(num.mean())}
            feature_types[col] = 'numeric'
            X_cols.append(num)

    # Build X safely — handle single-feature case
    if len(X_cols) == 1:
        X = X_cols[0].reshape(-1, 1)
    else:
        X = np.column_stack(X_cols)

    y = pd.to_numeric(df_seg[target_col], errors='coerce').values.astype(float)
    valid = ~np.isnan(y)
    X, y  = X[valid], y[valid]
    if len(y) < 3:
        return None

    # Remove outliers (3×IQR rule) for segments large enough
    if len(y) >= 10:
        q1, q3 = np.percentile(y, 25), np.percentile(y, 75)
        iqr    = q3 - q1
        mask   = (y >= q1 - 3*iqr) & (y <= q3 + 3*iqr)
        X, y   = X[mask], y[mask]
    if len(y) < 3:
        return None

    model = build_model(model_type)

    # Log-transform target if all values positive
    use_log = bool(np.all(y > 0))
    y_fit   = np.log1p(y) if use_log else y

    # For very small segments: just fit on all data, report train metrics
    # For medium (3-19): leave-one-out style with n_splits = min(n, 5)
    # For large (20+): proper train/test split
    if len(y) >= 20:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_fit, test_size=0.2, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred_fit = model.predict(X_te)
        y_pred = np.expm1(y_pred_fit) if use_log else y_pred_fit
        y_test = np.expm1(y_te)       if use_log else y_te
        r2     = float(r2_score(y_test, y_pred))
        mae    = float(mean_absolute_error(y_test, y_pred))
    elif len(y) >= 6:
        n_splits = min(3, len(y) // 2)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred_fit = cross_val_predict(model, X, y_fit, cv=cv)
        y_pred = np.expm1(y_pred_fit) if use_log else y_pred_fit
        y_test = np.expm1(y_fit)      if use_log else y_fit
        r2  = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        model.fit(X, y_fit)
    else:
        model.fit(X, y_fit)
        y_pred_fit = model.predict(X)
        y_pred = np.expm1(y_pred_fit) if use_log else y_pred_fit
        y_test = np.expm1(y_fit)      if use_log else y_fit
        r2     = None
        mae    = float(mean_absolute_error(y_test, y_pred))

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_).ravel()
        s = importances.sum() or 1
        importances = importances / s

    feature_importance = sorted(
        [{'feature': c, 'importance': round(float(i), 4)}
         for c, i in zip(feature_cols, importances)],
        key=lambda x: x['importance'], reverse=True
    )

    sample = min(40, len(y_test))
    return {
        'model':              model,
        'use_log':            use_log,
        'encoders':           encoders,
        'feature_types':      feature_types,
        'feature_cols':       feature_cols,
        'r2':                 round(r2, 4) if r2 is not None else None,
        'mae':                round(mae, 2),
        'n_rows':             int(len(y)),
        'mean_price':         round(float(y.mean()), 0),
        'feature_importance': feature_importance,
        'actual_vs_predicted': [
            {'actual': round(float(a), 2), 'predicted': round(float(p), 2)}
            for a, p in zip(y_test[:sample], y_pred[:sample])
        ]
    }


@app.route('/api/segment_train', methods=['POST'])
def segment_train():
    df = get_working_df()
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400

    data         = request.json or {}
    segment_col  = data.get('segment_column')   # e.g. "city"
    target_col   = data.get('target_column')
    feature_cols = data.get('feature_columns', [])
    model_type   = data.get('model_type', 'random_forest')

    if not segment_col or not target_col or not feature_cols:
        return jsonify({'error': 'segment_column, target_column and feature_columns are required'}), 400

    # Strip segment_col and target_col from features to avoid duplicate columns
    feature_cols = [c for c in feature_cols if c != segment_col and c != target_col]

    segments     = df[segment_col].dropna().unique().tolist()
    results      = {}
    segment_models = {}

    for seg_val in segments:
        mask   = df[segment_col].astype(str) == str(seg_val)
        df_seg = df[mask].copy()
        result = encode_and_fit(df_seg, feature_cols, target_col, model_type)
        if result:
            segment_models[str(seg_val)] = result
            results[str(seg_val)] = {k: v for k, v in result.items() if k != 'model'}

    if not results:
        return jsonify({'error': 'No segment had enough data (minimum 3 rows each)'}), 400

    uploaded_data['segment_models']  = segment_models
    uploaded_data['segment_col']     = segment_col
    uploaded_data['segment_target']  = target_col

    # Sort by R² descending
    sorted_results = dict(sorted(results.items(), key=lambda x: (x[1]['r2'] is not None, x[1]['r2'] or 0), reverse=True))

    return jsonify({
        'success':       True,
        'segment_col':   segment_col,
        'target_col':    target_col,
        'total_segments': len(sorted_results),
        'segments':      sorted_results
    })


@app.route('/api/segment_predict', methods=['POST'])
def segment_predict():
    segment_models = uploaded_data.get('segment_models')
    if not segment_models:
        return jsonify({'error': 'Segment models not trained yet'}), 400

    data      = request.json or {}
    seg_val   = str(data.get('segment_value', ''))
    in_values = data.get('values', {})

    if seg_val not in segment_models:
        return jsonify({'error': f'No model for segment "{seg_val}"'}), 400

    md = segment_models[seg_val]
    try:
        row = []
        for col in md['feature_cols']:
            val = in_values.get(col)
            enc = md['encoders'][col]
            if val is None or val == '':
                val = enc.get('mean', 0) if enc['type'] == 'numeric' else enc['classes'][0]
            if enc['type'] == 'categorical':
                classes = enc['classes']
                val_str = str(val)
                val     = float(classes.index(val_str)) if val_str in classes else 0.0
            else:
                val = float(val)
            row.append(val)

        prediction = md['model'].predict([row])[0]
        if md.get('use_log'):
            prediction = np.expm1(prediction)
        return jsonify({
            'prediction':    round(float(prediction), 2),
            'segment_value': seg_val,
            'segment_col':   uploaded_data.get('segment_col'),
            'target_col':    uploaded_data.get('segment_target'),
            'model_r2':      md['r2'],
            'model_mae':     md['mae'],
        })
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 400

# ── Undervalued Listings ─────────────────────────────────────────────────────
@app.route('/api/undervalued', methods=['POST'])
def undervalued():
    df_full    = uploaded_data.get('default')
    model_data = uploaded_data.get('default_model')

    if df_full is None:
        return jsonify({'error': 'No data loaded'}), 400
    if model_data is None:
        return jsonify({'error': 'Train a model first'}), 400

    data         = request.json or {}
    city_filter  = data.get('city_filter')
    top_n        = int(data.get('top_n', 9999))
    min_underval = float(data.get('min_underval_pct', 5))

    # Start with the globally filtered dataset (same as training data)
    # This ensures a model trained on Houses only scores Houses only
    _wdf = get_working_df(); df = _wdf if _wdf is not None else df_full.copy()

    # Apply additional city filter from the Undervalued sidebar
    city_col = uploaded_data.get('city_col')
    if city_filter and city_col and city_col in df.columns:
        cities = city_filter if isinstance(city_filter, list) else [str(city_filter)]
        df = df[df[city_col].astype(str).isin(cities)].reset_index(drop=True)

    model        = model_data['model']
    feature_cols = model_data['feature_cols']
    encoders     = model_data['encoders']
    target_col   = model_data['target_col']
    use_log      = model_data.get('use_log', False)

    # Build feature matrix for every row (skip rows missing target or key features)
    type_filter  = data.get('type_filter', [])
    require_sqft = data.get('require_sqft', True)
    min_beds     = data.get('min_beds')    # None = no filter
    min_baths    = data.get('min_baths')   # None = no filter

    # add address/street/mls for display if present
    display_cols = [c for c in ['street','district','city','real_mls','property_type','bedrooms','bathrooms','floor_area_sqft','summary_url'] if c in df.columns]
    df_score = df[feature_cols + [target_col] + [c for c in display_cols if c not in feature_cols + [target_col]]].copy()

    df_score = df_score.dropna(subset=[target_col]).reset_index(drop=True)
    df_score[target_col] = pd.to_numeric(df_score[target_col], errors='coerce')
    df_score = df_score.dropna(subset=[target_col]).reset_index(drop=True)

    # Exclude rows where sqft is missing — model has no basis to predict them
    sqft_col = next((c for c in ['floor_area_sqft','sqft','size_sqft','area_sqft'] if c in df_score.columns), None)
    if require_sqft and sqft_col:
        df_score = df_score.dropna(subset=[sqft_col]).reset_index(drop=True)

    # Fix city=null: use district as city fallback
    city_col_uv = uploaded_data.get('city_col', 'city')
    if city_col_uv in df_score.columns and 'district' in df_score.columns:
        df_score[city_col_uv] = df_score[city_col_uv].fillna(df_score['district'])

    # Filter by property type
    prop_type_col = next((c for c in ['property_type','type'] if c in df_score.columns), None)
    if type_filter and prop_type_col:
        df_score = df_score[df_score[prop_type_col].isin(type_filter)].reset_index(drop=True)

    # Filter by min beds/baths
    beds_col  = next((c for c in ['bedrooms','beds'] if c in df_score.columns), None)
    baths_col = next((c for c in ['bathrooms','baths'] if c in df_score.columns), None)
    if min_beds is not None and beds_col:
        df_score = df_score[pd.to_numeric(df_score[beds_col], errors='coerce').fillna(0) >= float(min_beds)].reset_index(drop=True)
    if min_baths is not None and baths_col:
        df_score = df_score[pd.to_numeric(df_score[baths_col], errors='coerce').fillna(0) >= float(min_baths)].reset_index(drop=True)

    # Apply categorical filters: {col: [allowed_values]}
    cat_filters = data.get('cat_filters', {})
    for col, allowed in cat_filters.items():
        if not allowed or col not in df_score.columns:
            continue
        allowed_set = set(str(v) for v in allowed)
        # For comma-separated cols: keep row if ANY of its values is in allowed_set
        def row_matches(cell):
            parts = set(v.strip() for v in str(cell).split(','))
            return bool(parts & allowed_set)
        mask = df_score[col].fillna('').apply(row_matches)
        df_score = df_score[mask].reset_index(drop=True)

    try:
        X_cols = []
        for col in feature_cols:
            col_data = df_score[col] if col in df_score.columns else pd.Series(['Unknown']*len(df_score))
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            col_data = col_data.reset_index(drop=True)
            enc = encoders[col]

            if enc['type'] == 'multi_ohe':
                ohe_arr, _ = ohe_multi_value(col_data, existing_values=enc.get('values', []))
                for k in range(ohe_arr.shape[1]):
                    X_cols.append(ohe_arr[:, k])
                continue
            elif enc['type'] == 'target_encoded':
                stats   = enc.get('_full_stats') or enc  # fallback
                med_map = enc.get('_median_map', {})
                ppsf_map= enc.get('_ppsf_map', {})
                fallback= enc.get('_global_median', float(df_score[target_col].median()))
                # Use stored stats from training
                stored  = model_data['encoders'][col]
                # re-derive from stored classes list + we stored stats separately
                # Actually use the full stats stored in encoders
                full_enc = model_data['encoders'][col]
                # target_encoded: we stored 'classes' as sorted keys of median dict
                # but we need the actual numeric values — stored in model_data['_te_stats']
                te_stats = model_data.get('_te_stats', {}).get(col)
                if te_stats:
                    med_val = col_data.map(te_stats['median']).fillna(te_stats['global_median']).astype(float)
                    X_cols.append(med_val.values)
                    if te_stats.get('ppsf'):
                        sqft_col = df_score.get('sqft', pd.Series([1000]*len(df_score)))
                        ppsf_val = col_data.map(te_stats['ppsf']).fillna(te_stats['global_median']/1000).astype(float)
                        X_cols.append(ppsf_val.values)
                else:
                    X_cols.append(np.zeros(len(df_score)))
            elif enc['type'] == 'categorical':
                classes = enc['classes']
                mapped  = col_data.astype(str).map({c: i for i, c in enumerate(classes)}).fillna(0).astype(float)
                X_cols.append(mapped.values)
            else:
                num = pd.to_numeric(col_data, errors='coerce')
                X_cols.append(num.fillna(enc.get('mean', 0)).values.astype(float))

        if len(X_cols) == 1:
            X = X_cols[0].reshape(-1, 1)
        else:
            X = np.column_stack(X_cols)

        y_pred_raw = model.predict(X)
        y_pred = np.expm1(y_pred_raw) if use_log else y_pred_raw
        y_actual = df_score[target_col].values.astype(float)

        gap     = y_pred - y_actual          # positive = undervalued (actual < predicted)
        gap_pct = gap / y_pred * 100

        # Prediction stability: std of individual tree predictions (RF only)
        pred_cv = np.zeros(len(X))   # coefficient of variation per row
        _m = model_data.get('model')
        if hasattr(_m, 'estimators_'):
            _tree_preds = np.array([t.predict(X) for t in _m.estimators_])  # shape (n_trees, n_rows)
            if use_log: _tree_preds = np.expm1(_tree_preds)
            _std  = _tree_preds.std(axis=0)
            _mean = np.abs(_tree_preds.mean(axis=0)) + 1
            pred_cv = _std / _mean  # lower = more stable

        # Compute confidence: how many training rows share the same district/city
        df_train = get_working_df()
        city_col_name = uploaded_data.get('city_col', 'city')
        district_counts = {}
        city_counts      = {}
        if df_train is not None:
            if 'district' in df_train.columns:
                district_counts = df_train['district'].value_counts().to_dict()
            if city_col_name in df_train.columns:
                city_counts = df_train[city_col_name].value_counts().to_dict()

        # Build results
        rows = []
        for i in range(len(df_score)):
            if gap_pct[i] < min_underval:
                continue
            dist  = str(df_score['district'].iloc[i]) if 'district' in df_score.columns else ''
            city  = str(df_score[city_col_name].iloc[i]) if city_col_name in df_score.columns else ''
            n_dist = district_counts.get(dist, 0)
            n_city = city_counts.get(city, 0)
            # Confidence: low if very few comparable listings in training data
            confidence = 'high' if n_dist >= 10 else 'medium' if n_dist >= 4 else 'low'
            cv_i = float(pred_cv[i])
            if   cv_i < 0.05: stability = 'stable'
            elif cv_i < 0.12: stability = 'moderate'
            else:              stability = 'unstable'
            row = {
                'actual':     round(float(y_actual[i]), 0),
                'predicted':  round(float(y_pred[i]), 0),
                'gap':        round(float(gap[i]), 0),
                'gap_pct':    round(float(gap_pct[i]), 1),
                'confidence': confidence,
                'n_district': n_dist,
                'n_city':     n_city,
                'stability':  stability,
                'pred_cv':    round(cv_i * 100, 1),
            }
            for c in display_cols:
                row[c] = str(df_score[c].iloc[i]) if c in df_score.columns else ''
            rows.append(mask_row(row))

        # Filter by confidence level
        min_confidence = data.get('min_confidence', 'medium')
        conf_order = {'low': 0, 'medium': 1, 'high': 2}
        min_conf_num = conf_order.get(min_confidence, 1)
        rows = [r for r in rows if conf_order.get(r['confidence'], 0) >= min_conf_num]

        # Sort by gap_pct descending, take top N
        rows.sort(key=lambda x: x['gap_pct'], reverse=True)
        rows = rows[:top_n]

        return jsonify({
            'success':        True,
            'total_found':    len(rows),
            'city_filter':    city_filter or 'All cities',
            'listings':       rows,
            'target_col':     target_col,
        })
    except Exception as e:
        import traceback
        return jsonify({'error': f'Scoring error: {str(e)}', 'trace': traceback.format_exc()}), 400



# ── Feature Analysis ──────────────────────────────────────────────────────────
@app.route('/api/feature_analysis', methods=['POST'])
def feature_analysis():
    """
    For each feature, compute:
    - correlation with target (if available)
    - R2 / MAE with feature included vs excluded (leave-one-out delta)
    - recommendation: include / exclude / optional
    """
    df = get_working_df()
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400

    data          = request.json or {}
    target_col    = data.get('target_column')
    feature_cols  = data.get('feature_columns', [])   # ALL features to analyse
    selected_cols = data.get('selected_columns')       # currently checked = baseline (None → use all)
    model_type    = data.get('model_type', 'random_forest')
    use_log       = data.get('use_log_target', True)
    remove_out    = data.get('remove_outliers', True)

    if not target_col or not feature_cols:
        return jsonify({'error': 'target_column and feature_columns required'}), 400

    feature_cols = [c for c in feature_cols if c != target_col]
    if selected_cols is not None:
        selected_cols = [c for c in selected_cols if c != target_col and c in feature_cols]
    else:
        selected_cols = feature_cols  # fallback: treat all as selected

    def quick_score(feat_list):
        """Train and CV-score a model with the given feature list. Returns (r2, mae) or (None, None)."""
        if not feat_list:
            return None, None
        df_m = df[feat_list + [target_col]].dropna(subset=[target_col]).reset_index(drop=True)
        if len(df_m) < 5:
            return None, None
        X_cols = []
        for col in feat_list:
            cd = df_m[col]
            if not pd.api.types.is_numeric_dtype(cd):
                cd = cd.fillna('Unknown')
                n_uniq = cd.nunique()
                if is_multi_value_col(cd):
                    ohe_arr, _ = ohe_multi_value(cd)
                    for k in range(ohe_arr.shape[1]):
                        X_cols.append(ohe_arr[:, k])
                elif n_uniq > 15:
                    enc_med, _, _ = target_encode(df_m.assign(**{col: cd}), col, target_col)
                    X_cols.append(enc_med.values)
                else:
                    le = LabelEncoder()
                    X_cols.append(le.fit_transform(cd.astype(str).values).astype(float))
            else:
                num = pd.to_numeric(cd, errors='coerce')
                X_cols.append(num.fillna(num.median() if num.notna().any() else 0).values.astype(float))
        if not X_cols:
            return None, None
        X = np.column_stack(X_cols) if len(X_cols) > 1 else X_cols[0].reshape(-1, 1)
        y = pd.to_numeric(df_m[target_col], errors='coerce').values.astype(float)
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid]
        if remove_out and len(y) >= 20:
            q1, q3 = np.percentile(y, 25), np.percentile(y, 75)
            mask = (y >= q1 - 3*(q3-q1)) & (y <= q3 + 3*(q3-q1))
            X, y = X[mask], y[mask]
        if len(y) < 5:
            return None, None
        y_fit = np.log1p(y) if (use_log and np.all(y > 0)) else y
        mdl = build_model(model_type)
        n_splits = min(5, max(2, len(y) // 5))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            preds = cross_val_predict(mdl, X, y_fit, cv=cv)
            if use_log and np.all(y > 0):
                preds = np.expm1(preds); y_eval = y
            else:
                y_eval = y
            r2  = float(r2_score(y_eval, preds))
            mae = float(mean_absolute_error(y_eval, preds))
            return round(r2, 4), round(mae, 2)
        except Exception:
            return None, None

    # Baseline: currently selected features
    base_r2, base_mae = quick_score(selected_cols)

    # Correlations with target
    corr_with_target = {}
    num_cols = get_numeric_columns(df)
    if target_col in num_cols:
        for col in feature_cols:
            if col in num_cols:
                try:
                    corr_val = float(df[[col, target_col]].dropna().corr().iloc[0, 1])
                    corr_with_target[col] = round(corr_val, 4) if not np.isnan(corr_val) else None
                except Exception:
                    corr_with_target[col] = None

    results = []
    selected_set = set(selected_cols)

    for col in feature_cols:
        in_baseline = col in selected_set
        corr        = corr_with_target.get(col)

        if in_baseline:
            # Feature is currently selected → show what happens WITHOUT it
            without     = [c for c in selected_cols if c != col]
            r2_alt, mae_alt = quick_score(without) if without else (None, None)
            # delta_r2 positive = feature contributes positively (removing hurts)
            delta_r2  = round((base_r2 - (r2_alt or 0)) * 100, 2) if base_r2 is not None and r2_alt is not None else None
            delta_mae = round((mae_alt or 0) - base_mae, 0)         if base_mae is not None and mae_alt is not None else None
            alt_label = 'R² without'
            r2_alt_pct = round(r2_alt * 100, 1) if r2_alt is not None else None
        else:
            # Feature is NOT selected → show what happens by ADDING it
            with_col    = selected_cols + [col]
            r2_alt, mae_alt = quick_score(with_col)
            # delta_r2 positive = adding this feature improves the model
            delta_r2  = round(((r2_alt or 0) - base_r2) * 100, 2) if base_r2 is not None and r2_alt is not None else None
            delta_mae = round(base_mae - (mae_alt or 0), 0)         if base_mae is not None and mae_alt is not None else None
            alt_label = 'R² if added'
            r2_alt_pct = round(r2_alt * 100, 1) if r2_alt is not None else None

        # Recommendation
        if delta_r2 is not None:
            if in_baseline:
                if delta_r2 >= 5:   rec, rec_color = 'keep — critical',     'success'
                elif delta_r2 >= 1: rec, rec_color = 'keep — helpful',      'accent'
                elif delta_r2 >= -1:rec, rec_color = 'optional',            'text2'
                else:               rec, rec_color = 'consider removing',   'danger'
            else:
                if delta_r2 >= 5:   rec, rec_color = 'add — big gain',      'success'
                elif delta_r2 >= 1: rec, rec_color = 'worth adding',        'accent'
                elif delta_r2 >= -1:rec, rec_color = 'minor effect',        'text2'
                else:               rec, rec_color = 'skip — hurts model',  'danger'
        else:
            rec, rec_color = 'unknown', 'text3'

        results.append({
            'feature':     col,
            'in_baseline': in_baseline,
            'corr':        corr,
            'base_r2':     round(base_r2  * 100, 1) if base_r2  is not None else None,
            'base_mae':    round(base_mae, 0)         if base_mae is not None else None,
            'r2_alt':      r2_alt_pct,
            'mae_alt':     round(mae_alt, 0)           if mae_alt  is not None else None,
            'alt_label':   alt_label,
            'delta_r2':    delta_r2,
            'delta_mae':   delta_mae,
            'recommendation': rec,
            'rec_color':   rec_color,
        })

    results.sort(key=lambda x: (x['delta_r2'] or -99), reverse=True)

    return jsonify({
        'success':    True,
        'base_r2':    round(base_r2 * 100, 1)  if base_r2  is not None else None,
        'base_mae':   round(base_mae, 0)         if base_mae is not None else None,
        'features':   results,
        'n_features': len(feature_cols),
    })


# ── Similar Listings ──────────────────────────────────────────────────────────
@app.route('/api/similar_listings', methods=['POST'])
def similar_listings():
    """
    Given a listing (by its index in df_score or by feature values),
    return the N most similar listings from the full dataset by Euclidean
    distance on numeric + encoded features.
    """
    df_full    = uploaded_data.get('default')
    model_data = uploaded_data.get('default_model')

    if df_full is None:
        return jsonify({'error': 'No data loaded'}), 400

    data         = request.json or {}
    target_vals  = data.get('values', {})     # feature values of the query listing
    top_n        = int(data.get('top_n', 5))
    target_col   = model_data['target_col'] if model_data else data.get('target_col', 'price')

    # Use model features if available; otherwise use whatever fields are in target_vals
    if model_data:
        feature_cols = model_data['feature_cols']
        encoders     = model_data['encoders']
    else:
        # No model: use fields present in target_vals as feature list
        # Build simple encoders on the fly
        df_tmp = uploaded_data.get('default') or df_full
        feature_cols = [k for k in target_vals.keys()
                        if k in df_tmp.columns and k != target_col]
        encoders = {}
        for col in feature_cols:
            cd = df_tmp[col].dropna()
            if pd.api.types.is_numeric_dtype(cd):
                encoders[col] = {'type': 'numeric', 'mean': float(cd.mean()),
                                 'min': float(cd.min()), 'max': float(cd.max())}
            else:
                encoders[col] = {'type': 'categorical', 'classes': sorted(cd.astype(str).unique().tolist())}

    try:
        df = get_working_df(); df = df if df is not None else df_full

        # Filter candidates: same city first, then nearby cities, then all
        # This ensures a Langley House never compares to Nanaimo Houses
        query_city = str(target_vals.get('city','') or target_vals.get('district','') or '')
        city_col_s = uploaded_data.get('city_col','city')
        nearby_cities = set(get_nearby_cities(query_city))

        def city_series(frame):
            col = city_col_s if city_col_s in frame.columns else None
            dist_col = 'district' if 'district' in frame.columns else None
            if col:
                return frame[col].fillna(frame[dist_col] if dist_col else '').astype(str)
            return pd.Series(['']*len(frame))

        query_prop = str(target_vals.get('property_type','') or '')

        df_same   = df[city_series(df) == query_city].reset_index(drop=True) if query_city else df.copy()
        df_nearby = df[city_series(df).isin(nearby_cities)].reset_index(drop=True)

        # Try to filter by same property_type within same city first
        prop_col = next((c for c in ['property_type','type'] if c in df.columns), None)
        if prop_col and query_prop:
            df_same_prop = df_same[df_same[prop_col].astype(str) == query_prop].reset_index(drop=True)
            df_near_prop = df_nearby[df_nearby[prop_col].astype(str) == query_prop].reset_index(drop=True)
        else:
            df_same_prop = df_same
            df_near_prop = df_nearby

        # Priority: same city+type → same city → nearby+type → nearby → all
        if len(df_same_prop) >= 3:
            df_use = df_same_prop
        elif len(df_same) >= 3:
            df_use = df_same
        elif len(df_near_prop) >= 3:
            df_use = df_near_prop
        elif len(df_nearby) >= 3:
            df_use = df_nearby
        else:
            df_use = df.reset_index(drop=True)

        def build_sim_vector(df_rows, qvals):
            Xp, qp, ws = [], [], []
            for col in feature_cols:
                enc  = encoders[col]
                qval = str(qvals.get(col,'') or '')
                if col in df_rows.columns:
                    cd = df_rows[col]
                    cd = (cd.iloc[:,0] if isinstance(cd, pd.DataFrame) else cd).reset_index(drop=True).fillna('').astype(str)
                else:
                    cd = pd.Series(['']*len(df_rows))
                if enc['type'] == 'multi_ohe':
                    ohe_vals = enc.get('values', [])
                    q_set    = set(v.strip() for v in qval.split(',') if v.strip())
                    q_vec    = np.array([1.0 if v in q_set else 0.0 for v in ohe_vals])
                    rows_vec = np.array([[1.0 if v in set(x.strip() for x in row.split(',')) else 0.0 for v in ohe_vals] for row in cd])
                    if len(ohe_vals):
                        Xp.append(rows_vec); qp.extend(q_vec); ws.extend([1.0/len(ohe_vals)]*len(ohe_vals))
                elif enc['type'] in ('categorical','target_encoded'):
                    if col in ('city','district'):
                        nearby = set(get_nearby_cities(qval))
                        def loc_score(v, qv=qval, nb=nearby):
                            return 1.0 if v==qv else (0.6 if v in nb else 0.0)
                        match = cd.apply(loc_score).values
                        w = 5.0 if col=='city' else 3.0
                    else:
                        match = (cd==qval).astype(float).values
                        w = 2.0 if col=='property_type' else 1.0
                    Xp.append(match.reshape(-1,1)); qp.append(1.0); ws.append(w)
                else:
                    num = pd.to_numeric(df_rows[col] if col in df_rows.columns else pd.Series([0]*len(df_rows)), errors='coerce').fillna(0)
                    qnum = float(qval) if qval.replace('.','',1).lstrip('-').isdigit() else 0.0
                    rng  = (num.max()-num.min()) or 1.0
                    Xp.append(((num-num.min())/rng).values.reshape(-1,1))
                    qp.append((qnum-num.min())/rng)
                    ws.append(2.0 if col=='floor_area_sqft' else 1.0)
            if not Xp: return None, None, None
            return np.hstack(Xp).astype(float), np.array(qp,dtype=float), np.array(ws,dtype=float)

        X_all, q, weights = build_sim_vector(df_use, target_vals)
        if X_all is None:
            return jsonify({'error': 'Could not build feature vectors'}), 400

        diff  = np.abs(X_all - q)
        dists = np.sum(diff * weights, axis=1) / weights.sum()

        top_idx = np.argsort(dists)[:top_n + 1]

        display_cols = [c for c in ['street','district','city','property_type',
                                     'bedrooms','bathrooms','floor_area_sqft',
                                     'summary_url','real_mls'] if c in df_use.columns]
        results = []
        for idx in top_idx:
            row = {
                'distance':   round(float(dists[idx]), 4),
                target_col:   round(float(pd.to_numeric(df_use[target_col].iloc[idx], errors='coerce') or 0), 0),
            }
            for c in display_cols:
                row[c] = str(df_use[c].iloc[idx]) if pd.notna(df_use[c].iloc[idx]) else ''
            results.append(row)

        return jsonify({
            'success':    True,
            'similar':    results,
            'target_col': target_col,
            'n_features': len(feature_cols),
            'scope':      ('same city+type' if df_use is df_same_prop
                           else 'same city' if df_use is df_same
                           else 'nearby+type' if df_use is df_near_prop
                           else 'nearby cities' if df_use is df_nearby
                           else 'all'),
        })
    except Exception as e:
        import traceback
        return jsonify({'error': f'Similar listings error: {str(e)}', 'trace': traceback.format_exc()}), 400

@app.route('/api/version')
def version():
    return jsonify({'version': '2.0', 'features': ['city_filter','similar_listings_v2','property_type_filter']})


# ── Comparable Listings (Comps) ───────────────────────────────────────────────
@app.route('/api/comps', methods=['POST'])
def get_comps():
    """Return all training listings from the same district (comparables used for confidence)."""
    df_train   = get_working_df()
    model_data = uploaded_data.get('default_model')

    if df_train is None:
        return jsonify({'error': 'No data loaded'}), 400

    data        = request.json or {}
    district    = data.get('district', '')
    city        = data.get('city', '')
    target_col  = model_data['target_col'] if model_data else 'price'
    # Query listing details used to filter comps

    # Listings from same district
    if 'district' in df_train.columns and district:
        df_comp = df_train[df_train['district'].astype(str) == str(district)].copy()
    elif city and uploaded_data.get('city_col') in df_train.columns:
        city_col = uploaded_data.get('city_col')
        df_comp = df_train[df_train[city_col].astype(str) == str(city)].copy()
    else:
        df_comp = df_train.copy()

    if len(df_comp) == 0:
        return jsonify({'comps': [], 'district': district, 'city': city, 'n': 0})

    display_cols = [c for c in ['street', 'district', 'city', 'property_type',
                                 'bedrooms', 'bathrooms', 'floor_area_sqft',
                                 'year_built', 'summary_url'] if c in df_comp.columns]

    # Get query listing details for filtering
    query_street   = data.get('street', '')
    query_sqft     = float(data.get('floor_area_sqft') or 0)
    query_beds     = str(data.get('bedrooms', ''))
    query_proptype = str(data.get('property_type', ''))

    comps = []
    for _, row in df_comp.iterrows():
        street = str(row.get('street', ''))

        # Exclude the listing itself
        if query_street and street == query_street:
            continue

        price = round(float(pd.to_numeric(row.get(target_col, 0), errors='coerce') or 0), 0)
        if price <= 0:
            continue

        # Filter by property type if known
        if query_proptype and 'property_type' in df_comp.columns:
            if str(row.get('property_type', '')) != query_proptype:
                continue

        # Filter by floor_area: within ±50% of query sqft
        if query_sqft > 0 and 'floor_area_sqft' in df_comp.columns:
            comp_sqft = float(pd.to_numeric(row.get('floor_area_sqft', 0), errors='coerce') or 0)
            if comp_sqft > 0:
                ratio = comp_sqft / query_sqft
                if ratio < 0.6 or ratio > 1.7:   # within ~50% size range
                    continue

        c = {'price': price}
        for col in display_cols:
            c[col] = str(row[col]) if pd.notna(row.get(col)) else ''
        comps.append(c)

    # If fewer than 2 comps in district after filtering, expand to whole city
    if len(comps) < 2 and city and uploaded_data.get('city_col') in df_train.columns:
        city_col_c = uploaded_data.get('city_col', 'city')
        df_city    = df_train[df_train[city_col_c].astype(str) == str(city)].copy()
        comps = []
        for _, row in df_city.iterrows():
            street = str(row.get('street', ''))
            if query_street and street == query_street:
                continue
            price = round(float(pd.to_numeric(row.get(target_col, 0), errors='coerce') or 0), 0)
            if price <= 0: continue
            if query_proptype and 'property_type' in df_city.columns:
                if str(row.get('property_type', '')) != query_proptype: continue
            if query_sqft > 0 and 'floor_area_sqft' in df_city.columns:
                comp_sqft = float(pd.to_numeric(row.get('floor_area_sqft', 0), errors='coerce') or 0)
                if comp_sqft > 0:
                    ratio = comp_sqft / query_sqft
                    if ratio < 0.6 or ratio > 1.7: continue
            c = {'price': price}
            for col in display_cols:
                c[col] = str(row[col]) if pd.notna(row.get(col)) else ''
            comps.append(c)

    # Filter by price range: within 3× of actual price
    query_actual = float(data.get('actual_price') or 0)
    if query_actual > 0:
        comps = [c for c in comps if 0 < c['price'] <= query_actual * 3 and c['price'] >= query_actual / 3]
    # Sort by price similarity
    if query_actual > 0:
        comps.sort(key=lambda x: abs(x['price'] - query_actual))
    else:
        comps.sort(key=lambda x: x['price'])
    comps = comps[:10]  # cap at 10

    return jsonify({
        'comps':    comps,
        'district': district,
        'city':     city,
        'n':        len(comps),
        'target_col': target_col,
    })


# ── Model Tuning / Grid Search ────────────────────────────────────────────────
import threading as _threading
_tune_jobs = {}  # job_id -> {'status': 'running'|'done'|'error', 'results': [], 'progress': 0, 'total': 0}

@app.route('/api/tune_status', methods=['GET'])
def tune_status():
    job_id = request.args.get('job_id')
    if not job_id or job_id not in _tune_jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(_tune_jobs[job_id])

@app.route('/api/tune_stop', methods=['POST'])
def tune_stop():
    job_id = request.args.get('job_id')
    if job_id and job_id in _tune_jobs:
        _tune_jobs[job_id]['status'] = 'stopped'
    return jsonify({'success': True})

@app.route('/api/tune', methods=['POST'])
def tune_model():
    """
    Exhaustive search over feature subsets + hyperparameters.
    Returns ranked combinations by R² (cross-validated).
    Streams progress via regular JSON (non-streaming for simplicity).
    """
    df = get_working_df()
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400

    data           = request.json or {}
    target_col     = data.get('target_column', 'price')
    candidate_cols = data.get('feature_columns', [])   # columns to search over
    use_log        = data.get('use_log_target', True)
    remove_out     = data.get('remove_outliers', True)
    min_features      = max(1, int(data.get('min_features', 1)))
    required_features = [f for f in data.get('required_features', []) if f in candidate_cols]
    max_features   = int(data.get('max_features', 4))
    if max_features >= 20: max_features = len(candidate_cols)  # 20 = all features
    max_features   = max(min_features, min(max_features, len(candidate_cols)))
    top_n          = int(data.get('top_n', 100))

    candidate_cols = [c for c in candidate_cols if c != target_col and c in df.columns]
    if not candidate_cols:
        return jsonify({'error': 'No candidate features provided'}), 400

    # Distributed search support: worker_id (0-based) and n_workers
    worker_id  = int(data.get('worker_id', 0))
    n_workers  = int(data.get('n_workers', 1))
    n_jobs          = int(data.get('n_jobs', -1))      # -1 = all CPU cores
    ncpu_avail      = _os.cpu_count() or 4
    # batch_size: how many combos to queue at once (can be large, e.g. 200)
    # n_threads:  how many run in parallel (capped at CPU cores)
    batch_size_req  = max(1, int(data.get('parallel_combos', 1)))
    n_threads       = min(max(1, int(data.get('n_threads', batch_size_req))), ncpu_avail * 2)
    parallel_combos = n_threads  # kept for backward compat
    batch_size_par  = batch_size_req  # queue size (user-facing "parallel combos")
    if n_workers < 1: n_workers = 1
    if worker_id >= n_workers: worker_id = 0

    import uuid as _uuid
    job_id = str(_uuid.uuid4())[:8]
    _tune_jobs[job_id] = {'status': 'running', 'results': [], 'progress': 0, 'total': 0,
                           'target_col': target_col}

    # 3-phase search:
    # Phase 1: ultra-fast RF screening (n=30) on ALL combos
    # Phase 2: full RF (n=200) on top 40%
    # Phase 3: GB + Linear on top 20%
    RF_FAST  = {'model': 'random_forest', 'n_estimators': 30,  'min_samples_leaf': 3, 'n_jobs': n_jobs, 'label': 'Random Forest'}
    RF_FULL  = {'model': 'random_forest', 'n_estimators': 200, 'min_samples_leaf': 2, 'n_jobs': n_jobs, 'label': 'Random Forest'}
    GB_FULL  = {'model': 'gradient_boosting', 'n_estimators': 200, 'learning_rate': 0.1,  'max_depth': 3, 'label': 'Grad. Boosting'}
    LIN      = {'model': 'linear', 'label': 'Linear'}
    MODEL_GRID = [RF_FAST, RF_FULL, GB_FULL, LIN]  # kept for compat

    def encode_features_with_stats(df_te, feat_list, target, df_tr=None):
        """Encode test set using stats from df_tr to avoid leakage."""
        X_parts = []
        src = df_tr if df_tr is not None else df_te
        for col in feat_list:
            cd = df_te[col].copy() if col in df_te.columns else pd.Series(['Unknown']*len(df_te))
            if not pd.api.types.is_numeric_dtype(cd):
                cd = cd.fillna('Unknown')
                cd_src = src[col].fillna('Unknown') if col in src.columns else cd
                if is_multi_value_col(cd_src):
                    all_vals = sorted(set(v.strip() for row in cd_src.astype(str) for v in row.split(',')
                                        if v.strip() and v.strip().lower() not in ('nan','unknown','')))
                    arr = np.array([[1.0 if v in set(x.strip() for x in str(row).split(',')) else 0.0
                                     for v in all_vals] for row in cd])
                    X_parts.append(arr)
                elif cd_src.nunique() > 15 and df_tr is not None:
                    # Target encode using TRAINING stats only
                    train_target = pd.to_numeric(src[target], errors='coerce')
                    med = src.assign(**{col: cd_src}).groupby(col)[target].median()
                    global_med = float(train_target.median())
                    X_parts.append(cd.map(med).fillna(global_med).values.reshape(-1,1).astype(float))
                else:
                    le = LabelEncoder()
                    le.fit(cd_src.astype(str))
                    mapped = cd.astype(str).map({c: i for i, c in enumerate(le.classes_)}).fillna(0)
                    X_parts.append(mapped.values.reshape(-1,1).astype(float))
            else:
                num = pd.to_numeric(cd, errors='coerce')
                src_num = pd.to_numeric(src[col] if col in src.columns else cd, errors='coerce')
                fill = src_num.median() if src_num.notna().any() else 0.0
                X_parts.append(num.fillna(fill).values.reshape(-1,1))
        return np.hstack(X_parts).astype(float) if X_parts else None

    def encode_features(df_m, feat_list, target):

        X_parts = []
        for col in feat_list:
            cd = df_m[col].copy() if col in df_m.columns else pd.Series(['Unknown']*len(df_m))
            if not pd.api.types.is_numeric_dtype(cd):
                cd = cd.fillna('Unknown')
                if is_multi_value_col(cd):
                    arr, _ = ohe_multi_value(cd)
                    X_parts.append(arr)
                elif cd.nunique() > 15:
                    # Include sqft from full df for ppsf feature (matches train_model behavior)
                    sqft_col = next((c for c in ['floor_area_sqft','sqft'] if c in df.columns), None)
                    df_te_src = df_m.copy()
                    if sqft_col and sqft_col not in df_te_src.columns:
                        df_te_src[sqft_col] = df[sqft_col].iloc[:len(df_te_src)].values
                    enc_med, enc_ppsf_te, _ = target_encode(df_te_src.assign(**{col: cd}), col, target)
                    X_parts.append(enc_med.values.reshape(-1,1))
                    if enc_ppsf_te is not None:
                        X_parts.append(enc_ppsf_te.values.reshape(-1,1))
                else:
                    le = LabelEncoder()
                    X_parts.append(le.fit_transform(cd.astype(str)).reshape(-1,1).astype(float))
            else:
                num = pd.to_numeric(cd, errors='coerce')
                X_parts.append(num.fillna(num.median() if num.notna().any() else 0).values.reshape(-1,1))
        return np.hstack(X_parts).astype(float) if X_parts else None

    def quick_cv(feat_list, model_cfg):
        df_m = df[feat_list + [target_col]].dropna(subset=[target_col]).reset_index(drop=True)
        df_m[target_col] = pd.to_numeric(df_m[target_col], errors='coerce')
        df_m = df_m.dropna(subset=[target_col]).reset_index(drop=True)
        if len(df_m) < 8:
            return None, None
        y = df_m[target_col].values.astype(float)
        if remove_out:
            q1,q3 = np.percentile(y,25), np.percentile(y,75)
            mask = (y >= q1-3*(q3-q1)) & (y <= q3+3*(q3-q1))
            df_m = df_m[mask].reset_index(drop=True)
            y = df_m[target_col].values.astype(float)
        if len(df_m) < 8:
            return None, None
        y_fit = np.log1p(y) if (use_log and np.all(y > 0)) else y
        # Encode once on full df_m (fast — acceptable ~2% optimism for screening)
        X = encode_features(df_m, feat_list, target_col)
        if X is None:
            return None, None
        mtype = model_cfg['model']
        # Same params as train_model for comparable results
        if mtype == 'random_forest':
            mdl = RandomForestRegressor(
                n_estimators=model_cfg.get('n_estimators', 200),
                min_samples_leaf=model_cfg.get('min_samples_leaf', 2),
                random_state=42, n_jobs=model_cfg.get('n_jobs', -1))
        elif mtype == 'gradient_boosting':
            mdl = GradientBoostingRegressor(
                n_estimators=model_cfg.get('n_estimators', 200),
                learning_rate=model_cfg.get('learning_rate', 0.1),
                max_depth=model_cfg.get('max_depth', 3),
                random_state=42)
        else:
            mdl = LinearRegression()
        n_splits = min(5, max(2, len(y)//5))
        cv_spl = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            preds = cross_val_predict(mdl, X, y_fit, cv=cv_spl)
            preds = np.expm1(preds) if (use_log and np.all(y > 0)) else preds
            r2  = float(r2_score(y, preds))
            mae = float(mean_absolute_error(y, preds))
            return round(r2, 4), round(mae, 0)
        except Exception:
            return None, None

    # Generate all feature subsets up to max_features size
    from itertools import combinations as _combs
    req_set = set(required_features)
    all_combos = []
    for size in range(min_features, max_features + 1):
        for combo in _combs(candidate_cols, size):
            # Skip combos that don't include ALL required features
            if req_set and not req_set.issubset(set(combo)):
                continue
            all_combos.append(list(combo))

    # No combo cap — user can stop anytime via Stop button
    # Distributed: each worker takes every n_workers-th combo

    # total will be set after 2-phase calculation below
    # Estimate for the initial response
    # Distribute combos across workers: worker i takes indices [i, i+n, i+2n, ...]
    if n_workers > 1:
        all_combos = all_combos[worker_id::n_workers]

    _phase2_est = max(1, int(len(all_combos) * 0.3))
    total = len(all_combos) + _phase2_est * 2
    _tune_jobs[job_id]['total'] = total

    ph2_size = max(1, int(len(all_combos) * 0.4))
    ph3_size = max(1, int(len(all_combos) * 0.2))
    exact_total = len(all_combos) + ph2_size + ph3_size * 2
    _tune_jobs[job_id]['total'] = exact_total

    def store(results):
        results.sort(key=lambda x: x['r2_raw'], reverse=True)
        _tune_jobs[job_id]['results'] = [
            {k:v for k,v in r.items() if k!='r2_raw'} for r in results[:top_n]]

    def run_tune():
        from joblib import Parallel as _Parallel, delayed as _delayed
        import time as _time, warnings as _warn
        import threading as _th
        all_results = []
        lock        = _th.Lock()

        ncpu = _os.cpu_count() or 4
        # n_threads: actual parallel workers (≤ CPU cores)
        # batch_size_par: how many combos queued per round (can be >> n_threads)
        # Each thread gets ncpu//n_threads cores per RF
        jobs_per_combo = max(1, ncpu // n_threads) if n_threads > 1 else (n_jobs if n_jobs != 0 else -1)
        RF_FAST_PAR = {**RF_FAST, 'n_jobs': jobs_per_combo}

        def eval_combo(feat_list):
            if _tune_jobs[job_id].get('status') == 'stopped':
                return feat_list, (None, None)
            return feat_list, quick_cv(feat_list, RF_FAST_PAR)

        # Phase 1: send batch_size_par combos per round, execute with n_threads workers
        _tune_jobs[job_id]['phase'] = f'Phase 1/3: fast screening (batch={batch_size_par}, threads={n_threads}, {jobs_per_combo} CPU each)'
        _tune_jobs[job_id]['start_time'] = _time.time()
        batches = [all_combos[i:i+batch_size_par] for i in range(0, len(all_combos), batch_size_par)]
        done = 0
        for batch in batches:
            if _tune_jobs[job_id].get('status') == 'stopped': break
            with _warn.catch_warnings():
                _warn.filterwarnings('ignore', category=UserWarning, module='sklearn')
                batch_res = _Parallel(n_jobs=n_threads, prefer='threads')(
                    _delayed(eval_combo)(fl) for fl in batch
                )
            for feat_list, (r2, mae) in batch_res:
                done += 1
                elapsed = _time.time() - _tune_jobs[job_id]['start_time']
                speed   = done / elapsed if elapsed > 0 else 0
                remaining = (_tune_jobs[job_id]['total'] - done) / speed if speed > 0 else 0
                _tune_jobs[job_id]['progress']  = done
                _tune_jobs[job_id]['eta']        = round(remaining)
                _tune_jobs[job_id]['speed']      = round(speed, 2)
                if r2 is None: continue
                with lock:
                    all_results.append({'features': feat_list, 'n_features': len(feat_list),
                                        'model': RF_FULL['label'], 'r2': round(r2*100,1),
                                        'mae': round(mae,0), 'r2_raw': r2})
            store(all_results)

        # Phase 2: top 40% with full RF (n=200)
        if _tune_jobs[job_id].get('status') != 'stopped':
            _tune_jobs[job_id]['phase'] = 'Phase 2/3: refining top 40% with RF n=200'
            if 'start_time' not in _tune_jobs[job_id]: _tune_jobs[job_id]['start_time'] = _time.time()
            top_ph2 = [r['features'] for r in
                       sorted(all_results, key=lambda x: x['r2_raw'], reverse=True)[:ph2_size]]
            for feat_list in top_ph2:
                if _tune_jobs[job_id].get('status') == 'stopped': break
                r2, mae = quick_cv(feat_list, RF_FULL)
                done += 1
                _elapsed = _time.time() - _tune_jobs[job_id].get('start_time', _time.time())
                _speed   = done / _elapsed if _elapsed > 0 else 0
                _tune_jobs[job_id]['progress'] = done
                _tune_jobs[job_id]['eta']       = round((_tune_jobs[job_id]['total'] - done) / _speed) if _speed > 0 else 0
                _tune_jobs[job_id]['speed']     = round(_speed, 2)
                if r2 is None: continue
                # Update result for this feature set
                for existing in all_results:
                    if existing['features'] == feat_list and existing['model'] == RF_FULL['label']:
                        existing['r2'] = round(r2*100,1)
                        existing['mae'] = round(mae,0)
                        existing['r2_raw'] = r2
                        break
                else:
                    all_results.append({'features': feat_list, 'n_features': len(feat_list),
                                        'model': RF_FULL['label'], 'r2': round(r2*100,1),
                                        'mae': round(mae,0), 'r2_raw': r2})
                store(all_results)

        # Phase 3: top 20% with GB + Linear
        if _tune_jobs[job_id].get('status') != 'stopped':
            _tune_jobs[job_id]['phase'] = 'Phase 3/3: testing top 20% with Grad.Boosting + Linear'
            top_ph3 = [r['features'] for r in
                       sorted(all_results, key=lambda x: x['r2_raw'], reverse=True)[:ph3_size]]
            for feat_list in top_ph3:
                if _tune_jobs[job_id].get('status') == 'stopped': break
                for mcfg in [GB_FULL, LIN]:
                    if _tune_jobs[job_id].get('status') == 'stopped': break
                    r2, mae = quick_cv(feat_list, mcfg)
                    done += 1
                    _elapsed = _time.time() - _tune_jobs[job_id].get('start_time', _time.time())
                    _speed   = done / _elapsed if _elapsed > 0 else 0
                    _tune_jobs[job_id]['progress'] = done
                    _tune_jobs[job_id]['eta']       = round((_tune_jobs[job_id]['total'] - done) / _speed) if _speed > 0 else 0
                    _tune_jobs[job_id]['speed']     = round(_speed, 2)
                    if r2 is None: continue
                    all_results.append({'features': feat_list, 'n_features': len(feat_list),
                                        'model': mcfg['label'], 'r2': round(r2*100,1),
                                        'mae': round(mae,0), 'r2_raw': r2})
                    store(all_results)

        if _tune_jobs[job_id].get('status') != 'stopped':
            _tune_jobs[job_id]['status'] = 'done'

    t = _threading.Thread(target=run_tune, daemon=True)
    t.start()

    return jsonify({'job_id': job_id, 'total': total, 'status': 'running'})


@app.route('/api/tune_estimate', methods=['POST'])
def tune_estimate():
    from itertools import combinations as _combs
    data           = request.json or {}
    candidate_cols = data.get('feature_columns', [])
    min_features   = max(1, int(data.get('min_features', 1)))
    max_features   = int(data.get('max_features', 4))
    df = get_working_df()
    n_rows = len(df) if df is not None else 0
    n_cols = len(candidate_cols)
    if max_features >= 20: max_features = n_cols
    max_features = max(min_features, min(max_features, n_cols))
    required_feat = data.get('required_features', [])
    req_set_e = set(required_feat)
    if req_set_e:
        # Count only combos that include all required features
        n_req = len(req_set_e)
        free_cols = n_cols - n_req
        n_combos = sum(
            len(list(_combs(range(free_cols), k - n_req)))
            for k in range(max(min_features, n_req), max_features + 1)
            if k >= n_req and (k - n_req) <= free_cols
        )
    else:
        n_combos = sum(len(list(_combs(range(n_cols), k))) for k in range(min_features, max_features+1))
    phase2 = max(1, int(n_combos * 0.3))
    total_tests = n_combos + phase2 * 2
    secs_per_test = max(0.3, 1.2 * n_rows / 100)
    est_seconds = total_tests * secs_per_test
    return jsonify({'n_combos': n_combos, 'total_tests': total_tests,
                    'est_seconds': round(est_seconds),
                    'est_minutes': round(est_seconds / 60, 1), 'n_rows': n_rows})


@app.route('/api/tune_merge', methods=['POST'])
def tune_merge():
    """Merge results from multiple tune jobs (distributed workers) into one ranked list."""
    data    = request.json or {}
    job_ids = data.get('job_ids', [])
    top_n   = int(data.get('top_n', 100))

    all_results = []
    for jid in job_ids:
        job = _tune_jobs.get(jid)
        if not job:
            continue
        for r in job.get('results', []):
            all_results.append(r)

    # Deduplicate (same features + model may appear across workers in phase 2)
    seen = set()
    unique = []
    for r in all_results:
        key = (tuple(sorted(r['features'])), r['model'])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    unique.sort(key=lambda x: x['r2'], reverse=True)
    return jsonify({
        'success': True,
        'results': unique[:top_n],
        'total':   len(unique),
        'workers': len(job_ids),
    })


# ── Saved Models ──────────────────────────────────────────────────────────────
import pickle as _pickle, datetime as _dt, os as _os
_MODELS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'saved_models')
_os.makedirs(_MODELS_DIR, exist_ok=True)

def _model_path(mid): return _os.path.join(_MODELS_DIR, f'{mid}.pkl')

def _list_saved():
    out = []
    for fn in sorted(_os.listdir(_MODELS_DIR), reverse=True):
        if not fn.endswith('.pkl'): continue
        try:
            with open(_os.path.join(_MODELS_DIR, fn), 'rb') as f:
                m = _pickle.load(f)
            out.append({k: m[k] for k in ('id','name','model_type','target_col','feature_cols',
                        'r2_cv','mae_cv','r2_score','filters','n_rows','created') if k in m})
        except Exception: pass
    return out

@app.route('/api/models', methods=['GET'])
def api_get_models():
    return jsonify({'models': _list_saved()})

@app.route('/api/models/save', methods=['POST'])
def api_save_model():
    md = uploaded_data.get('default_model')
    if not md: return jsonify({'error': 'No trained model'}), 400
    data = request.json or {}
    name = data.get('name','').strip() or 'Model '+_dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    mid  = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    payload = {
        'id': mid, 'name': name,
        'model_type':   md.get('model_type','random_forest'),
        'target_col':   md.get('target_col',''),
        'feature_cols': md.get('feature_cols',[]),
        'r2_cv':        md.get('r2_cv'), 'mae_cv': md.get('mae_cv'),
        'r2_score':     md.get('r2_score'),
        'filters':      uploaded_data.get('global_filters',{}),
        'n_rows':       int(uploaded_data.get('n_rows',0)),
        'created':      _dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'model_data':   md,
    }
    with open(_model_path(mid), 'wb') as f: _pickle.dump(payload, f)
    return jsonify({'success': True, 'id': mid, 'name': name})

@app.route('/api/models/load', methods=['POST'])
def api_load_model():
    mid = (request.json or {}).get('id','')
    path = _model_path(mid)
    if not _os.path.exists(path): return jsonify({'error': 'Model not found'}), 404
    with open(path, 'rb') as f: payload = _pickle.load(f)
    uploaded_data['default_model'] = payload['model_data']
    md = payload['model_data']
    return jsonify({
        'success': True, 'id': payload['id'], 'name': payload['name'],
        'model_type': payload.get('model_type',''), 'target_col': payload.get('target_col',''),
        'feature_cols': payload.get('feature_cols',[]), 'r2_cv': payload.get('r2_cv'),
        'mae_cv': payload.get('mae_cv'), 'r2_score': payload.get('r2_score'),
        'filters': payload.get('filters',{}), 'n_rows': payload.get('n_rows',0),
        'created': payload.get('created',''),
        'feature_importance': md.get('feature_importance',[]),
        'train_size': md.get('train_size',0), 'test_size': md.get('test_size',0),
    })

@app.route('/api/models/delete', methods=['POST'])
def api_delete_model():
    mid = (request.json or {}).get('id','')
    path = _model_path(mid)
    if _os.path.exists(path): _os.remove(path)
    return jsonify({'success': True})

@app.route('/api/models/rename', methods=['POST'])
def api_rename_model():
    data = request.json or {}
    mid, new_name = data.get('id',''), data.get('name','').strip()
    path = _model_path(mid)
    if not _os.path.exists(path): return jsonify({'error':'Not found'}), 404
    with open(path,'rb') as f: payload = _pickle.load(f)
    payload['name'] = new_name or payload['name']
    with open(path,'wb') as f: _pickle.dump(payload, f)
    return jsonify({'success': True})


# ── Tune Batch (client-driven) ────────────────────────────────────────────────
@app.route('/api/tune_batch', methods=['POST'])
def tune_batch():
    """
    Client sends a batch of feature combinations, server evaluates them all
    in parallel and returns results immediately.
    No background job needed — client controls the queue.
    """
    df = get_working_df()
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400

    data             = request.json or {}
    target_col       = data.get('target_column', 'price')
    combos           = data.get('combos', [])          # list of feature lists
    model_type_req   = data.get('model_type', 'rf_fast')  # rf_fast | rf_full | gb | linear
    use_log          = data.get('use_log_target', True)
    remove_out       = data.get('remove_outliers', True)
    n_threads        = min(max(1, int(data.get('n_threads', 1))), (_os.cpu_count() or 4) * 2)
    n_jobs_per_model = int(data.get('n_jobs', -1))

    if not combos:
        return jsonify({'results': [], 'n': 0})

    _ncpu  = _os.cpu_count() or 4
    # Auto n_threads: use batch_size as thread count, capped at ncpu
    n_threads = min(max(1, n_threads), _ncpu)
    _jpm   = max(1, _ncpu // n_threads)  # cores per model
    MODEL_CFGS = {
        'rf_fast': {'model': 'random_forest',     'n_estimators': 30,  'min_samples_leaf': 3,
                    'n_jobs': _jpm, 'label': 'Random Forest'},
        'rf_full': {'model': 'random_forest',     'n_estimators': 200, 'min_samples_leaf': 2,
                    'n_jobs': _jpm, 'label': 'Random Forest'},
        'gb':      {'model': 'gradient_boosting', 'n_estimators': 200, 'learning_rate': 0.1,
                    'max_depth': 3, 'label': 'Grad. Boosting'},
        'gb_deep': {'model': 'gradient_boosting', 'n_estimators': 300, 'learning_rate': 0.05,
                    'max_depth': 4, 'subsample': 0.8, 'min_samples_leaf': 3, 'label': 'GB deep'},
        'stacking':{'model': 'stacking', 'label': 'Stacking'},
        'linear':  {'model': 'linear', 'label': 'Linear'},
    }
    mcfg = MODEL_CFGS.get(model_type_req, MODEL_CFGS['rf_fast'])

    # Inline quick_cv (same logic as tune_model)
    def _quick_cv(feat_list):
        try:
            # Use full df for target encoding (includes sqft for ppsf) then slice to feat_list rows
            cols_needed = [c for c in feat_list if c in df.columns]
            df_m = df[cols_needed + [target_col]].dropna(subset=[target_col]).reset_index(drop=True)
            df_m[target_col] = pd.to_numeric(df_m[target_col], errors='coerce')
            df_m = df_m.dropna(subset=[target_col]).reset_index(drop=True)
            if len(df_m) < 8: return feat_list, None, None
            y = df_m[target_col].values.astype(float)
            if remove_out:
                q1,q3 = np.percentile(y,25),np.percentile(y,75)
                m = (y>=q1-3*(q3-q1))&(y<=q3+3*(q3-q1))
                df_m,y = df_m[m].reset_index(drop=True),y[m]
            if len(df_m) < 8: return feat_list, None, None
            y_fit = np.log1p(y) if (use_log and np.all(y>0)) else y
            X_parts = []
            for col in feat_list:
                if col not in df_m.columns: continue
                cd = df_m[col].copy()
                if not pd.api.types.is_numeric_dtype(cd):
                    cd = cd.fillna('Unknown')
                    if is_multi_value_col(cd):
                        arr, _ = ohe_multi_value(cd)
                        X_parts.append(arr)
                    elif cd.nunique() > 15:
                        # Use same target_encode as train_model (includes ppsf)
                        enc_med, enc_ppsf, _ = target_encode(df_m.assign(**{col: cd}), col, target_col)
                        X_parts.append(enc_med.values.reshape(-1,1))
                        if enc_ppsf is not None:
                            X_parts.append(enc_ppsf.values.reshape(-1,1))
                    else:
                        le = LabelEncoder()
                        X_parts.append(le.fit_transform(cd.astype(str)).reshape(-1,1).astype(float))
                else:
                    num = pd.to_numeric(cd, errors='coerce')
                    X_parts.append(num.fillna(num.median() if num.notna().any() else 0).values.reshape(-1,1))
            if not X_parts: return feat_list, None, None
            X = np.hstack(X_parts).astype(float)
            mt = mcfg['model']
            if mt == 'random_forest':
                mdl = RandomForestRegressor(n_estimators=mcfg.get('n_estimators',30),
                        min_samples_leaf=mcfg.get('min_samples_leaf',3),
                        n_jobs=mcfg.get('n_jobs',-1), random_state=42)
            elif mt == 'gradient_boosting':
                mdl = GradientBoostingRegressor(n_estimators=mcfg.get('n_estimators',200),
                        learning_rate=mcfg.get('learning_rate',0.1), max_depth=mcfg.get('max_depth',3),
                        subsample=mcfg.get('subsample',1.0), min_samples_leaf=mcfg.get('min_samples_leaf',1),
                        random_state=42)
            elif mt == 'stacking':
                from sklearn.ensemble import StackingRegressor
                mdl = StackingRegressor(estimators=[('rf',RandomForestRegressor(n_estimators=100,min_samples_leaf=2,random_state=42,n_jobs=1)),('gb',GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42))],final_estimator=LinearRegression(),cv=3)
            else:
                mdl = LinearRegression()
            n_sp = min(5, max(2, len(y)//5))
            cv   = KFold(n_splits=n_sp, shuffle=True, random_state=42)
            preds = cross_val_predict(mdl, X, y_fit, cv=cv)
            if use_log and np.all(y>0): preds = np.expm1(preds)
            # Fit once on full data to get feature importances
            imps = None
            try:
                mdl.fit(X, y_fit)
                if hasattr(mdl, 'feature_importances_'):
                    raw_imp = mdl.feature_importances_
                elif hasattr(mdl, 'estimators_'):
                    ies = [e.feature_importances_ for e in mdl.estimators_ if hasattr(e,'feature_importances_')]
                    raw_imp = np.mean(ies, axis=0) if ies else None
                else:
                    raw_imp = None
                if raw_imp is not None:
                    s = raw_imp.sum() or 1
                    imps = {feat_list[j]: round(float(raw_imp[j]/s*100), 1) for j in range(len(feat_list))}
            except Exception:
                imps = None
            return feat_list, round(float(r2_score(y,preds))*100,1), round(float(mean_absolute_error(y,preds)),0), imps
        except Exception:
            return feat_list, None, None, None

    from joblib import Parallel as _P, delayed as _d
    import warnings as _w
    with _w.catch_warnings():
        _w.filterwarnings('ignore')
        raw = _P(n_jobs=n_threads, prefer='threads')(_d(_quick_cv)(fl) for fl in combos)

    results = [{'features': fl, 'n_features': len(fl),
                'model': mcfg['label'], 'r2': r2, 'mae': mae,
                'importance': imps or {}}
               for fl, r2, mae, imps in raw if r2 is not None]
    ncpu = _os.cpu_count() or 1
    jobs_info = f'{n_threads} threads × {_jpm} cores each = {min(n_threads*_jpm,_ncpu)}/{_ncpu} cores used'
    # Cache results for reuse across restarts
    df_w = get_working_df()
    if df_w is not None:
        _ck = _tune_cache_key(df_w, combos, model_type_req, use_log, remove_out)
        cached = _tune_cache_read(_ck)
        if cached is not None:
            # Return cached — but only the combos that match this request
            feat_keys = {tuple(sorted(c)) for c in combos}
            cached_results = [r for r in cached if tuple(sorted(r['features'])) in feat_keys]
            if len(cached_results) == len(combos):
                return jsonify({'results': cached_results, 'n': len(combos),
                               'evaluated': len(cached_results), 'cached': True,
                               'cpu_info': 'cached result'})
        _tune_cache_write(_ck, results)
    return jsonify({'results': results, 'n': len(combos), 'evaluated': len(results), 'cpu_info': jobs_info})


# ── Tuning Cache ──────────────────────────────────────────────────────────────
import hashlib as _hashlib, json as _json

def _tune_cache_key(df, combos, model_type, use_log, remove_outliers):
    """Hash of: data fingerprint + sorted combos + model params."""
    data_fp = str(len(df)) + '|' + '|'.join(sorted(df.columns))
    combos_fp = _json.dumps(sorted([sorted(c) for c in combos]))
    raw = f"{data_fp}|{combos_fp}|{model_type}|{use_log}|{remove_outliers}"
    return _hashlib.md5(raw.encode()).hexdigest()[:16]

def _tune_cache_path(key):
    cache_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'tune_cache')
    _os.makedirs(cache_dir, exist_ok=True)
    return _os.path.join(cache_dir, f'{key}.json')

def _tune_cache_read(key):
    path = _tune_cache_path(key)
    if _os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return _json.load(f)
        except Exception:
            pass
    return None

def _tune_cache_write(key, results):
    try:
        with open(_tune_cache_path(key), 'w', encoding='utf-8') as f:
            _json.dump(results, f)
    except Exception:
        pass


@app.route('/api/typical_values', methods=['POST'])
def typical_values():
    """Return median/mode for each feature column — used in Predict Price form hints."""
    df = get_working_df()
    if df is None:
        return jsonify({'error': 'No data'}), 400
    data = request.json or {}
    feature_cols = data.get('feature_cols', [])
    target_col   = data.get('target_col', 'price')
    result = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        if pd.api.types.is_numeric_dtype(series):
            result[col] = {
                'type': 'numeric',
                'median': round(float(series.median()), 2),
                'mean':   round(float(series.mean()), 2),
                'min':    round(float(series.min()), 2),
                'max':    round(float(series.max()), 2),
                'p25':    round(float(series.quantile(0.25)), 2),
                'p75':    round(float(series.quantile(0.75)), 2),
            }
        else:
            vc = series.astype(str).value_counts()
            result[col] = {
                'type':     'categorical',
                'mode':     vc.index[0] if len(vc) else '',
                'top3':     vc.head(3).index.tolist(),
                'n_unique': int(series.nunique()),
            }
    # Also return most common full row (row closest to median price)
    if target_col in df.columns:
        target_num = pd.to_numeric(df[target_col], errors='coerce').dropna()
        median_price = float(target_num.median())
        df_c = df.copy()
        df_c['_dist'] = (pd.to_numeric(df_c[target_col], errors='coerce') - median_price).abs()
        typical_row = df_c.nsmallest(1, '_dist')
        if len(typical_row):
            row_vals = {}
            for col in feature_cols:
                if col in typical_row.columns:
                    v = typical_row[col].iloc[0]
                    row_vals[col] = '' if pd.isna(v) else (round(float(v), 2) if pd.api.types.is_numeric_dtype(df[col]) else str(v))
            result['_typical_row'] = row_vals
    return jsonify(result)


@app.route('/api/upload_extra', methods=['POST'])
def upload_extra():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file  = request.files['file']
    ftype = request.form.get('file_type', 'listings')
    weight = float(request.form.get('weight', 3.0 if ftype=='sold' else 1.0))
    try:
        cb = file.read()
        try:    new_df = pd.read_csv(io.BytesIO(cb), encoding='utf-8', sep=None, engine='python')
        except: new_df = pd.read_csv(io.BytesIO(cb), encoding='latin-1', sep=None, engine='python')
        new_df = clean_dataframe(new_df)
        new_df = add_derived_features(new_df)
        new_df['_source']        = file.filename
        new_df['_file_type']     = ftype
        new_df['_sample_weight'] = weight
        existing = uploaded_data.get('default')
        if existing is None:
            uploaded_data['default'] = new_df
        else:
            if '_source' not in existing.columns:
                existing = existing.copy()
                existing['_source']        = 'primary'
                existing['_file_type']     = 'listings'
                existing['_sample_weight'] = 1.0
                uploaded_data['default']   = existing
            merged = pd.concat([existing, new_df], ignore_index=True, sort=False)
            uploaded_data['default'] = merged
        df = uploaded_data['default']
        num_cols  = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in ['listing_id','_sample_weight']]
        cat_cols  = [c for c in df.select_dtypes(exclude=[np.number]).columns
                     if c not in ['_source','_file_type'] and df[c].nunique() <= 200]
        files_summary = df['_source'].value_counts().to_dict() if '_source' in df.columns else {}
        return jsonify({'success': True, 'rows': int(len(df)), 'new_rows': int(len(new_df)),
                        'file_type': ftype, 'files': files_summary,
                        'numeric_columns': num_cols, 'categorical_columns': cat_cols})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/overvalued', methods=['POST'])
def overvalued():
    data       = request.json or {}
    df         = get_working_df()
    model_data = uploaded_data.get('default_model')
    if df is None or not model_data:
        return jsonify({'error': 'No data or model'}), 400
    top_n        = int(data.get('top_n', 50))
    min_over_pct = float(data.get('min_over_pct', 5))
    city_filter  = data.get('city', '')
    prop_filter  = data.get('property_type', '')
    target_col   = model_data['target_col']
    feature_cols = model_data['feature_cols']
    encoders     = model_data['encoders']
    use_log      = model_data.get('use_log', True)
    model        = model_data['model']

    df_score = df.copy()
    df_score[target_col] = pd.to_numeric(df_score[target_col], errors='coerce')
    df_score = df_score.dropna(subset=[target_col]).reset_index(drop=True)
    # Remove price outliers before scoring
    y_raw = df_score[target_col].values.astype(float)
    q1,q3 = np.percentile(y_raw,25), np.percentile(y_raw,75)
    df_score = df_score[(y_raw>=q1-3*(q3-q1))&(y_raw<=q3+3*(q3-q1))].reset_index(drop=True)
    if city_filter and 'city' in df_score.columns:
        df_score = df_score[df_score['city'].astype(str)==city_filter].reset_index(drop=True)
    if prop_filter and 'property_type' in df_score.columns:
        df_score = df_score[df_score['property_type'].astype(str)==prop_filter].reset_index(drop=True)
    if len(df_score) == 0: return jsonify({'listings':[],'total':0})

    X_cols = []
    for col in feature_cols:
        col_data = df_score[col] if col in df_score.columns else pd.Series(['Unknown']*len(df_score))
        enc = encoders.get(col, {})
        enc_type = enc.get('type','numeric')
        if enc_type == 'multi_ohe':
            arr, _ = ohe_multi_value(col_data, existing_values=enc.get('values',[]))
            for k in range(arr.shape[1]): X_cols.append(arr[:,k])
            continue
        elif enc_type == 'target_encoded':
            te = model_data.get('_te_stats',{}).get(col)
            if te:
                X_cols.append(col_data.map(te['median']).fillna(te['global_median']).astype(float).values)
                if te.get('ppsf'):
                    X_cols.append(col_data.map(te['ppsf']).fillna(te['global_median']/1000).astype(float).values)
            else:
                X_cols.append(np.zeros(len(df_score)))
        elif enc_type == 'categorical':
            X_cols.append(col_data.astype(str).map({c:i for i,c in enumerate(enc['classes'])}).fillna(0).astype(float).values)
        else:
            num = pd.to_numeric(col_data, errors='coerce')
            X_cols.append(num.fillna(enc.get('mean',0)).values.astype(float))
    if not X_cols: return jsonify({'listings':[],'total':0})
    X = np.column_stack(X_cols) if len(X_cols)>1 else X_cols[0].reshape(-1,1)
    y_pred_raw = model.predict(X)
    y_pred   = np.expm1(y_pred_raw) if use_log else y_pred_raw
    y_actual = df_score[target_col].values.astype(float)
    gap      = y_actual - y_pred
    gap_pct  = gap / np.maximum(y_pred, 1) * 100

    pred_cv = np.zeros(len(X))
    if hasattr(model, 'estimators_'):
        tp = np.array([t.predict(X) for t in model.estimators_])
        if use_log: tp = np.expm1(tp)
        pred_cv = tp.std(axis=0) / (np.abs(tp.mean(axis=0))+1)

    df_train = get_working_df()
    dist_counts = df_train['district'].value_counts().to_dict() if df_train is not None and 'district' in df_train.columns else {}
    disp = [c for c in ['street','district','city','real_mls','property_type','bedrooms','bathrooms','floor_area_sqft','summary_url','year_built'] if c in df_score.columns]
    rows = []
    for i in range(len(df_score)):
        if gap_pct[i] < min_over_pct: continue
        dist   = str(df_score['district'].iloc[i]) if 'district' in df_score.columns else ''
        n_dist = dist_counts.get(dist, 0)
        conf   = 'high' if n_dist>=10 else 'medium' if n_dist>=4 else 'low'
        cv_i   = float(pred_cv[i])
        stab   = 'stable' if cv_i<0.05 else 'moderate' if cv_i<0.12 else 'unstable'
        row    = {'actual':round(float(y_actual[i]),0),'predicted':round(float(y_pred[i]),0),
                  'gap':round(float(gap[i]),0),'gap_pct':round(float(gap_pct[i]),1),
                  'confidence':conf,'n_district':n_dist,'stability':stab,'pred_cv':round(cv_i*100,1)}
        for c in disp: row[c] = str(df_score[c].iloc[i]) if c in df_score.columns else ''
        rows.append(mask_row(row))
    rows = sorted(rows, key=lambda x: -x['gap_pct'])[:top_n]
    return jsonify({'listings': rows, 'total': len(rows)})


# ── File manager ──────────────────────────────────────────────────────────────
@app.route('/api/files', methods=['GET'])
def list_files():
    df = uploaded_data.get('default')
    if df is None: return jsonify({'files': [], 'total_rows': 0})
    if '_source' not in df.columns:
        return jsonify({'files': [{'name':'primary','rows':len(df),'type':'listings','weight':1.0,'excluded':False}], 'total_rows': len(df)})
    files = []
    for src in df['_source'].unique():
        grp    = df[df['_source']==src]
        ftype  = str(grp['_file_type'].iloc[0])  if '_file_type'  in grp.columns else 'listings'
        weight = float(grp['_sample_weight'].iloc[0]) if '_sample_weight' in grp.columns else 1.0
        files.append({'name':src,'rows':len(grp),'type':ftype,'weight':weight,'excluded':src in _excluded_sources})
    active = df[~df['_source'].isin(_excluded_sources)] if _excluded_sources else df
    return jsonify({'files': files, 'total_rows': len(active)})

@app.route('/api/files/toggle', methods=['POST'])
def toggle_file():
    src = (request.json or {}).get('source','')
    if src in _excluded_sources: _excluded_sources.discard(src)
    else: _excluded_sources.add(src)
    return jsonify({'excluded': src in _excluded_sources, 'source': src})

@app.route('/api/files/remove', methods=['POST'])
def remove_file():
    src = (request.json or {}).get('source','')
    df = uploaded_data.get('default')
    if df is not None and '_source' in df.columns:
        uploaded_data['default'] = df[df['_source']!=src].reset_index(drop=True)
    _excluded_sources.discard(src)
    return jsonify({'success':True})

@app.route('/api/files/set_weight', methods=['POST'])
def set_file_weight():
    data = request.json or {}
    src, weight = data.get('source',''), float(data.get('weight',1.0))
    df = uploaded_data.get('default')
    if df is not None and '_source' in df.columns:
        df.loc[df['_source']==src,'_sample_weight'] = weight
        uploaded_data['default'] = df
    return jsonify({'success':True})


@app.route('/api/upload_meta', methods=['GET'])
def upload_meta():
    """Return column metadata for already-loaded data (no file needed)."""
    df = uploaded_data.get('default')
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400
    df = add_derived_features(df)
    uploaded_data['default'] = df
    numeric_cols     = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    null_rates       = {c: round(float(df[c].isna().mean()), 3) for c in df.columns}
    city_col         = next((c for c in ['city','City','municipality'] if c in df.columns), None)
    uploaded_data['city_col'] = city_col or 'city'
    city_to_districts = {}
    if city_col and 'district' in df.columns:
        for city, grp in df.groupby(city_col):
            city_to_districts[str(city)] = sorted(grp['district'].dropna().unique().tolist())
    cat_filter_cols = {}
    for col in categorical_cols:
        skip = ['real_mls','mls','street','address','summary_url','virtual_tour',
                '_source','_file_type','listing_id']
        if col in skip: continue
        vals = df[col].dropna().astype(str).unique().tolist()
        if 2 <= len(vals) <= 50:
            cat_filter_cols[col] = sorted(vals)
    preview = df.head(5).replace({float('nan'): None}).to_dict(orient='records')
    cities = df[city_col].dropna().unique().tolist() if city_col and city_col in df.columns else []
    cat_n_unique = {c: int(df[c].nunique()) for c in categorical_cols}
    return jsonify({
        'rows':              int(len(df)),
        'columns':           int(df.shape[1]),
        'all_columns':       df.columns.tolist(),
        'numeric_columns':   numeric_cols,
        'categorical_columns': categorical_cols,
        'null_rates':        null_rates,
        'city_to_districts': city_to_districts,
        'cat_filter_cols':   cat_filter_cols,
        'cat_n_unique':      cat_n_unique,
        'city_col':          city_col,
        'cities':            cities,
        'preview':           preview,
    })

# Auto-load data folders on first request
_autoloaded = False

@app.before_request
def autoload_once():
    global _autoloaded
    if not _autoloaded:
        _autoloaded = True
        try:
            _autoload_data_folders()
        except Exception as _e:
            print(f'[autoload] failed: {_e}')

if __name__ == '__main__':
    app.run(debug=True, port=5000)