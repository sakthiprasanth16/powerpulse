import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

st.set_page_config(page_title="Household Power Consumption Prediction", layout="wide")

RAW_FEATURES_CSV = r"E:\Sakthi\prasanth\projects\household\power\Scripts\raw_features.csv"
MODEL_PKL = r"E:\Sakthi\prasanth\projects\household\power\Scripts\trained_models\random_forest_model.pkl"
SCALER_PKL = r"E:\Sakthi\prasanth\projects\household\power\Scripts\trained_models\scaler.pkl"

FEATURES = [
    'Global_reactive_power', 'Voltage',
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
    'Global_active_power_daily_avg', 'Global_reactive_power_daily_avg', 'Voltage_daily_avg',
    'Hour', 'Is_peak_hour', 'Is_daytime'
]

NUMERIC_COLS_TO_SCALE = [
    'Global_reactive_power', 'Voltage',
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
    'Global_active_power_daily_avg', 'Global_reactive_power_daily_avg', 'Voltage_daily_avg'
]

SUBMETER_COLS = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# load model and csv
@st.cache_resource
def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_resource
def load_pickle(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

raw_df = load_csv(RAW_FEATURES_CSV)
scaler = load_pickle(SCALER_PKL)
model = load_pickle(MODEL_PKL)

if raw_df is None:
    st.error(f"raw_features.csv not found at: {RAW_FEATURES_CSV}")
    st.stop()

if model is None:
    st.warning("Model not found or failed to load. Prediction will be disabled.")
if scaler is None:
    st.warning("Scaler not found or failed to load. Prediction will be disabled.")

# Session defaults & pools
if 'suggestion_pools' not in st.session_state:
    st.session_state.suggestion_pools = {}

# Build suggestion
def build_pool_for_feature(feat):
    if feat in raw_df.columns:
        vals = raw_df[feat].dropna().unique().tolist()
        if len(vals) == 0:
            return [0.0]
        return vals
    else:
        if feat == 'Hour':
            return list(range(0, 24))
        elif feat in SUBMETER_COLS:
            return [0.0, 1.0, 2.0, 5.0, 10.0]
        else:
            return [0.0, 1.0, 2.0, 3.0, 4.0]

for feat in FEATURES:
    st.session_state.suggestion_pools[feat] = build_pool_for_feature(feat)

# Pre-fill sample input
def generate_custom_prefill():
    for feat, pool in st.session_state.suggestion_pools.items():
        try:
            val = np.random.choice(pool)
        except Exception:
            val = 0 if feat == 'Hour' else 0.0
        if feat == 'Hour':
            st.session_state[f"cust_{feat}"] = int(float(val))
            st.session_state[f"cust_txt_{feat}"] = str(int(float(val)))
        else:
            st.session_state[f"cust_txt_{feat}"] = f"{float(val):.6f}"
            st.session_state[f"cust_{feat}"] = float(val)


# UI 
st.title("Household Power Consumption Prediction")

if st.button("Generate Random values"):
    generate_custom_prefill()
    try:
        st.experimental_rerun()
    except Exception:
        pass

cols = st.columns(2)
editable_values = {}
i = 0
for feat in FEATURES:
    if feat in ['Is_peak_hour', 'Is_daytime']:
        continue
    colw = cols[i % 2]
    i += 1
    if feat == 'Hour':
        default_val = st.session_state.get(f"cust_{feat}", 9)
        val = colw.number_input("Hour (0-23)", min_value=0, max_value=23, value=int(default_val), step=1, format="%d", key=f"cust_{feat}")
        editable_values['Hour'] = int(val)
    else:
        suggested = st.session_state.suggestion_pools.get(feat, [])
        placeholder = ""
        if len(suggested) > 0:
            try:
                placeholder = f" (e.g. {float(suggested[0]):.3f})"
            except Exception:
                placeholder = f" (e.g. {suggested[0]})"
        default_txt = st.session_state.get(f"cust_txt_{feat}", "")
        txt = colw.text_input(f"{feat}{placeholder}", value=default_txt, key=f"cust_txt_{feat}")
        if txt.strip() == "":
            editable_values[feat] = None
        else:
            try:
                editable_values[feat] = float(txt)
            except Exception:
                colw.error("Invalid numeric value")
                editable_values[feat] = None

# auto flags
h = int(editable_values.get('Hour', 0) if editable_values.get('Hour', 0) is not None else 0)
editable_values['Is_daytime'] = 1 if (6 <= h < 18) else 0
editable_values['Is_peak_hour'] = 1 if (17 <= h <= 20) else 0

# Show all input columns in the preview
st.markdown("### Custom input preview (all features + flags)")
preview = {k: v for k, v in editable_values.items()}
preview_df = pd.DataFrame([preview])
cols_to_show = [c for c in FEATURES if c in preview_df.columns]
st.dataframe(preview_df[cols_to_show], use_container_width=True)

st.markdown("---")
predict_btn = st.button("Predict Global Active Power")


# Prediction logic
if predict_btn:
    # validate custom inputs
    missing = [feat for feat in FEATURES if feat not in editable_values or (editable_values[feat] is None and feat not in ['Is_peak_hour','Is_daytime'])]
    if len(missing) > 0:
        st.error(f"Please fill values for: {missing}")
        st.stop()
    row = editable_values.copy()

    # ensure model & scaler present
    if model is None:
        st.error("Model not loaded. Fix MODEL_PKL path.")
        st.stop()
    if scaler is None:
        st.error("Scaler not loaded. Fix SCALER_PKL path.")
        st.stop()

    # Build DataFrame row and ensure all FEATURES present
    row_df = pd.DataFrame([row], index=["user"])
    for c in FEATURES:
        if c not in row_df.columns:
            if c == 'Is_daytime':
                h = int(row_df['Hour'].iloc[0])
                row_df[c] = 1 if (6 <= h < 18) else 0
            elif c == 'Is_peak_hour':
                h = int(row_df['Hour'].iloc[0])
                row_df[c] = 1 if (17 <= h <= 20) else 0
            else:
                row_df[c] = 0.0

    # Ensure numeric conversion
    try:
        row_df = row_df.astype(float)
    except Exception:
        st.error("Some inputs could not be converted to float — check your values.")
        st.stop()

    # Save raw copy (hide flags in preview)
    raw_to_show = row_df[FEATURES].copy()

    # Apply log1p to submeter columns
    log_df = raw_to_show.copy()
    for c in SUBMETER_COLS:
        log_df[c] = np.log1p(log_df[c].astype(float))

    # Scale numeric columns
    try:
        scaled_vals = scaler.transform(log_df[NUMERIC_COLS_TO_SCALE].values)
    except Exception as e:
        st.error(f"Scaler.transform failed: {e}")
        st.stop()

    scaled_df = log_df.copy()
    scaled_df.loc[:, NUMERIC_COLS_TO_SCALE] = scaled_vals

    X_for_model = scaled_df[FEATURES].values
    try:
        pred = model.predict(X_for_model)[0]
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    st.success(f"Predicted Global_active_power: **{pred:.6f}** (model units)")