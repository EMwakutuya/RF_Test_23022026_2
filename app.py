import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Optional models
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

st.set_page_config(page_title="Zimbabwe Aircraft Movements – Ensemble Forecast", layout="wide")
st.title("✈️ Zimbabwe Aircraft Movements – Ensemble Forecast Dashboard")
st.caption("Improved Random Forest with engineered features + optional XGBoost & Prophet ensemble. Uses weather features.")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    years = st.slider("Forecast horizon (years)", 1, 15, 10)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    st.markdown("### Ensemble Weights")
    w_rf = st.slider("Random Forest", 0.0, 1.0, 0.34, 0.01)
    w_xgb = st.slider("XGBoost", 0.0, 1.0, 0.33, 0.01)
    w_prophet = st.slider("Prophet", 0.0, 1.0, 0.33, 0.01)

# Utility: normalize weights for available models
def normalize_weights(w):
    total = sum([v for v in w.values() if v is not None])
    if total == 0:
        # fallback equal among available
        available = [k for k,v in w.items() if v is not None]
        return {k: 1/len(available) for k in available}
    return {k: (v/total if v is not None else None) for k,v in w.items()}

np.random.seed(seed)

# Load data
DATA_FILE = "Monthly_Expanded_Weather.csv"
df = pd.read_csv(DATA_FILE)
df['Date'] = pd.to_datetime(df[['Year','Month']].assign(DAY=1))
df = df.sort_values('Date')

# Feature engineering

df['MonthVal'] = df['Date'].dt.month
(df:=df.assign(
    YearVal=df['Date'].dt.year,
    TimeIndex=np.arange(len(df)),
    Lag1=df['Total'].shift(1),
    Lag12=df['Total'].shift(12),
    Rolling3=df['Total'].rolling(3).mean(),
    Rolling12=df['Total'].rolling(12).mean()
))

# Drop early NAs from lags/rolls
df = df.dropna().reset_index(drop=True)

features = [
    'TimeIndex','MonthVal','YearVal','Lag1','Lag12','Rolling3','Rolling12','Temp_C','Precip_mm'
]
X = df[features]
y = df['Total']

# Train / test split to display MAE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# =============== Train models ===============
rf = RandomForestRegressor(n_estimators=600, random_state=seed, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)

xgb = None
xgb_mae = None
if HAS_XGB:
    xgb = XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)

prophet_model = None
prophet_mae = None
# For Prophet we train on entire training window with regressors Temp_C & Precip_mm.
# Prepare Prophet-compatible frame
if HAS_PROPHET:
    p_df = df[['Date','Total','Temp_C','Precip_mm']].rename(columns={'Date':'ds','Total':'y'})
    # Ensure monotonic time
    p_train = p_df.iloc[:len(X_train)]
    p_test = p_df.iloc[len(X_train):]
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    prophet_model.add_regressor('Temp_C')
    prophet_model.add_regressor('Precip_mm')
    prophet_model.fit(p_train)
    p_future = p_test[['ds','Temp_C','Precip_mm']]
    p_fore = prophet_model.predict(p_future)
    prophet_pred = p_fore['yhat'].values
    prophet_mae = mean_absolute_error(y_test.values, prophet_pred)

# =============== Future feature generator ===============

def gen_future_weather(month):
    # Southern Africa climate logic used in data creation
    if month in [5,6,7]:
        temp = np.random.uniform(6,12)
        precip = 0.0
    elif month == 10:
        temp = np.random.uniform(30,36)
        precip = np.random.uniform(5,40)
    elif month in [11,12,1]:
        if month == 11:
            temp = np.random.uniform(18,26)
            precip = np.random.uniform(20,60)
        elif month == 12:
            temp = np.random.uniform(18,26)
            precip = np.random.uniform(60,120)
        else:  # Jan peak
            temp = np.random.uniform(16,24)
            precip = np.random.uniform(120,180)
    else:
        temp = np.random.uniform(18,26)
        precip = np.random.uniform(5,40)
    return temp, precip

# Recursive feature creation for tree models

def forecast_tree_models(horizon_months=120):
    rows = []
    last_row = df.iloc[-1]
    last_total = last_row['Total']
    last_month = int(last_row['MonthVal'])
    last_year = int(last_row['YearVal'])
    last_time = int(last_row['TimeIndex'])

    # Keep a rolling buffer to compute Lag12/rolling features from predictions
    totals_series = list(df['Total'].values[-12:])  # last 12 months

    for i in range(1, horizon_months+1):
        month = ((last_month + i - 1) % 12) + 1
        year = last_year + ((last_month + i - 1) // 12)
        timeindex = last_time + i

        # Weather
        t, p = gen_future_weather(month)

        lag1 = totals_series[-1]
        lag12 = totals_series[-12]
        rolling3 = np.mean(totals_series[-3:]) if len(totals_series) >= 3 else np.mean(totals_series)
        rolling12 = np.mean(totals_series)

        row = {
            'TimeIndex': timeindex,
            'MonthVal': month,
            'YearVal': year,
            'Lag1': lag1,
            'Lag12': lag12,
            'Rolling3': rolling3,
            'Rolling12': rolling12,
            'Temp_C': t,
            'Precip_mm': p
        }
        rows.append(row)

        # Predict step with available models to update totals_series for next step
        Xrow = pd.DataFrame([row])[features]
        rf_yhat = rf.predict(Xrow)[0]
        # Choose RF for recursion baseline; it's stable
        totals_series.append(rf_yhat)
        if len(totals_series) > 12:
            totals_series = totals_series[-12:]

    future_df = pd.DataFrame(rows)

    # Get predictions from each model
    rf_fore = rf.predict(future_df[features])
    xgb_fore = None
    if HAS_XGB:
        xgb_fore = xgb.predict(future_df[features])

    prophet_fore = None
    if HAS_PROPHET:
        # Build future frame for Prophet with known future weather
        start = df['Date'].iloc[-1]
        dates = pd.date_range(start, periods=horizon_months+1, freq='M')[1:]
        p_future = pd.DataFrame({'ds': dates, 'Temp_C': future_df['Temp_C'], 'Precip_mm': future_df['Precip_mm']})
        p_out = prophet_model.predict(p_future)
        prophet_fore = p_out['yhat'].values

    return future_df, rf_fore, xgb_fore, prophet_fore

horizon = years * 12
future_df, rf_fore, xgb_fore, prophet_fore = forecast_tree_models(horizon)

# Normalize weights for available models
weights = {
    'rf': w_rf,
    'xgb': (w_xgb if HAS_XGB else None),
    'prophet': (w_prophet if HAS_PROPHET else None)
}
weights = normalize_weights(weights)

# Compose ensemble
preds = []
labels = []
if 'rf' in weights: preds.append(weights['rf'] * rf_fore); labels.append('RF')
if HAS_XGB and 'xgb' in weights and xgb_fore is not None: preds.append(weights['xgb'] * xgb_fore); labels.append('XGB')
if HAS_PROPHET and 'prophet' in weights and prophet_fore is not None: preds.append(weights['prophet'] * prophet_fore); labels.append('Prophet')
ensemble_fore = np.sum(preds, axis=0) if preds else rf_fore

# Dates for plotting
future_dates = pd.date_range(df['Date'].iloc[-1], periods=horizon+1, freq='M')[1:]

# ================= Layout =================
col1, col2 = st.columns([2,1], gap='large')
with col1:
    st.subheader("Historical vs Forecast (Ensemble)")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df['Date'], df['Total'], label='Historical', color='#1f77b4')
    ax.plot(future_dates, ensemble_fore, label='Ensemble Forecast', color='#d62728')
    ax.plot(future_dates, rf_fore, label='RF', color='#2ca02c', alpha=0.4)
    if HAS_XGB and xgb_fore is not None:
        ax.plot(future_dates, xgb_fore, label='XGB', color='#9467bd', alpha=0.4)
    if HAS_PROPHET and prophet_fore is not None:
        ax.plot(future_dates, prophet_fore, label='Prophet', color='#ff7f0e', alpha=0.4)
    ax.set_xlabel('Date'); ax.set_ylabel('Movements'); ax.set_title('10-Year Forecast (Monthly)')
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Holdout Performance (MAE)")
    st.write(f"**Random Forest**: {rf_mae:,.0f}")
    if HAS_XGB and xgb_mae is not None:
        st.write(f"**XGBoost**: {xgb_mae:,.0f}")
    if HAS_PROPHET and prophet_mae is not None:
        st.write(f"**Prophet**: {prophet_mae:,.0f}")
    st.info("Models are trained at app startup for reproducibility and environment compatibility.")

st.subheader("Feature Importances")
fi_cols = st.columns(2)
with fi_cols[0]:
    st.markdown("**Random Forest**")
    rf_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
    fig2, ax2 = plt.subplots(figsize=(6,5))
    rf_imp.plot(kind='barh', ax=ax2, color='#2ca02c')
    ax2.set_title('RF Feature Importance'); ax2.set_xlabel('Importance')
    st.pyplot(fig2)

with fi_cols[1]:
    if HAS_XGB:
        st.markdown("**XGBoost**")
        try:
            xgb_imp = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=True)
            fig3, ax3 = plt.subplots(figsize=(6,5))
            xgb_imp.plot(kind='barh', ax=ax3, color='#9467bd')
            ax3.set_title('XGB Feature Importance'); ax3.set_xlabel('Importance')
            st.pyplot(fig3)
        except Exception as e:
            st.warning(f"XGBoost importance unavailable: {e}")
    else:
        st.info("XGBoost not available in this environment.")

# Output table and download
out_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast_Ensemble': ensemble_fore,
    'Forecast_RF': rf_fore,
    'Forecast_XGB': xgb_fore if HAS_XGB else np.nan,
    'Forecast_Prophet': prophet_fore if HAS_PROPHET else np.nan
})

st.subheader("Forecast Table")
st.dataframe(out_df)

st.download_button(
    label="Download Ensemble Forecast (CSV)",
    data=out_df.to_csv(index=False),
    file_name="ensemble_forecast_10y.csv",
    mime="text/csv"
)

st.caption("Tip: Adjust ensemble weights in the sidebar. Prophet/XGBoost weights are re-normalized if a library is unavailable.")
