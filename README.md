# Zimbabwe Aircraft Movements – Ensemble Forecast Dashboard

This package contains a Streamlit app that trains **Random Forest**, **XGBoost**, and **Prophet** (if available) using engineered features and weather covariates, then produces a **10‑year monthly forecast** with an **interactive ensemble**.

## Contents
- `app.py` – Streamlit dashboard
- `Monthly_Expanded_Weather.csv` – dataset used by the app
- `requirements.txt` – Python dependencies
- `runtime.txt` – Pin Python 3.10 for Prophet compatibility
- `Improved_RF_Weather_Model.ipynb` – notebook (from analysis)
- `Ensemble_Model_Workbook.ipynb` – ensemble example notebook

## Quickstart (Streamlit Cloud)
1. Push these files to a GitHub repo
2. Go to **https://streamlit.io/cloud** → **Deploy app**
3. Select the repo and branch, app file = `app.py`
4. The app installs requirements and starts at a public URL

> **Note**: `runtime.txt` pins Python **3.10** to improve `prophet` install success. If you remove it, Streamlit may use a newer Python (e.g., 3.13) where Prophet wheels may be unavailable.

## Quickstart (Azure Web App)
1. Create a Web App with Python 3.10
2. Deploy these files via GitHub Actions or zip deploy
3. Set startup command:
   ```bash
   streamlit run app.py --server.port 8000 --server.address 0.0.0.0
   ```

## How forecasts are generated
- **Features**: `TimeIndex`, `MonthVal`, `YearVal`, lags (`Lag1`, `Lag12`), rolling means (`Rolling3`, `Rolling12`), plus weather (`Temp_C`, `Precip_mm`).
- **RF & XGB**: trained on tabular features, forecast via recursive month‑ahead loop.
- **Prophet**: trained on `y=Total`, regressors `Temp_C`, `Precip_mm`; predicts monthly `yhat`.
- **Ensemble**: weighted average of available model forecasts; weights are set in the sidebar and automatically re‑normalized.

## Reproducibility
- A sidebar seed controls stochastic elements (e.g., synthetic future weather sampling within climatological constraints).

## Troubleshooting
- If deployment logs show Prophet build issues, keep Python at 3.10 (via `runtime.txt`). If Prophet still fails, the app will run with RF + XGB and re‑normalize weights.
