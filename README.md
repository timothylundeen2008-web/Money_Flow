# 📊 Institutional Rotation Dashboard

A daily-updating Streamlit app that tracks institutional money flow across S&P 500 sectors using free Yahoo Finance price data (stooq fallback) and Relative Rotation Graph (RRG) mathematics.

## What it shows

| Panel | What it detects |
|---|---|
| **Relative Rotation Graph** | Where each sector sits in the rotation cycle (Leading / Weakening / Lagging / Improving) |
| **Performance Heatmap** | All 11 sectors × 7 timeframes (1D, 1W, 1M, 3M, 6M, 1Y, YTD) |
| **Institutional Signals** | Ranked list of sectors by accumulation/distribution signal strength |
| **Spread Bars** | 1M performance vs 6M run-rate — the clearest proxy for institutional accumulation |

## How the math works

### RS-Ratio (where money IS)
```
RS-Ratio = (1 + sector_perf) / (1 + benchmark_perf) × 100
```
- Above 100 → sector is outperforming the equal-weighted average
- Below 100 → underperforming

### RS-Momentum (where money is GOING)
```
RS-Momentum = RS_short_term / RS_medium_term × 100
```
- Above 100 → relative strength is accelerating
- Below 100 → decelerating

### Four Quadrants
```
RS-Ratio ≥ 100 + RS-Mom ≥ 100  → LEADING    (institutions IN, riding momentum)
RS-Ratio ≥ 100 + RS-Mom < 100  → WEAKENING  (institutions distributing)
RS-Ratio < 100 + RS-Mom < 100  → LAGGING    (avoid)
RS-Ratio < 100 + RS-Mom ≥ 100  → IMPROVING  (early accumulation signal ⭐)
```

### Accumulation/Distribution Spread
```
Spread = 1M_performance − (3M_performance ÷ 3)
```
Changed in rotation_math v3 (was 1M − 6M÷6): the 3-month run-rate is more
responsive to 1–3-month institutional rotation cycles; the 6M baseline
carried stale history that masked current accumulation/distribution.
Large positive spread = this month running HOT above the 3-month run rate = accumulation.
Large negative spread = distribution. Actionable threshold in the app: ±0.8%.

## Data source

**Yahoo Finance (yfinance) primary, stooq.com CSV fallback** — all sector
performance timeframes are computed from actual ETF price history
(`data_fetcher.py`). A hard-coded demo dataset renders only when both
sources fail, and is labeled as such in the UI.

An earlier version of this README described Finviz scraping — that path
was removed (Finviz returns HTTP 403 to cloud IPs, and computing from
price history is more accurate; Finviz rounds to 1dp). If this README and
`data_fetcher.py` ever disagree again, trust the code and fix the README.

Data refreshes every 60 minutes via `@st.cache_data(ttl=3600)`.

## Setup

### 1. Clone / download

```bash
git clone https://github.com/YOUR_USERNAME/rotation-dashboard.git
cd rotation-dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

## Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

The app will auto-deploy and get a public URL like:
`https://your-app-name.streamlit.app`

### Auto-refresh on Streamlit Cloud
The app caches data for 60 minutes via `@st.cache_data(ttl=3600)`.
Each visit that's more than 60 minutes after the last fetch triggers a fresh pull from Yahoo Finance.
You can also force-refresh with the **🔄 Refresh** button.

## Project structure

```
rotation-dashboard/
├── app.py              # Main Streamlit UI
├── data_fetcher.py     # yfinance + stooq fetchers, demo-data fallback
├── rotation_math.py    # RS-Ratio, RS-Momentum, spread, quadrant math
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Theme + server config
└── README.md
```

## Data-source fair use note

The app makes one batched yfinance download per hour (per cache expiry);
the stooq fallback fetches one CSV per ticker with a 0.2s delay. Do not
shorten the cache TTL below 15 minutes.

## Disclaimer

This dashboard is for informational and educational purposes only.
It does not constitute financial advice. Past sector rotation patterns
do not guarantee future results.