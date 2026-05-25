"""
Institutional Rotation Dashboard
Fetches live sector performance data from Finviz (free tier)
and renders a full rotation analysis dashboard in Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

from data_fetcher import fetch_sector_data, get_cache_age_minutes
from rotation_math import (
    compute_rs_ratio,
    compute_rs_momentum,
    compute_spread,
    classify_quadrant,
    rank_signals,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Institutional Rotation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling (mirrors the Claude widget aesthetic) ──────────────────────────────
st.markdown("""
<style>
  /* Global */
  [data-testid="stAppViewContainer"] { background: #fafaf9; }
  [data-testid="stHeader"] { background: transparent; }
  .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

  /* Metric cards */
  .metric-card {
    background: #f1efe8;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0;
  }
  .metric-label { font-size: 12px; color: #5f5e5a; margin-bottom: 4px; font-weight: 500; }
  .metric-value { font-size: 22px; font-weight: 500; }
  .metric-sub   { font-size: 11px; color: #888780; margin-top: 2px; }

  /* Section labels */
  .section-label {
    font-size: 11px; font-weight: 500; color: #888780;
    text-transform: uppercase; letter-spacing: .06em;
    margin-bottom: 8px; margin-top: 4px;
  }

  /* Signal rows */
  .signal-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 8px 0; border-bottom: 0.5px solid rgba(0,0,0,0.08);
  }
  .signal-row:last-child { border-bottom: none; }
  .sig-badge {
    font-size: 10px; font-weight: 500; padding: 2px 8px;
    border-radius: 6px; white-space: nowrap; flex-shrink: 0; margin-top: 2px;
  }
  .sig-sector { font-size: 13px; font-weight: 500; color: #2c2c2a; }
  .sig-detail { font-size: 11px; color: #5f5e5a; margin-top: 1px; }

  /* Cards */
  .dash-card {
    background: #fff; border: 0.5px solid rgba(0,0,0,0.1);
    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
  [data-testid="collapsedControl"] { display: none; }

  /* Plotly chart backgrounds */
  .js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Color helpers ──────────────────────────────────────────────────────────────
QUAD_COLORS = {
    "Leading":   {"bg": "#E1F5EE", "fg": "#0F6E56", "dot": "#1D9E75"},
    "Weakening": {"bg": "#FAEEDA", "fg": "#854F0B", "dot": "#BA7517"},
    "Lagging":   {"bg": "#FCEBEB", "fg": "#A32D2D", "dot": "#E24B4A"},
    "Improving": {"bg": "#E6F1FB", "fg": "#185FA5", "dot": "#378ADD"},
}

def perf_color(v: float) -> tuple[str, str]:
    """Returns (bg, fg) hex for a performance value."""
    if v >  8: return "#085041", "#9FE1CB"
    if v >  4: return "#1D9E75", "#04342C"
    if v >  1: return "#9FE1CB", "#04342C"
    if v > -1: return "#D3D1C7", "#444441"
    if v > -4: return "#F0997B", "#4A1B0C"
    if v > -8: return "#D85A30", "#ffffff"
    return "#4A1B0C", "#F5C4B3"

def fmt_pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    return fetch_sector_data()

with st.spinner("Fetching latest sector data from Finviz…"):
    df = load_data()

if df is None or df.empty:
    st.error("⚠️ Could not fetch data from Finviz. Check your connection and try again.")
    st.stop()

# ── Compute derived metrics ────────────────────────────────────────────────────
df = compute_rs_ratio(df)
df = compute_rs_momentum(df)
df = compute_spread(df)
df = classify_quadrant(df)
df = rank_signals(df)

# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("## 📊 Institutional Rotation Dashboard")
    cache_age = get_cache_age_minutes()
    last_update = datetime.now() - timedelta(minutes=cache_age)
    st.caption(f"Data via Finviz · Last updated {last_update.strftime('%b %d, %Y %H:%M')} · Refreshes every 60 min")
with col_refresh:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()

# ── Metric cards ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Market rotation overview</div>', unsafe_allow_html=True)

quad_counts = df["quadrant"].value_counts().to_dict()
leading_n   = quad_counts.get("Leading", 0)
improving_n = quad_counts.get("Improving", 0)
weakening_n = quad_counts.get("Weakening", 0)
lagging_n   = quad_counts.get("Lagging", 0)

top_mom    = df.loc[df["rs_momentum"].idxmax()]
top_accum  = df.loc[df["spread"].idxmax()]
top_dist   = df.loc[df["spread"].idxmin()]

metrics = [
    {"label": "Leading sectors",    "val": str(leading_n),        "sub": "Outperforming & accelerating",      "color": "#1D9E75"},
    {"label": "Improving sectors",  "val": str(improving_n),      "sub": "Underperforming but turning ↑",     "color": "#378ADD"},
    {"label": "Top RS momentum",    "val": top_mom["ticker"],     "sub": f"{top_mom['sector']} · mom {top_mom['rs_momentum']:.1f}", "color": "#BA7517"},
    {"label": "Weakening sectors",  "val": str(weakening_n),      "sub": "Distribution phase",                "color": "#D85A30"},
    {"label": "Lagging sectors",    "val": str(lagging_n),        "sub": "Underperforming & decelerating",    "color": "#A32D2D"},
    {"label": "Top accum signal",   "val": top_accum["ticker"],   "sub": f"Spread: {fmt_pct(top_accum['spread'])}", "color": "#534AB7"},
]

cols = st.columns(6)
for col, m in zip(cols, metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{m['label']}</div>
          <div class="metric-value" style="color:{m['color']}">{m['val']}</div>
          <div class="metric-sub">{m['sub']}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── RRG + Signals ──────────────────────────────────────────────────────────────
col_rrg, col_signals = st.columns([3, 2])

with col_rrg:
    st.markdown('<div class="section-label">Relative rotation graph · RS-Momentum vs RS-Ratio</div>', unsafe_allow_html=True)

    tf_options = {"1W": "rs_ratio_1w", "1M": "rs_ratio_1m", "3M": "rs_ratio_3m"}
    mom_options = {"1W": "rs_momentum_1w", "1M": "rs_momentum_1m", "3M": "rs_momentum_3m"}
    tf_sel = st.radio("Timeframe", list(tf_options.keys()), horizontal=True, key="rrg_tf", label_visibility="collapsed")

    ratio_col = tf_options[tf_sel]
    mom_col   = mom_options[tf_sel]

    fig_rrg = go.Figure()

    # Quadrant shading
    fig_rrg.add_shape(type="rect", x0=100, x1=108, y0=100, y1=108, fillcolor="rgba(29,158,117,0.07)", line_width=0)
    fig_rrg.add_shape(type="rect", x0=100, x1=108, y0=92,  y1=100, fillcolor="rgba(186,117,23,0.07)", line_width=0)
    fig_rrg.add_shape(type="rect", x0=92,  x1=100, y0=92,  y1=100, fillcolor="rgba(226,75,74,0.07)",  line_width=0)
    fig_rrg.add_shape(type="rect", x0=92,  x1=100, y0=100, y1=108, fillcolor="rgba(55,138,221,0.07)", line_width=0)

    # Axes lines at 100
    fig_rrg.add_shape(type="line", x0=100, x1=100, y0=92, y1=108, line=dict(color="rgba(136,135,128,0.4)", width=1.5, dash="dot"))
    fig_rrg.add_shape(type="line", x0=92, x1=108,  y0=100, y1=100, line=dict(color="rgba(136,135,128,0.4)", width=1.5, dash="dot"))

    # Quadrant labels
    for label, x, y, color in [
        ("LEADING",   106, 107, "#1D9E75"),
        ("WEAKENING", 106, 93,  "#BA7517"),
        ("LAGGING",   94,  93,  "#E24B4A"),
        ("IMPROVING", 94,  107, "#378ADD"),
    ]:
        fig_rrg.add_annotation(x=x, y=y, text=label, showarrow=False,
            font=dict(size=9, color=color, family="monospace"), opacity=0.7)

    # Sector dots
    for _, row in df.iterrows():
        q = row["quadrant"]
        color = QUAD_COLORS[q]["dot"]
        rx = row.get(ratio_col, row["rs_ratio"])
        ry = row.get(mom_col, row["rs_momentum"])
        fig_rrg.add_trace(go.Scatter(
            x=[rx], y=[ry],
            mode="markers+text",
            name=row["ticker"],
            text=[row["ticker"]],
            textposition="top right",
            textfont=dict(size=10, color="#2c2c2a"),
            marker=dict(size=14, color=color, line=dict(color="#fff", width=1.5)),
            hovertemplate=(
                f"<b>{row['sector']}</b><br>"
                f"RS-Ratio: {rx:.1f}<br>"
                f"RS-Momentum: {ry:.1f}<br>"
                f"Quadrant: {q}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig_rrg.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[92, 108], title="RS-Ratio (relative strength)", gridcolor="rgba(136,135,128,0.15)", zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(range=[92, 108], title="RS-Momentum (acceleration)",   gridcolor="rgba(136,135,128,0.15)", zeroline=False, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_rrg, use_container_width=True, config={"displayModeBar": False})

    # RRG legend
    st.markdown("""
    <div style="display:flex;gap:16px;font-size:12px;color:#5f5e5a;margin-top:-12px;">
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#1D9E75;margin-right:4px"></span>Leading</span>
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#BA7517;margin-right:4px"></span>Weakening</span>
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#E24B4A;margin-right:4px"></span>Lagging</span>
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#378ADD;margin-right:4px"></span>Improving</span>
    </div>""", unsafe_allow_html=True)

with col_signals:
    st.markdown('<div class="section-label">Institutional signals</div>', unsafe_allow_html=True)

    signals_html = ""
    for _, row in df.sort_values("signal_rank").head(8).iterrows():
        q     = row["quadrant"]
        bc    = QUAD_COLORS[q]
        spread_dir = "▲ Accumulation" if row["spread"] > 0 else "▼ Distribution"
        spread_c   = "#1D9E75" if row["spread"] > 0 else "#D85A30"
        signals_html += f"""
        <div class="signal-row">
          <span class="sig-badge" style="background:{bc['bg']};color:{bc['fg']}">{q}</span>
          <div>
            <div class="sig-sector">{row['sector']} <span style="font-size:11px;font-weight:400;color:#888780">{row['ticker']}</span></div>
            <div class="sig-detail" style="color:{spread_c}">{spread_dir} · spread {fmt_pct(row['spread'])}</div>
            <div class="sig-detail">RS-Ratio {row['rs_ratio']:.1f} · RS-Mom {row['rs_momentum']:.1f}</div>
          </div>
        </div>"""

    st.markdown(f'<div class="dash-card" style="min-height:420px">{signals_html}</div>', unsafe_allow_html=True)

# ── Performance Heatmap ────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Performance heatmap · all timeframes</div>', unsafe_allow_html=True)

timeframe_cols = {
    "1D": "perf_1d", "1W": "perf_1w", "1M": "perf_1m",
    "3M": "perf_3m", "6M": "perf_6m", "1Y": "perf_1y", "YTD": "perf_ytd",
}

sort_opt = st.selectbox(
    "Sort by", ["Sector"] + list(timeframe_cols.keys()),
    key="hm_sort", label_visibility="collapsed",
    index=0,
)

hm_df = df.copy()
if sort_opt == "Sector":
    hm_df = hm_df.sort_values("sector")
else:
    hm_df = hm_df.sort_values(timeframe_cols[sort_opt], ascending=False)

# Build Plotly heatmap table
header_vals = ["Sector"] + list(timeframe_cols.keys())
cell_vals   = [hm_df["sector"].tolist()]
fill_colors = [["#f1efe8"] * len(hm_df)]
font_colors = [["#2c2c2a"] * len(hm_df)]

for tf, col in timeframe_cols.items():
    col_vals, fills, fonts = [], [], []
    for v in hm_df[col]:
        bg, fg = perf_color(v)
        col_vals.append(fmt_pct(v))
        fills.append(bg)
        fonts.append(fg)
    cell_vals.append(col_vals)
    fill_colors.append(fills)
    font_colors.append(fonts)

fig_hm = go.Figure(go.Table(
    columnwidth=[180] + [80] * len(timeframe_cols),
    header=dict(
        values=[f"<b>{h}</b>" for h in header_vals],
        fill_color="#f1efe8",
        font=dict(color="#2c2c2a", size=12),
        align=["left"] + ["center"] * len(timeframe_cols),
        height=32,
        line_color="rgba(0,0,0,0.08)",
    ),
    cells=dict(
        values=cell_vals,
        fill_color=fill_colors,
        font=dict(color=font_colors, size=12),
        align=["left"] + ["center"] * len(timeframe_cols),
        height=30,
        line_color="rgba(0,0,0,0.05)",
    ),
))
fig_hm.update_layout(
    height=len(hm_df) * 32 + 80,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})

# Color legend
st.markdown("""
<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:11px;color:#5f5e5a;margin-top:-8px;margin-bottom:16px">
  <span style="font-weight:500">Scale:</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#085041;margin-right:3px"></span>&gt;+8%</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#1D9E75;margin-right:3px"></span>+4–8%</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#9FE1CB;margin-right:3px"></span>+1–4%</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#D3D1C7;margin-right:3px"></span>0±1%</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#F0997B;margin-right:3px"></span>-1–4%</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#D85A30;margin-right:3px"></span>-4–8%</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#4A1B0C;margin-right:3px"></span>&lt;-8%</span>
</div>""", unsafe_allow_html=True)

# ── Spread / Accumulation-Distribution bars ────────────────────────────────────
st.markdown('<div class="section-label">Short vs long-term spread · accumulation / distribution detector</div>', unsafe_allow_html=True)

spread_df   = df.sort_values("spread", ascending=False)
max_spread  = spread_df["spread"].abs().max() or 1

fig_spread = go.Figure()
for _, row in spread_df.iterrows():
    v = row["spread"]
    color = "#1D9E75" if v > 2 else "#D85A30" if v < -2 else "#888780"
    label = "Accumulation" if v > 2 else "Distribution" if v < -2 else "Neutral"
    fig_spread.add_trace(go.Bar(
        x=[v],
        y=[f"{row['sector']} ({row['ticker']})"],
        orientation="h",
        marker_color=color,
        width=0.6,
        hovertemplate=f"<b>{row['sector']}</b><br>Spread: {fmt_pct(v)}<br>Signal: {label}<br>1M: {fmt_pct(row['perf_1m'])} · 6M: {fmt_pct(row['perf_6m'])}<extra></extra>",
        showlegend=False,
    ))

fig_spread.add_vline(x=0, line_width=1.5, line_color="rgba(136,135,128,0.5)")
fig_spread.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=10, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        title="Spread: 1M performance − (6M ÷ 6)",
        gridcolor="rgba(136,135,128,0.15)",
        zeroline=False,
        ticksuffix="%",
        tickfont=dict(size=10),
    ),
    yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
    bargap=0.3,
)
st.plotly_chart(fig_spread, use_container_width=True, config={"displayModeBar": False})

st.caption("Spread = 1M performance − (6M performance ÷ 6). Large positive spread → short-term acceleration above long-term trend = institutional accumulation signal. Large negative = distribution.")

# ── Raw data expander ──────────────────────────────────────────────────────────
with st.expander("📋 Raw data table"):
    display_cols = ["sector", "ticker", "quadrant", "rs_ratio", "rs_momentum", "spread",
                    "perf_1d", "perf_1w", "perf_1m", "perf_3m", "perf_6m", "perf_1y", "perf_ytd"]
    st.dataframe(
        df[display_cols].rename(columns={
            "sector": "Sector", "ticker": "ETF", "quadrant": "Quadrant",
            "rs_ratio": "RS-Ratio", "rs_momentum": "RS-Momentum", "spread": "Spread",
            "perf_1d": "1D%", "perf_1w": "1W%", "perf_1m": "1M%",
            "perf_3m": "3M%", "perf_6m": "6M%", "perf_1y": "1Y%", "perf_ytd": "YTD%",
        }).style.format({
            "RS-Ratio": "{:.1f}", "RS-Momentum": "{:.1f}", "Spread": "{:+.1f}",
            "1D%": "{:+.1f}", "1W%": "{:+.1f}", "1M%": "{:+.1f}",
            "3M%": "{:+.1f}", "6M%": "{:+.1f}", "1Y%": "{:+.1f}", "YTD%": "{:+.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "⬇ Download CSV",
        df[display_cols].to_csv(index=False),
        file_name=f"rotation_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Data sourced from [Finviz](https://finviz.com/groups.ashx) (free tier) via HTML scraping. Not financial advice.")