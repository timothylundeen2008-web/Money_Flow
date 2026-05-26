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
from top_movers import fetch_top_movers, fetch_sector_flow_data, SIGNAL_COLORS, TIER_THRESHOLDS, TIER_LABELS
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
  [data-testid="stAppViewContainer"] { background: #0e1117; }
  [data-testid="stHeader"] { background: transparent; }
  .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

  /* Metric cards */
  .metric-card {
    background: #1a1d24;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0;
  }
  .metric-label { font-size: 12px; color: #a0a0a0; margin-bottom: 4px; font-weight: 500; }
  .metric-value { font-size: 22px; font-weight: 500; }
  .metric-sub   { font-size: 11px; color: #6b7280; margin-top: 2px; }

  /* Section labels */
  .section-label {
    font-size: 11px; font-weight: 500; color: #6b7280;
    text-transform: uppercase; letter-spacing: .06em;
    margin-bottom: 8px; margin-top: 4px;
  }

  /* Signal rows */
  .signal-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 8px 0; border-bottom: 0.5px solid rgba(255,255,255,0.08);
  }
  .signal-row:last-child { border-bottom: none; }
  .sig-badge {
    font-size: 10px; font-weight: 500; padding: 2px 8px;
    border-radius: 6px; white-space: nowrap; flex-shrink: 0; margin-top: 2px;
  }
  .sig-sector { font-size: 13px; font-weight: 500; color: #f0f0f0; }
  .sig-detail { font-size: 11px; color: #9ca3af; margin-top: 1px; }

  /* Cards */
  .dash-card {
    background: #1a1d24; border: 0.5px solid rgba(255,255,255,0.1);
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

with st.spinner("Fetching latest ETF price data from Yahoo Finance…"):
    df = load_data()

if df is None or df.empty:
    st.error('⚠️ Could not fetch data. Check your connection and try again.')
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
    st.caption(f"Data via yfinance · Last updated {last_update.strftime('%b %d, %Y %H:%M')} · Refreshes every 60 min")
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
    tf_sel = st.radio("Timeframe", list(tf_options.keys()), horizontal=True, key="rrg_tf", label_visibility="collapsed", index=2)

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
            textfont=dict(size=10, color="#e5e7eb"),
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
        paper_bgcolor="rgba(14,17,23,0)",
        plot_bgcolor="rgba(14,17,23,0)",
        xaxis=dict(range=[92, 108], title="RS-Ratio (relative strength)", gridcolor="rgba(255,255,255,0.08)", zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(range=[92, 108], title="RS-Momentum (acceleration)",   gridcolor="rgba(255,255,255,0.08)", zeroline=False, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_rrg, use_container_width=True, config={"displayModeBar": False})

    # RRG legend
    st.markdown("""
    <div style="display:flex;gap:16px;font-size:12px;color:#9ca3af;margin-top:-12px;">
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
            <div class="sig-sector">{row['sector']} <span style="font-size:11px;font-weight:400;color:#6b7280">{row['ticker']}</span></div>
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
fill_colors = [["#1e2330"] * len(hm_df)]
font_colors = [["#f0f0f0"] * len(hm_df)]

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
        fill_color="#1e2330",
        font=dict(color="#f0f0f0", size=12),
        align=["left"] + ["center"] * len(timeframe_cols),
        height=32,
        line_color="rgba(255,255,255,0.08)",
    ),
    cells=dict(
        values=cell_vals,
        fill_color=fill_colors,
        font=dict(color=font_colors, size=12),
        align=["left"] + ["center"] * len(timeframe_cols),
        height=30,
        line_color="rgba(255,255,255,0.05)",
    ),
))
fig_hm.update_layout(
    height=len(hm_df) * 32 + 80,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="rgba(14,17,23,0)",
)
st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})

# Color legend
st.markdown("""
<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:11px;color:#9ca3af;margin-top:-8px;margin-bottom:16px">
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
    color = "#1D9E75" if v > 2 else "#D85A30" if v < -2 else "#6b7280"
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
    paper_bgcolor="rgba(14,17,23,0)",
    plot_bgcolor="rgba(14,17,23,0)",
    xaxis=dict(
        title="Spread: 1M performance − (6M ÷ 6)",
        gridcolor="rgba(255,255,255,0.08)",
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



# ── Sector Rotation Flow Visualization ────────────────────────────────────────
st.markdown("---")
st.markdown("## 🌊 Sector Money Flow & Rotation")
st.caption("Aggregate institutional flow scores across all ETF categories · Where capital is moving right now")

with st.spinner("Loading sector flow data…"):
    flow_df = fetch_sector_flow_data()

if flow_df is not None and not flow_df.empty:

    # Aggregate by category
    agg = (flow_df.groupby("category")
           .agg(
               avg_flow   =("flow_score",  "mean"),
               avg_spread =("spread",      "mean"),
               avg_1m     =("perf_1m",     "mean"),
               avg_3m     =("perf_3m",     "mean"),
               vol_spikes =("vol_spike",   "sum"),
               etf_count  =("ticker",      "count"),
           )
           .reset_index()
           .sort_values("avg_flow", ascending=False))

    agg["inflow_pct"] = (agg["avg_flow"] - agg["avg_flow"].min()) /                         (agg["avg_flow"].max() - agg["avg_flow"].min() + 0.001) * 100

    # ── Chart 1: Horizontal Flow Bar (ranked inflow strength) ──────────────
    col_flow1, col_flow2 = st.columns([3, 2])

    with col_flow1:
        st.markdown('<div class="section-label">Inflow strength by category — ranked</div>', unsafe_allow_html=True)

        fig_flow = go.Figure()
        for _, row in agg.sort_values("avg_flow").iterrows():
            v = row["avg_flow"]
            color = ("#1D9E75" if v > 8 else "#2BAD7E" if v > 4 else
                     "#378ADD" if v > 0 else "#D85A30" if v > -4 else "#A32D2D")
            spike_marker = " 🔊" if row["vol_spikes"] > 0 else ""
            fig_flow.add_trace(go.Bar(
                x=[v],
                y=[f"{row['category']}{spike_marker}"],
                orientation="h",
                marker_color=color,
                marker_line_width=0,
                width=0.65,
                hovertemplate=(
                    f"<b>{row['category']}</b><br>"
                    f"Flow Score: {v:.1f}<br>"
                    f"Avg 1M: {fmt_pct(row['avg_1m'])}<br>"
                    f"Avg Spread: {fmt_pct(row['avg_spread'])}<br>"
                    f"Vol Spikes: {int(row['vol_spikes'])}/{int(row['etf_count'])} ETFs"
                    f"<extra></extra>"
                ),
                showlegend=False,
            ))
        fig_flow.add_vline(x=0, line_width=1, line_color="rgba(136,135,128,0.4)")
        fig_flow.update_layout(
            height=360,
            margin=dict(l=10, r=20, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Aggregate Flow Score", gridcolor="rgba(255,255,255,0.06)",
                       zeroline=False, tickfont=dict(size=10)),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
            bargap=0.25,
        )
        st.plotly_chart(fig_flow, use_container_width=True, config={"displayModeBar": False})

    with col_flow2:
        st.markdown('<div class="section-label">Flow summary</div>', unsafe_allow_html=True)

        top3    = agg.head(3)
        bottom3 = agg.tail(3)

        st.markdown("**💚 Top inflows**")
        for _, r in top3.iterrows():
            spikes = f" · 🔊 {int(r['vol_spikes'])} spike{'s' if r['vol_spikes']!=1 else ''}" if r["vol_spikes"] > 0 else ""
            st.markdown(f"""
            <div style="padding:6px 10px;background:rgba(29,158,117,0.1);border-left:3px solid #1D9E75;
                 border-radius:4px;margin-bottom:4px">
              <span style="font-weight:600;color:#f0f0f0">{r['category']}</span>
              <span style="color:#1D9E75;font-size:12px;margin-left:6px">Score {r['avg_flow']:.1f}</span>
              <span style="font-size:11px;color:#6b7280">{spikes}</span><br>
              <span style="font-size:11px;color:#9ca3af">1M: {fmt_pct(r['avg_1m'])} · Spread: {fmt_pct(r['avg_spread'])}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("**🔴 Top outflows**")
        for _, r in bottom3.iterrows():
            st.markdown(f"""
            <div style="padding:6px 10px;background:rgba(168,45,45,0.1);border-left:3px solid #A32D2D;
                 border-radius:4px;margin-bottom:4px">
              <span style="font-weight:600;color:#f0f0f0">{r['category']}</span>
              <span style="color:#D85A30;font-size:12px;margin-left:6px">Score {r['avg_flow']:.1f}</span><br>
              <span style="font-size:11px;color:#9ca3af">1M: {fmt_pct(r['avg_1m'])} · Spread: {fmt_pct(r['avg_spread'])}</span>
            </div>""", unsafe_allow_html=True)

    # ── Chart 2: Bubble chart — Spread (x) vs Vol Spikes (y) vs Flow (size) ──
    st.markdown('<div class="section-label" style="margin-top:12px">Rotation map — spread acceleration vs volume conviction</div>', unsafe_allow_html=True)
    st.caption("Bubble size = Flow Score · Right = accelerating above trend · Up = more vol spikes · Green = inflow · Red = outflow")

    fig_bubble = go.Figure()
    for _, row in agg.iterrows():
        v     = row["avg_flow"]
        color = ("#1D9E75" if v > 6 else "#2BAD7E" if v > 2 else
                 "#378ADD" if v > 0 else "#D85A30" if v > -4 else "#A32D2D")
        size  = max(20, min(60, abs(v) * 4 + 20))
        fig_bubble.add_trace(go.Scatter(
            x=[row["avg_spread"]],
            y=[row["vol_spikes"]],
            mode="markers+text",
            name=row["category"],
            text=[row["category"]],
            textposition="top center",
            textfont=dict(size=10, color="#e5e7eb"),
            marker=dict(
                size=size, color=color,
                line=dict(color="rgba(255,255,255,0.2)", width=1),
                opacity=0.85,
            ),
            hovertemplate=(
                f"<b>{row['category']}</b><br>"
                f"Avg Spread: {row['avg_spread']:.2f}%<br>"
                f"Vol Spikes: {int(row['vol_spikes'])}<br>"
                f"Flow Score: {v:.1f}<br>"
                f"Avg 1M: {fmt_pct(row['avg_1m'])}"
                f"<extra></extra>"
            ),
            showlegend=False,
        ))

    # Quadrant lines
    fig_bubble.add_vline(x=0, line_width=1, line_color="rgba(136,135,128,0.3)", line_dash="dot")
    fig_bubble.add_hline(y=0.5, line_width=1, line_color="rgba(136,135,128,0.3)", line_dash="dot")

    # Quadrant labels
    for label, x, y, color in [
        ("↗ HOT MONEY",   2.5, agg["vol_spikes"].max()*0.9, "#1D9E75"),
        ("↘ ACCUMULATE",  2.5, -0.3,                         "#378ADD"),
        ("↖ DISTRIBUTION",-2, agg["vol_spikes"].max()*0.9,  "#D85A30"),
        ("↙ OUTFLOW",     -2, -0.3,                          "#888780"),
    ]:
        fig_bubble.add_annotation(x=x, y=y, text=label, showarrow=False,
            font=dict(size=9, color=color), opacity=0.6)

    fig_bubble.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Avg Spread (1M vs 3M run-rate)", gridcolor="rgba(255,255,255,0.06)",
                   zeroline=False, ticksuffix="%", tickfont=dict(size=10)),
        yaxis=dict(title="# ETFs with Vol Spike", gridcolor="rgba(255,255,255,0.06)",
                   zeroline=False, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_bubble, use_container_width=True, config={"displayModeBar": False})

    # ── Threshold reference table ──────────────────────────────────────────
    with st.expander("📊 Volume spike threshold reference — data-driven by ADV tier"):
        st.markdown("""
        Thresholds are set by **Average Daily Dollar Volume (ADDV)** — the higher the liquidity,
        the larger the spike needed to confirm institutional conviction:

        | Tier | Liquidity | ADDV | Spike Threshold | Examples | Why |
        |---|---|---|---|---|---|
        | **1** | Mega Liquid | >$2B/day | **1.25×** | QQQ, XLK, XLF, XLV | At $2B+ ADV, institutions move $500M routinely. 1.25× = $500M extra in one session = directional conviction |
        | **2** | High Liquid | $200M–$2B | **1.50×** | XLE, XLI, GLD, TLT, IWM | Classic institutional threshold. Filters noise, catches real rotation flows |
        | **3** | Moderate | $50M–$200M | **2.00×** | SMH, IBB, ITA, KRE | Single large hedge fund trade = 1.5×. Need 2× for broad confirmation |
        | **4** | Lower Liquid | <$50M/day | **3.00×** | XBI, SKYY, HACK, DBA | Retail noise spikes these 1.5–2× routinely. 3× = real institutional entry |

        *Source: ValuEngine (Oct 2024), SeekingAlpha ADV data, Oxford Academic ETF Liquidity study,
        Morpheus Trading institutional volume research*
        """)

else:
    st.info("Sector flow data unavailable. Refresh to retry.", icon="📡")


# ── Where's the Money — Top 10 ETF Flow Panel ──────────────────────────────────
st.markdown("---")
st.markdown("## 💸 Where's the Money")
st.caption("Top 10 sub-sector ETFs ranked by Institutional Flow Score · Updated hourly")

with st.spinner("Loading ETF flow data…"):
    movers_df = fetch_top_movers(top_n=10)

if movers_df is not None and not movers_df.empty:

    # ── Filter controls ──────────────────────────────────────────────────────
    col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 1, 1])
    with col_f1:
        cats = ["All"] + sorted(movers_df["category"].unique().tolist())
        cat_filter = st.selectbox("Filter by category", cats, key="mover_cat", label_visibility="collapsed")
    with col_f2:
        sigs = ["All signals"] + list(SIGNAL_COLORS.keys())
        sig_filter = st.selectbox("Filter by signal", sigs, key="mover_sig", label_visibility="collapsed")
    with col_f3:
        top_n = st.selectbox("Show top", [10, 15, 20], key="mover_n", label_visibility="collapsed")

    with col_f4:
        vol_spike_only = st.toggle("🔊 Vol Spike Filter", value=False, key="vol_spike_filter",
                                   help="Show only ETFs with volume ≥1.5× their 20-day average — the institutional conviction threshold")

    # Re-fetch with updated top_n if changed
    if top_n != 10:
        movers_df = fetch_top_movers(top_n=top_n)

    display_df = movers_df.copy()
    if cat_filter != "All":
        display_df = display_df[display_df["category"] == cat_filter]
    if sig_filter != "All signals":
        display_df = display_df[display_df["signal"] == sig_filter]
    if vol_spike_only:
        display_df = display_df[display_df["vol_spike"] == True]

    # Banner when vol spike filter is active
    if vol_spike_only:
        spike_count = len(display_df)
        if spike_count > 0:
            st.markdown(f"""
            <div style="background:rgba(29,158,117,0.12);border:1px solid rgba(29,158,117,0.4);
                 border-radius:8px;padding:8px 14px;margin-bottom:10px;display:flex;align-items:center;gap:10px">
              <span style="font-size:18px">🔊</span>
              <div>
                <span style="font-weight:600;color:#1D9E75">{spike_count} ETF{"s" if spike_count!=1 else ""} with volume spike ≥1.5×</span>
                <span style="font-size:12px;color:#6b7280;margin-left:8px">— institutional conviction signal active</span>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(136,135,128,0.1);border:1px solid rgba(136,135,128,0.3);
                 border-radius:8px;padding:8px 14px;margin-bottom:10px">
              <span style="color:#888780">⚪ No ETFs currently showing volume ≥1.5× — no confirmed institutional spike today</span>
            </div>""", unsafe_allow_html=True)
    if vol_spike_only:
        display_df = display_df[display_df["vol_spike"] == True]

    # Banner when vol spike filter is active
    if vol_spike_only:
        spike_count = len(display_df)
        if spike_count > 0:
            st.markdown(f"""
            <div style="background:rgba(29,158,117,0.12);border:1px solid rgba(29,158,117,0.4);
                 border-radius:8px;padding:8px 14px;margin-bottom:10px;display:flex;align-items:center;gap:10px">
              <span style="font-size:18px">🔊</span>
              <div>
                <span style="font-weight:600;color:#1D9E75">{spike_count} ETF{"s" if spike_count!=1 else ""} showing volume spike ≥1.5×</span>
                <span style="font-size:12px;color:#6b7280;margin-left:8px">— institutional conviction signal active</span>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(136,135,128,0.1);border:1px solid rgba(136,135,128,0.3);
                 border-radius:8px;padding:8px 14px;margin-bottom:10px">
              <span style="color:#888780">⚪ No ETFs currently showing volume ≥1.5× — no confirmed institutional spike today</span>
            </div>""", unsafe_allow_html=True)

    # ── Rank cards ───────────────────────────────────────────────────────────
    for rank, (_, row) in enumerate(display_df.iterrows(), 1):
        sig_fg = row["signal_fg"]
        sig_bg = row["signal_bg"]

        # Spark bar: visual of 1D/1W/1M/3M as mini bar chart
        def spark_bar(v, max_abs=15):
            pct = min(abs(v) / max_abs * 100, 100)
            color = "#1D9E75" if v >= 0 else "#D85A30"
            return f'<div style="display:inline-flex;align-items:center;gap:4px;width:80px"><div style="height:6px;width:{pct:.0f}%;max-width:60px;background:{color};border-radius:3px"></div><span style="font-size:10px;color:{"#1D9E75" if v>=0 else "#D85A30"};font-weight:500">{fmt_pct(v)}</span></div>'

        vol_spike   = row["vol_ratio"] >= 1.5
        vol_color   = "#1D9E75" if row["vol_ratio"] >= 1.5 else "#2BAD7E" if row["vol_ratio"] > 1.2 else "#D85A30" if row["vol_ratio"] < 0.8 else "#888780"
        spread_color = "#1D9E75" if row["spread"] > 0 else "#D85A30"

        border_style = "1.5px solid rgba(29,158,117,0.6)" if vol_spike else "0.5px solid rgba(255,255,255,0.08)"
        glow_style   = "box-shadow:0 0 12px rgba(29,158,117,0.2);" if vol_spike else ""
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;
             background:var(--secondary-background-color);border-radius:10px;
             border:{border_style};{glow_style}margin-bottom:6px;">

          <!-- Rank -->
          <div style="font-size:18px;font-weight:700;color:#6b7280;width:24px;text-align:center;flex-shrink:0">
            {rank}
          </div>

          <!-- Ticker + Name -->
          <div style="flex:0 0 130px">
            <div style="font-size:15px;font-weight:600;color:var(--text-color)">{row["ticker"]}</div>
            <div style="font-size:11px;color:#6b7280;margin-top:1px">{row["name"]}</div>
            <div style="font-size:10px;color:#4b5563;margin-top:1px">{row["category"]}</div>
          </div>

          <!-- Signal badge + tier -->
          <div style="flex:0 0 155px">
            <span style="background:{sig_bg};color:{sig_fg};font-size:10px;font-weight:600;
                  padding:3px 8px;border-radius:6px">{row["signal"]}</span>
            <div style="margin-top:4px;font-size:9px;color:#6b7280">
              {row["tier_label"]} · spike≥{row["spike_label"]}
            </div>
          </div>

          <!-- Perf bars -->
          <div style="flex:1;display:grid;grid-template-columns:repeat(4,1fr);gap:4px">
            <div style="text-align:center">
              <div style="font-size:9px;color:#6b7280;margin-bottom:2px">1D</div>
              {spark_bar(row["perf_1d"])}
            </div>
            <div style="text-align:center">
              <div style="font-size:9px;color:#6b7280;margin-bottom:2px">1W</div>
              {spark_bar(row["perf_1w"])}
            </div>
            <div style="text-align:center">
              <div style="font-size:9px;color:#6b7280;margin-bottom:2px">1M</div>
              {spark_bar(row["perf_1m"])}
            </div>
            <div style="text-align:center">
              <div style="font-size:9px;color:#6b7280;margin-bottom:2px">3M</div>
              {spark_bar(row["perf_3m"])}
            </div>
          </div>

          <!-- Spread + Vol -->
          <div style="flex:0 0 110px;text-align:right">
            <div style="font-size:11px;color:#6b7280">Spread</div>
            <div style="font-size:13px;font-weight:600;color:{spread_color}">{fmt_pct(row["spread"])}</div>
            <div style="font-size:10px;color:#6b7280;margin-top:3px">Vol ×{row["vol_ratio"]:.1f}</div>
            <div style="font-size:{"12px" if vol_spike else "10px"};font-weight:{"700" if vol_spike else "400"};color:{vol_color}">
              {"🔊 SPIKE" if vol_spike else ("↑ Heavy" if row["vol_ratio"]>1.3 else "↓ Light" if row["vol_ratio"]<0.8 else "Normal")}
            </div>
          </div>

          <!-- Flow score -->
          <div style="flex:0 0 60px;text-align:center">
            <div style="font-size:9px;color:#6b7280;margin-bottom:2px">FLOW</div>
            <div style="font-size:17px;font-weight:700;color:{sig_fg}">{row["flow_score"]:.0f}</div>
          </div>

        </div>
        """, unsafe_allow_html=True)

    # ── Flow score methodology note ──────────────────────────────────────────
    with st.expander("📐 How the Flow Score is calculated"):
        st.markdown("""
        The **Institutional Flow Score** combines four signals weighted by reliability:

        | Weight | Component | What it measures |
        |---|---|---|
        | **40%** | Momentum consistency | Equal-weighted avg of 1W + 1M + 3M performance |
        | **30%** | Acceleration (spread) | 1M perf minus (3M ÷ 3) monthly run-rate — positive = accelerating above trend |
        | **20%** | Volume conviction | Recent 5-day avg volume vs 20-day baseline — above 1× = institutional size |
        | **10%** | Timeframe unity | Fraction of 1W / 1M / 3M timeframes that are all positive |

        **Signal labels** are assigned based on spread + volume together:
        - **Strong Accumulation** → spread > 1.5% *and* vol ratio > 1.3×
        - **Accumulation** → spread > 0.5% *and* vol ratio > 1.1×
        - **Inflow** → positive 1M *and* positive spread
        - **Distribution / Strong Distribution** → negative spread with declining volume
        """)

else:
    st.info("Top movers data unavailable. Check connection and refresh.", icon="📡")


st.markdown("---")
st.caption("Data sourced from yfinance (Yahoo Finance) via ETF price history. Not financial advice.")
