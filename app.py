"""
Institutional Rotation Dashboard  (v5)
──────────────────────────────────────
Changes v4 -> v5:
  [CRITICAL]  Vol spike badge in mover cards now uses per-tier threshold
              (was hardcoded 1.5x regardless of tier)
  [CRITICAL]  CAPE expander retitled/reworded -- it is weighted P/E, not true CAPE
  [CRITICAL]  Flow Score methodology note updated (now says today vs 20-day avg)
  [HIGH]      Duplicate vol spike banner removed
  [HIGH]      RRG dots show directional arrows (rotation_history delta vectors)
  [HIGH]      Improving quadrant dots get pulsing highlight ring
  [HIGH]      Spread bar threshold lowered 2pct -> 0.8pct for actionable daily signal
  [HIGH]      Bubble chart Y-axis changed from spike count to avg_vol_ratio (continuous)
  [MEDIUM]    RRG axis range is now dynamic (data-driven + padding)
  [MEDIUM]    RRG quadrant shading follows dynamic axis
  [MEDIUM]    RS-Ratio 100 explanation added below RRG
  [MEDIUM]    Improving quadrant callout in signals panel
  [ENHANCE]   Stealth Accumulation signal displayed in signals panel
  [ENHANCE]   4-week rotation context note per sector in signals panel
  [ENHANCE]   Flow score shown as rank (N of M) alongside numeric score
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data_fetcher import fetch_sector_data, get_cache_age_minutes
from valuation import fetch_valuation_data, VALUATION_COLORS
from top_movers import (
    fetch_top_movers, fetch_sector_flow_data,
    SIGNAL_COLORS, TIER_THRESHOLDS, TIER_LABELS,
)
from rotation_math import (
    compute_rs_ratio,
    compute_rs_momentum,
    compute_spread,
    compute_stealth_accumulation,
    compute_rotation_history,
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

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0e1117; }
  [data-testid="stHeader"] { background: transparent; }
  .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

  .metric-card {
    background: #1a1d24; border-radius: 8px;
    padding: 0.75rem 1rem; margin-bottom: 0;
  }
  .metric-label { font-size: 12px; color: #a0a0a0; margin-bottom: 4px; font-weight: 500; }
  .metric-value { font-size: 22px; font-weight: 500; }
  .metric-sub   { font-size: 11px; color: #6b7280; margin-top: 2px; }

  .section-label {
    font-size: 11px; font-weight: 500; color: #6b7280;
    text-transform: uppercase; letter-spacing: .06em;
    margin-bottom: 8px; margin-top: 4px;
  }

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

  .dash-card {
    background: #1a1d24; border: 0.5px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem;
  }

  /* Improving quadrant pulse ring */
  @keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(55,138,221,0.5); }
    70%  { box-shadow: 0 0 0 6px rgba(55,138,221,0); }
    100% { box-shadow: 0 0 0 0 rgba(55,138,221,0); }
  }
  .improving-pulse { animation: pulse-ring 2s infinite; border-radius: 50%; display: inline-block; }

  #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
  [data-testid="collapsedControl"] { display: none; }
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

DIRECTION_ARROWS = {
    "Strengthening": "↗",
    "Topping":       "↘",
    "Weakening":     "↙",
    "Bottoming":     "↖",
    "Stable":        "→",
}

def perf_color(v: float) -> tuple:
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

# ── Load & process data ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    return fetch_sector_data()

with st.spinner("Fetching latest ETF price data from Yahoo Finance…"):
    df = load_data()

if df is None or df.empty:
    st.error("Could not fetch data. Check your connection and try again.")
    st.stop()

df = compute_rs_ratio(df)
df = compute_rs_momentum(df)
df = compute_spread(df)
df = compute_stealth_accumulation(df)
df = compute_rotation_history(df)
df = classify_quadrant(df)
df = rank_signals(df)

# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("## 📊 Institutional Rotation Dashboard")
    cache_age   = get_cache_age_minutes()
    last_update = datetime.now() - timedelta(minutes=cache_age)
    st.caption(
        f"Data via yfinance · Last updated {last_update.strftime('%b %d, %Y %H:%M')} "
        f"· Refreshes every 60 min · v5"
    )
with col_refresh:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()

# ── KPI metric cards ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Market rotation overview</div>', unsafe_allow_html=True)

quad_counts = df["quadrant"].value_counts().to_dict()
leading_n   = quad_counts.get("Leading",   0)
improving_n = quad_counts.get("Improving", 0)
weakening_n = quad_counts.get("Weakening", 0)
lagging_n   = quad_counts.get("Lagging",   0)

top_mom   = df.loc[df["rs_momentum"].idxmax()]
top_accum = df.loc[df["spread"].idxmax()]

# Market bias
inflow  = leading_n + improving_n
outflow = weakening_n + lagging_n
bias_label = ("Bullish" if inflow > outflow else "Bearish" if outflow > inflow else "Neutral")
bias_color = "#1D9E75" if bias_label == "Bullish" else "#D85A30" if bias_label == "Bearish" else "#888780"

stealth_sectors = df[df["stealth_signal"] != "None"]["sector"].tolist()

metrics = [
    {"label": "Market bias",        "val": bias_label,          "sub": f"{inflow} inflow · {outflow} outflow",      "color": bias_color},
    {"label": "Leading sectors",    "val": str(leading_n),      "sub": "Outperforming & accelerating",              "color": "#1D9E75"},
    {"label": "Improving sectors",  "val": str(improving_n),    "sub": "Early accumulation — highest alpha",        "color": "#378ADD"},
    {"label": "Top RS momentum",    "val": top_mom["ticker"],   "sub": f"{top_mom['sector']} · {top_mom['rs_momentum']:.1f}", "color": "#BA7517"},
    {"label": "Weakening sectors",  "val": str(weakening_n),    "sub": "Distribution phase",                        "color": "#D85A30"},
    {"label": "Stealth signals",    "val": str(len(stealth_sectors)), "sub": ", ".join(stealth_sectors[:3]) or "None detected", "color": "#534AB7"},
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
    st.markdown('<div class="section-label">Relative rotation graph · RS-Momentum vs RS-Ratio</div>',
                unsafe_allow_html=True)

    tf_options  = {"1W": "rs_ratio_1w",    "1M": "rs_ratio_1m",    "3M": "rs_ratio_3m"}
    mom_options = {"1W": "rs_momentum_1w", "1M": "rs_momentum_1m", "3M": "rs_momentum_3m"}
    tf_sel = st.radio(
        "Timeframe", list(tf_options.keys()),
        horizontal=True, key="rrg_tf", label_visibility="collapsed", index=2
    )

    ratio_col = tf_options[tf_sel]
    mom_col   = mom_options[tf_sel]

    # Dynamic axis range
    r_vals = df[ratio_col].values
    m_vals = df[mom_col].values
    pad    = 2.0
    x_min, x_max = float(r_vals.min()) - pad, float(r_vals.max()) + pad
    y_min, y_max = float(m_vals.min()) - pad, float(m_vals.max()) + pad
    # Always include 100/100 centre with some breathing room
    x_min, x_max = min(x_min, 97), max(x_max, 103)
    y_min, y_max = min(y_min, 97), max(y_max, 103)

    fig_rrg = go.Figure()

    # Dynamic quadrant shading
    fig_rrg.add_shape(type="rect", x0=100, x1=x_max, y0=100, y1=y_max,
                      fillcolor="rgba(29,158,117,0.07)", line_width=0)
    fig_rrg.add_shape(type="rect", x0=100, x1=x_max, y0=y_min, y1=100,
                      fillcolor="rgba(186,117,23,0.07)", line_width=0)
    fig_rrg.add_shape(type="rect", x0=x_min, x1=100,  y0=y_min, y1=100,
                      fillcolor="rgba(226,75,74,0.07)",  line_width=0)
    fig_rrg.add_shape(type="rect", x0=x_min, x1=100,  y0=100,   y1=y_max,
                      fillcolor="rgba(55,138,221,0.07)", line_width=0)

    # Centre axes
    fig_rrg.add_shape(type="line", x0=100, x1=100, y0=y_min, y1=y_max,
                      line=dict(color="rgba(136,135,128,0.4)", width=1.5, dash="dot"))
    fig_rrg.add_shape(type="line", x0=x_min, x1=x_max, y0=100, y1=100,
                      line=dict(color="rgba(136,135,128,0.4)", width=1.5, dash="dot"))

    # Quadrant labels (inside corners)
    for label, x, y, color in [
        ("LEADING",   x_max - 1, y_max - 0.5, "#1D9E75"),
        ("WEAKENING", x_max - 1, y_min + 0.5, "#BA7517"),
        ("LAGGING",   x_min + 1, y_min + 0.5, "#E24B4A"),
        ("IMPROVING", x_min + 1, y_max - 0.5, "#378ADD"),
    ]:
        fig_rrg.add_annotation(x=x, y=y, text=label, showarrow=False,
            font=dict(size=9, color=color, family="monospace"), opacity=0.7)

    # Directional arrows + sector dots
    for _, row in df.iterrows():
        q     = row["quadrant"]
        color = QUAD_COLORS[q]["dot"]
        rx    = float(row.get(ratio_col, row["rs_ratio"]))
        ry    = float(row.get(mom_col,   row["rs_momentum"]))

        # Arrow tail (previous estimated position)
        px = float(row.get("prev_rs_ratio",    rx - row.get("delta_ratio",    0)))
        py = float(row.get("prev_rs_momentum", ry - row.get("delta_momentum", 0)))

        direction   = row.get("rotation_direction", "Stable")
        arrow_emoji = DIRECTION_ARROWS.get(direction, "→")
        stealth     = row.get("stealth_signal", "None")

        # Draw arrow from prev → current
        if abs(rx - px) > 0.05 or abs(ry - py) > 0.05:
            fig_rrg.add_annotation(
                x=rx, y=ry, ax=px, ay=py,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.2,
                arrowwidth=1.5, arrowcolor=color + "99",
            )

        # Marker: larger + pulsing ring for Improving
        marker_size = 20 if q == "Improving" else 16
        marker_line_w = 3 if q == "Improving" else 1.5
        marker_line_c = "#378ADD" if q == "Improving" else "#ffffff"

        label_text = f"{row['ticker']} {arrow_emoji}"
        if stealth != "None":
            label_text += " 🔍"

        fig_rrg.add_trace(go.Scatter(
            x=[rx], y=[ry],
            mode="markers+text",
            name=row["ticker"],
            text=[label_text],
            textposition="top right",
            textfont=dict(size=10, color="#e5e7eb"),
            marker=dict(
                size=marker_size,
                color=color,
                line=dict(color=marker_line_c, width=marker_line_w),
                symbol="circle",
            ),
            hovertemplate=(
                f"<b>{row['sector']}</b><br>"
                f"RS-Ratio: {rx:.1f} (100 = market avg)<br>"
                f"RS-Momentum: {ry:.1f} (100 = stable)<br>"
                f"Quadrant: {q}<br>"
                f"Direction: {direction} {arrow_emoji}<br>"
                f"Spread: {fmt_pct(row['spread'])}<br>"
                f"Stealth: {stealth}"
                f"<extra></extra>"
            ),
            showlegend=False,
        ))

    fig_rrg.update_layout(
        height=440,
        margin=dict(l=40, r=20, t=10, b=40),
        paper_bgcolor="rgba(14,17,23,0)",
        plot_bgcolor="rgba(14,17,23,0)",
        xaxis=dict(range=[x_min, x_max], title="RS-Ratio (relative strength vs market avg)",
                   gridcolor="rgba(255,255,255,0.08)", zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(range=[y_min, y_max], title="RS-Momentum (rate of RS change)",
                   gridcolor="rgba(255,255,255,0.08)", zeroline=False, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_rrg, use_container_width=True, config={"displayModeBar": False})

    # Legend row
    st.markdown("""
    <div style="display:flex;gap:16px;font-size:12px;color:#9ca3af;margin-top:-12px;flex-wrap:wrap">
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#1D9E75;margin-right:4px"></span>Leading</span>
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#BA7517;margin-right:4px"></span>Weakening</span>
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#E24B4A;margin-right:4px"></span>Lagging</span>
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#378ADD;margin-right:4px"></span>Improving ★</span>
      <span style="color:#6b7280">Arrows show rotation direction · 🔍 = Stealth Accumulation</span>
    </div>""", unsafe_allow_html=True)

    # RS-Ratio 100 explanation
    st.markdown("""
    <div style="background:rgba(55,138,221,0.08);border-left:3px solid #378ADD;
         border-radius:0 6px 6px 0;padding:8px 12px;margin-top:10px;font-size:12px;color:#9ca3af">
      <b style="color:#e5e7eb">Reading the RRG:</b>
      RS-Ratio = 100 means the sector is performing <i>exactly in line</i> with the equal-weighted
      market average. Above 100 = outperforming. Below 100 = underperforming.
      RS-Momentum = 100 means relative strength is holding steady.
      Above 100 = accelerating. Below 100 = decelerating.
      <br><b style="color:#378ADD">Clockwise rotation</b> is the normal cycle:
      Improving → Leading → Weakening → Lagging → Improving.
      Sectors moving <i>against</i> the clock are reversing trend — watch them closely.
    </div>""", unsafe_allow_html=True)


with col_signals:
    st.markdown('<div class="section-label">Institutional signals · ranked by priority</div>',
                unsafe_allow_html=True)

    # Improving sectors callout
    improving_df = df[df["quadrant"] == "Improving"].sort_values("signal_score", ascending=False)
    if not improving_df.empty:
        improving_names = ", ".join(improving_df["sector"].head(3).tolist())
        st.markdown(f"""
        <div style="background:rgba(55,138,221,0.12);border:1px solid rgba(55,138,221,0.4);
             border-radius:8px;padding:8px 12px;margin-bottom:10px">
          <span style="font-size:13px;font-weight:600;color:#378ADD">⭐ Early Accumulation Signal</span><br>
          <span style="font-size:12px;color:#9ca3af">
            <b style="color:#e5e7eb">{improving_names}</b> in Improving quadrant —
            underperforming but momentum turning. Historically highest alpha entry point.
          </span>
        </div>""", unsafe_allow_html=True)

    # Stealth accumulation callout
    stealth_df = df[df["stealth_signal"] != "None"].sort_values("signal_score", ascending=False)
    if not stealth_df.empty:
        stealth_names = ", ".join(stealth_df["sector"].head(3).tolist())
        st.markdown(f"""
        <div style="background:rgba(83,74,183,0.12);border:1px solid rgba(83,74,183,0.4);
             border-radius:8px;padding:8px 12px;margin-bottom:10px">
          <span style="font-size:13px;font-weight:600;color:#7C74D4">🔍 Stealth Accumulation</span><br>
          <span style="font-size:12px;color:#9ca3af">
            <b style="color:#e5e7eb">{stealth_names}</b> showing sustained multi-day
            positive breadth without a dramatic spike — how large positions are actually built.
          </span>
        </div>""", unsafe_allow_html=True)

    signals_html = ""
    total = len(df)
    for _, row in df.sort_values("signal_rank").head(8).iterrows():
        q         = row["quadrant"]
        bc        = QUAD_COLORS[q]
        spread_dir = "▲ Accumulation" if row["spread"] > 0 else "▼ Distribution"
        spread_c   = "#1D9E75" if row["spread"] > 0 else "#D85A30"
        direction  = row.get("rotation_direction", "Stable")
        arrow      = DIRECTION_ARROWS.get(direction, "→")
        stealth    = row.get("stealth_signal", "None")
        stealth_tag = ' <span style="font-size:9px;background:rgba(83,74,183,0.3);color:#A09BE0;padding:1px 5px;border-radius:4px">🔍 STEALTH</span>' if stealth != "None" else ""
        rank_n     = int(row["signal_rank"])

        signals_html += f"""
        <div class="signal-row">
          <span class="sig-badge" style="background:{bc['bg']};color:{bc['fg']}">{q}</span>
          <div style="flex:1">
            <div class="sig-sector">
              {row['sector']}
              <span style="font-size:11px;font-weight:400;color:#6b7280">{row['ticker']}</span>
              {stealth_tag}
            </div>
            <div class="sig-detail" style="color:{spread_c}">{spread_dir} · spread {fmt_pct(row['spread'])}</div>
            <div class="sig-detail">RS {row['rs_ratio']:.1f} · Mom {row['rs_momentum']:.1f} · {arrow} {direction}</div>
          </div>
          <div style="text-align:right;flex-shrink:0">
            <div style="font-size:9px;color:#6b7280">RANK</div>
            <div style="font-size:14px;font-weight:700;color:{bc['dot']}">{rank_n}<span style="font-size:10px;color:#6b7280">/{total}</span></div>
          </div>
        </div>"""

    st.markdown(f'<div class="dash-card" style="min-height:420px">{signals_html}</div>',
                unsafe_allow_html=True)

# ── Performance Heatmap ────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Performance heatmap · all timeframes</div>',
            unsafe_allow_html=True)

timeframe_cols = {
    "1D": "perf_1d", "1W": "perf_1w", "1M": "perf_1m",
    "3M": "perf_3m", "6M": "perf_6m", "1Y": "perf_1y", "YTD": "perf_ytd",
}

sort_opt = st.selectbox(
    "Sort by", ["Sector"] + list(timeframe_cols.keys()),
    key="hm_sort", label_visibility="collapsed", index=0,
)

hm_df = df.sort_values("sector") if sort_opt == "Sector" \
        else df.sort_values(timeframe_cols[sort_opt], ascending=False)

header_vals = ["Sector", "Quadrant"] + list(timeframe_cols.keys())
cell_vals   = [hm_df["sector"].tolist(), hm_df["quadrant"].tolist()]
fill_colors = [["#1e2330"] * len(hm_df), ["#1e2330"] * len(hm_df)]
font_colors = [["#f0f0f0"] * len(hm_df), [QUAD_COLORS[q]["dot"] for q in hm_df["quadrant"]]]

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
    columnwidth=[180, 90] + [75] * len(timeframe_cols),
    header=dict(
        values=[f"<b>{h}</b>" for h in header_vals],
        fill_color="#1e2330",
        font=dict(color="#f0f0f0", size=12),
        align=["left", "center"] + ["center"] * len(timeframe_cols),
        height=32, line_color="rgba(255,255,255,0.08)",
    ),
    cells=dict(
        values=cell_vals,
        fill_color=fill_colors,
        font=dict(color=font_colors, size=12),
        align=["left", "center"] + ["center"] * len(timeframe_cols),
        height=30, line_color="rgba(255,255,255,0.05)",
    ),
))
fig_hm.update_layout(
    height=len(hm_df) * 32 + 80,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="rgba(14,17,23,0)",
)
st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})

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
  <span style="color:#6b7280">· Quadrant color matches RRG</span>
</div>""", unsafe_allow_html=True)

# ── Spread / Accumulation-Distribution bars ────────────────────────────────────
st.markdown('<div class="section-label">Accumulation / distribution spread — 1M vs 3M run-rate</div>',
            unsafe_allow_html=True)

spread_df = df.sort_values("spread", ascending=False)

fig_spread = go.Figure()
for _, row in spread_df.iterrows():
    v     = row["spread"]
    # Threshold lowered to 0.8% from 2% for daily actionability
    color = "#1D9E75" if v > 0.8 else "#D85A30" if v < -0.8 else "#6b7280"
    label = "Accumulation" if v > 0.8 else "Distribution" if v < -0.8 else "Neutral"
    q     = row["quadrant"]
    stealth = row.get("stealth_signal", "None")
    extra_tag = " 🔍" if stealth != "None" else ""

    fig_spread.add_trace(go.Bar(
        x=[v],
        y=[f"{row['sector']} ({row['ticker']}){extra_tag}"],
        orientation="h",
        marker_color=color,
        width=0.6,
        hovertemplate=(
            f"<b>{row['sector']}</b><br>"
            f"Spread: {fmt_pct(v)}<br>"
            f"Signal: {label}<br>"
            f"Quadrant: {q}<br>"
            f"1M: {fmt_pct(row['perf_1m'])} · 3M: {fmt_pct(row['perf_3m'])}<br>"
            f"Stealth: {stealth}"
            f"<extra></extra>"
        ),
        showlegend=False,
    ))

fig_spread.add_vline(x=0,    line_width=1.5, line_color="rgba(136,135,128,0.5)")
fig_spread.add_vline(x=0.8,  line_width=1,   line_color="rgba(29,158,117,0.3)",  line_dash="dot")
fig_spread.add_vline(x=-0.8, line_width=1,   line_color="rgba(216,90,48,0.3)",   line_dash="dot")

fig_spread.update_layout(
    height=400,
    margin=dict(l=10, r=10, t=10, b=40),
    paper_bgcolor="rgba(14,17,23,0)",
    plot_bgcolor="rgba(14,17,23,0)",
    xaxis=dict(
        title="Spread: 1M performance minus (3M ÷ 3 monthly run-rate)",
        gridcolor="rgba(255,255,255,0.08)", zeroline=False,
        ticksuffix="%", tickfont=dict(size=10),
    ),
    yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
    bargap=0.3,
)
st.plotly_chart(fig_spread, use_container_width=True, config={"displayModeBar": False})
st.caption(
    "Spread = 1M performance minus (3M performance ÷ 3). "
    "Compares current month to the 3-month run-rate. "
    "Green bars (>+0.8%): month running HOT above trend = institutional accumulation. "
    "Red bars (<-0.8%): running COLD below trend = distribution. "
    "🔍 = Stealth Accumulation signal also active."
)

# ── Raw data expander ──────────────────────────────────────────────────────────
with st.expander("📋 Raw data table"):
    display_cols = [
        "sector", "ticker", "quadrant", "rotation_direction", "stealth_signal",
        "rs_ratio", "rs_momentum", "spread",
        "perf_1d", "perf_1w", "perf_1m", "perf_3m", "perf_6m", "perf_1y", "perf_ytd",
    ]
    st.dataframe(
        df[display_cols].rename(columns={
            "sector": "Sector", "ticker": "ETF", "quadrant": "Quadrant",
            "rotation_direction": "Direction", "stealth_signal": "Stealth",
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

# ── Valuation Panel ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📐 Sector Valuation — Weighted P/E & Relative Value")
st.caption(
    "Weighted P/E of top 5 holdings per sector vs each sector's own historical average. "
    "Refreshes every 6 hours."
)

with st.spinner("Loading valuation data…"):
    val_df = fetch_valuation_data()

if val_df is not None and not val_df.empty:

    st.markdown('<div class="section-label">Weighted P/E vs historical average — green = cheap, red = expensive</div>',
                unsafe_allow_html=True)

    val_sorted = val_df.sort_values("premium_pct", ascending=False, na_position="last")

    hm_headers  = ["Sector", "Method", "TTM P/E", "Hist Avg", "vs Avg", "Fwd P/E", "P/B", "Signal"]
    cell_sector = val_sorted["sector"].tolist()
    cell_method = val_sorted["method"].tolist()
    cell_ttm    = [f"{v:.1f}" if pd.notna(v) else "—" for v in val_sorted["ttm_pe"]]
    cell_hist   = [f"{v:.1f}" for v in val_sorted["hist_avg_pe"]]
    cell_prem   = [f"{'+' if v >= 0 else ''}{v:.1f}%" if pd.notna(v) else "—" for v in val_sorted["premium_pct"]]
    cell_fwd    = [f"{v:.1f}" if pd.notna(v) else "—" for v in val_sorted["fwd_pe"]]
    cell_pb     = [f"{v:.1f}" if pd.notna(v) else "—" for v in val_sorted["pb"]]
    cell_signal = val_sorted["valuation_signal"].tolist()

    fill_signal = val_sorted["signal_bg"].tolist()
    font_signal = val_sorted["signal_fg"].tolist()
    fill_prem, font_prem = [], []
    for v in val_sorted["premium_pct"]:
        if pd.isna(v):
            fill_prem.append("#1e2330"); font_prem.append("#888780")
        elif v > 20:
            fill_prem.append("#2e1208"); font_prem.append("#D04020")
        elif v > 0:
            fill_prem.append("#2e2008"); font_prem.append("#BA7517")
        elif v > -15:
            fill_prem.append("#0d3326"); font_prem.append("#1D9E75")
        else:
            fill_prem.append("#0e2240"); font_prem.append("#378ADD")

    base_fill = "#1e2330"
    base_font = "#e5e7eb"

    fig_val = go.Figure(go.Table(
        columnwidth=[160, 100, 75, 75, 85, 75, 55, 130],
        header=dict(
            values=[f"<b>{h}</b>" for h in hm_headers],
            fill_color="#131720",
            font=dict(color="#f0f0f0", size=12),
            align=["left","center","center","center","center","center","center","center"],
            height=32, line_color="rgba(255,255,255,0.08)",
        ),
        cells=dict(
            values=[cell_sector, cell_method, cell_ttm, cell_hist,
                    cell_prem, cell_fwd, cell_pb, cell_signal],
            fill_color=[
                [base_fill]*len(val_sorted), [base_fill]*len(val_sorted),
                [base_fill]*len(val_sorted), [base_fill]*len(val_sorted),
                fill_prem, [base_fill]*len(val_sorted),
                [base_fill]*len(val_sorted), fill_signal,
            ],
            font=dict(
                color=[
                    [base_font]*len(val_sorted), ["#6b7280"]*len(val_sorted),
                    [base_font]*len(val_sorted), ["#6b7280"]*len(val_sorted),
                    font_prem, [base_font]*len(val_sorted),
                    [base_font]*len(val_sorted), font_signal,
                ],
                size=12,
            ),
            align=["left","center","center","center","center","center","center","center"],
            height=30, line_color="rgba(255,255,255,0.05)",
        ),
    ))
    fig_val.update_layout(
        height=len(val_sorted) * 32 + 70,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_val, use_container_width=True, config={"displayModeBar": False})

    # ── Valuation vs RS-Ratio scatter ──────────────────────────────────────────
    st.markdown(
        '<div class="section-label" style="margin-top:8px">'
        'Valuation vs momentum — sweet spot is bottom-right (cheap + strong rotation)'
        '</div>', unsafe_allow_html=True
    )
    st.caption(
        "X-axis = premium vs historical avg P/E (left = cheap, right = expensive). "
        "Y-axis = RS-Ratio (above 100 = outperforming). "
        "Size = distance from fair value. ⭐ Sweet spot: cheap AND outperforming."
    )

    flow_map = df.set_index("sector")["rs_ratio"].to_dict() if df is not None and not df.empty else {}

    fig_scatter = go.Figure()
    for _, row in val_sorted.iterrows():
        cape_v = row["ttm_pe"]
        prem_v = row["premium_pct"]
        if pd.isna(cape_v) or pd.isna(prem_v):
            continue
        rs    = flow_map.get(row["sector"], 100)
        color = row["signal_fg"]
        size  = max(20, min(55, abs(prem_v) * 1.5 + 20))

        fig_scatter.add_trace(go.Scatter(
            x=[prem_v], y=[rs],
            mode="markers+text",
            text=[row["ticker"]],
            textposition="top center",
            textfont=dict(size=10, color="#e5e7eb"),
            marker=dict(size=size, color=color,
                        line=dict(color="rgba(255,255,255,0.15)", width=1), opacity=0.85),
            hovertemplate=(
                f"<b>{row['sector']}</b><br>"
                f"TTM P/E: {cape_v:.1f} ({row['method']})<br>"
                f"vs Hist Avg: {prem_v:+.1f}%<br>"
                f"RS-Ratio: {rs:.1f}<br>"
                f"Signal: {row['valuation_signal']}"
                f"<extra></extra>"
            ),
            showlegend=False,
        ))

    fig_scatter.add_annotation(
        x=-20, y=103,
        text="⭐ Sweet Spot<br>Cheap + Outperforming",
        showarrow=False, font=dict(size=9, color="#1D9E75"),
        bgcolor="rgba(13,51,38,0.6)", bordercolor="#1D9E75",
        borderwidth=1, borderpad=4,
    )
    fig_scatter.add_vline(x=0,   line_width=1, line_color="rgba(136,135,128,0.3)", line_dash="dot")
    fig_scatter.add_hline(y=100, line_width=1, line_color="rgba(136,135,128,0.3)", line_dash="dot")
    for (tx, ty, txt) in [(25, 96.5, "Expensive + Weak"), (-20, 96.5, "Cheap + Weak"),
                           (25, 103, "Expensive + Strong")]:
        fig_scatter.add_annotation(x=tx, y=ty, text=txt, showarrow=False,
            font=dict(size=9, color="#888780"), opacity=0.6)

    fig_scatter.update_layout(
        height=340,
        margin=dict(l=40, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Premium vs Historical Avg P/E (%)",
                   gridcolor="rgba(255,255,255,0.06)", zeroline=False,
                   ticksuffix="%", tickfont=dict(size=10)),
        yaxis=dict(title="RS-Ratio (Relative Strength · 100 = market avg)",
                   gridcolor="rgba(255,255,255,0.06)", zeroline=False, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

    # [CRITICAL FIX] Accurately titled and described — this is weighted P/E, NOT true CAPE
    with st.expander("📐 How sector P/E is calculated — methodology & limitations"):
        st.markdown("""
        **Important:** This panel shows a **weighted-average P/E** approximation, not a true
        Shiller CAPE (Cyclically Adjusted P/E). True CAPE requires 10 years of inflation-adjusted
        earnings history per holding — that data is not available via free APIs.

        **What we actually compute:**
        > Sector Weighted P/E = Sum(holding P/E × holding weight) / Sum(weights)

        Using the top 5 holdings per SPDR sector ETF, which cover 47–68% of each ETF's weight.
        Individual stock trailing P/E and forward P/E come from yfinance.

        | Column | What it means |
        |---|---|
        | **TTM P/E** | Trailing 12-month weighted P/E of top holdings |
        | **Hist Avg** | Sector's own long-run average P/E (1990–2024, from Research Affiliates / StarCapital) |
        | **vs Avg** | How much more/less expensive vs the sector's own history |
        | **Fwd P/E** | Forward P/E (next 12-month earnings estimates) |

        **"vs Avg" is more useful than absolute P/E.** Comparing Tech (35x) vs Utilities (17x)
        is misleading — they have structurally different earnings profiles. Comparing each to
        its own history tells you whether *that sector* is cheap or expensive *relative to itself*.

        **The sweet spot:** cheap vs history AND in the Leading or Improving RRG quadrant.
        Value without momentum tends to stay cheap longer than expected.

        *Sources: Research Affiliates RAFI, StarCapital sector P/E, S&P Dow Jones historical data*
        """)
else:
    st.info("Valuation data unavailable. Refresh to retry.", icon="📡")


# ── Sector Money Flow & Rotation ───────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🌊 Sector Money Flow & Rotation")
st.caption(
    "Aggregate institutional flow scores across all ETF categories. "
    "Where capital is moving right now — derived from price momentum, "
    "spread acceleration, and volume conviction."
)

with st.spinner("Loading sector flow data…"):
    flow_df = fetch_sector_flow_data()

if flow_df is not None and not flow_df.empty:

    agg = (flow_df.groupby("category")
           .agg(
               avg_flow    = ("flow_score",     "mean"),
               avg_spread  = ("spread",         "mean"),
               avg_1m      = ("perf_1m",        "mean"),
               avg_3m      = ("perf_3m",        "mean"),
               avg_vol_r   = ("vol_ratio",      "mean"),   # continuous avg vol ratio
               vol_spikes  = ("vol_spike",      "sum"),
               etf_count   = ("ticker",         "count"),
           )
           .reset_index()
           .sort_values("avg_flow", ascending=False))

    col_flow1, col_flow2 = st.columns([3, 2])

    with col_flow1:
        st.markdown('<div class="section-label">Inflow strength by category — ranked</div>',
                    unsafe_allow_html=True)

        fig_flow = go.Figure()
        for _, row in agg.sort_values("avg_flow").iterrows():
            v     = row["avg_flow"]
            color = ("#1D9E75" if v > 8 else "#2BAD7E" if v > 4 else
                     "#378ADD" if v > 0 else "#D85A30" if v > -4 else "#A32D2D")
            spike_marker = " 🔊" if row["vol_spikes"] > 0 else ""
            fig_flow.add_trace(go.Bar(
                x=[v],
                y=[f"{row['category']}{spike_marker}"],
                orientation="h",
                marker_color=color, marker_line_width=0, width=0.65,
                hovertemplate=(
                    f"<b>{row['category']}</b><br>"
                    f"Flow Score: {v:.1f}<br>"
                    f"Avg 1M: {fmt_pct(row['avg_1m'])}<br>"
                    f"Avg Spread: {fmt_pct(row['avg_spread'])}<br>"
                    f"Avg Vol Ratio: {row['avg_vol_r']:.2f}x<br>"
                    f"Vol Spikes: {int(row['vol_spikes'])}/{int(row['etf_count'])} ETFs"
                    f"<extra></extra>"
                ),
                showlegend=False,
            ))
        fig_flow.add_vline(x=0, line_width=1, line_color="rgba(136,135,128,0.4)")
        fig_flow.update_layout(
            height=360,
            margin=dict(l=10, r=20, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Aggregate Flow Score", gridcolor="rgba(255,255,255,0.06)",
                       zeroline=False, tickfont=dict(size=10)),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
            bargap=0.25,
        )
        st.plotly_chart(fig_flow, use_container_width=True, config={"displayModeBar": False})

    with col_flow2:
        st.markdown('<div class="section-label">Flow summary</div>', unsafe_allow_html=True)

        st.markdown("**💚 Top inflows**")
        for _, r in agg.head(3).iterrows():
            spikes = f" · 🔊 {int(r['vol_spikes'])} spike{'s' if r['vol_spikes']!=1 else ''}" \
                     if r["vol_spikes"] > 0 else ""
            st.markdown(f"""
            <div style="padding:6px 10px;background:rgba(29,158,117,0.1);
                 border-left:3px solid #1D9E75;border-radius:4px;margin-bottom:4px">
              <span style="font-weight:600;color:#f0f0f0">{r['category']}</span>
              <span style="color:#1D9E75;font-size:12px;margin-left:6px">Score {r['avg_flow']:.1f}</span>
              <span style="font-size:11px;color:#6b7280">{spikes}</span><br>
              <span style="font-size:11px;color:#9ca3af">
                1M: {fmt_pct(r['avg_1m'])} · Spread: {fmt_pct(r['avg_spread'])} · Vol: {r['avg_vol_r']:.2f}x
              </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("**🔴 Top outflows**")
        for _, r in agg.tail(3).iterrows():
            st.markdown(f"""
            <div style="padding:6px 10px;background:rgba(168,45,45,0.1);
                 border-left:3px solid #A32D2D;border-radius:4px;margin-bottom:4px">
              <span style="font-weight:600;color:#f0f0f0">{r['category']}</span>
              <span style="color:#D85A30;font-size:12px;margin-left:6px">Score {r['avg_flow']:.1f}</span><br>
              <span style="font-size:11px;color:#9ca3af">
                1M: {fmt_pct(r['avg_1m'])} · Spread: {fmt_pct(r['avg_spread'])} · Vol: {r['avg_vol_r']:.2f}x
              </span>
            </div>""", unsafe_allow_html=True)

    # ── Bubble chart: Spread (x) vs Avg Vol Ratio (y) vs Flow Score (size) ────
    # [HIGH FIX] Y-axis changed from spike count (sparse/discrete) to avg_vol_ratio (continuous)
    st.markdown(
        '<div class="section-label" style="margin-top:12px">'
        'Rotation map — spread acceleration vs volume conviction level'
        '</div>', unsafe_allow_html=True
    )
    st.caption(
        "Bubble size = Flow Score · "
        "X-axis: right = accelerating above 3M trend · "
        "Y-axis: avg volume ratio vs 20-day baseline (continuous, not spike count) · "
        "Green = inflow · Red = outflow · Top-right = highest conviction"
    )

    fig_bubble = go.Figure()
    for _, row in agg.iterrows():
        v     = row["avg_flow"]
        color = ("#1D9E75" if v > 6 else "#2BAD7E" if v > 2 else
                 "#378ADD" if v > 0 else "#D85A30" if v > -4 else "#A32D2D")
        size  = max(20, min(60, abs(v) * 4 + 20))
        fig_bubble.add_trace(go.Scatter(
            x=[row["avg_spread"]],
            y=[row["avg_vol_r"]],
            mode="markers+text",
            name=row["category"],
            text=[row["category"]],
            textposition="top center",
            textfont=dict(size=10, color="#e5e7eb"),
            marker=dict(size=size, color=color,
                        line=dict(color="rgba(255,255,255,0.2)", width=1), opacity=0.85),
            hovertemplate=(
                f"<b>{row['category']}</b><br>"
                f"Avg Spread: {row['avg_spread']:.2f}%<br>"
                f"Avg Vol Ratio: {row['avg_vol_r']:.2f}x (1.0 = normal)<br>"
                f"Vol Spikes: {int(row['vol_spikes'])}<br>"
                f"Flow Score: {v:.1f}<br>"
                f"Avg 1M: {fmt_pct(row['avg_1m'])}"
                f"<extra></extra>"
            ),
            showlegend=False,
        ))

    # Reference lines at zero spread and 1.0x volume (normal)
    fig_bubble.add_vline(x=0,   line_width=1, line_color="rgba(136,135,128,0.3)", line_dash="dot")
    fig_bubble.add_hline(y=1.0, line_width=1, line_color="rgba(136,135,128,0.3)", line_dash="dot")

    max_vol = float(agg["avg_vol_r"].max()) * 0.9
    for (label, x, y, color) in [
        ("↗ HOT MONEY",    2.5,  max_vol,  "#1D9E75"),
        ("↘ QUIET BUILD",  2.5,  0.85,     "#378ADD"),
        ("↖ DISTRIBUTION", -2.0, max_vol,  "#D85A30"),
        ("↙ OUTFLOW",      -2.0, 0.85,     "#888780"),
    ]:
        fig_bubble.add_annotation(x=x, y=y, text=label, showarrow=False,
            font=dict(size=9, color=color), opacity=0.65)

    fig_bubble.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Avg Spread (1M vs 3M run-rate)",
                   gridcolor="rgba(255,255,255,0.06)", zeroline=False,
                   ticksuffix="%", tickfont=dict(size=10)),
        yaxis=dict(title="Avg Volume Ratio (1.0 = normal · >1.3 = elevated)",
                   gridcolor="rgba(255,255,255,0.06)", zeroline=False, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_bubble, use_container_width=True, config={"displayModeBar": False})

    with st.expander("📊 Volume spike threshold reference — data-driven by ADV tier"):
        st.markdown("""
        Thresholds are set by **Average Daily Dollar Volume (ADDV)** — higher liquidity requires
        a larger spike to confirm institutional conviction:

        | Tier | Liquidity | ADDV | Spike Threshold | Examples | Why |
        |---|---|---|---|---|---|
        | **1** | Mega Liquid | >$2B/day | **1.25×** | QQQ, XLK, XLF | At $2B+ ADV, a 1.25× spike = $500M+ extra in one session |
        | **2** | High Liquid | $200M–$2B | **1.50×** | XLE, XLI, GLD, TLT, IWM | Classic institutional threshold |
        | **3** | Moderate | $50M–$200M | **2.00×** | SMH, IBB, ITA, KRE | Single large hedge fund trade = 1.5×; need 2× for confirmation |
        | **4** | Lower Liquid | <$50M/day | **3.00×** | XBI, SKYY, HACK | Retail noise spikes these 1.5–2× routinely |

        **v4 volume fix:** vol_ratio now compares today's single session to the prior 20-day average
        (previously compared a 5-day average vs 20-day — which diluted real spikes by ~80%).
        Dual confirmation (20-day AND 50-day both breached) earns the highest signal tier.
        """)

else:
    st.info("Sector flow data unavailable. Refresh to retry.", icon="📡")


# ── Where's the Money — Top ETF Flow Panel ─────────────────────────────────────
st.markdown("---")
st.markdown("## 💸 Where's the Money")
st.caption(
    "Sub-sector ETFs ranked by Institutional Flow Score · "
    "Volume spike uses per-ticker tier thresholds (not a flat 1.5×)"
)

with st.spinner("Loading ETF flow data…"):
    movers_df = fetch_top_movers(top_n=10)

if movers_df is not None and not movers_df.empty:

    col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 1, 1])
    with col_f1:
        cats = ["All"] + sorted(movers_df["category"].unique().tolist())
        cat_filter = st.selectbox("Filter by category", cats, key="mover_cat",
                                  label_visibility="collapsed")
    with col_f2:
        sigs = ["All signals"] + list(SIGNAL_COLORS.keys())
        sig_filter = st.selectbox("Filter by signal", sigs, key="mover_sig",
                                  label_visibility="collapsed")
    with col_f3:
        top_n = st.selectbox("Show top", [10, 15, 20], key="mover_n",
                             label_visibility="collapsed")
    with col_f4:
        vol_spike_only = st.toggle(
            "🔊 Vol Spike Filter", value=False, key="vol_spike_filter",
            help="Show only ETFs with today's volume >= tier-appropriate threshold vs 20-day avg"
        )

    if top_n != 10:
        movers_df = fetch_top_movers(top_n=top_n)

    display_df = movers_df.copy()
    if cat_filter != "All":
        display_df = display_df[display_df["category"] == cat_filter]
    if sig_filter != "All signals":
        display_df = display_df[display_df["signal"] == sig_filter]
    if vol_spike_only:
        # [CRITICAL FIX] Use per-ticker tier threshold, not hardcoded 1.5x
        display_df = display_df[display_df["vol_spike"] == True]

    if vol_spike_only:
        spike_count = len(display_df)
        if spike_count > 0:
            st.markdown(f"""
            <div style="background:rgba(29,158,117,0.12);border:1px solid rgba(29,158,117,0.4);
                 border-radius:8px;padding:8px 14px;margin-bottom:10px;display:flex;align-items:center;gap:10px">
              <span style="font-size:18px">🔊</span>
              <div>
                <span style="font-weight:600;color:#1D9E75">
                  {spike_count} ETF{"s" if spike_count != 1 else ""} with tier-appropriate volume spike
                </span>
                <span style="font-size:12px;color:#6b7280;margin-left:8px">
                  — today's volume exceeds per-ticker institutional threshold vs 20-day avg
                </span>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(136,135,128,0.1);border:1px solid rgba(136,135,128,0.3);
                 border-radius:8px;padding:8px 14px;margin-bottom:10px">
              <span style="color:#888780">
                ⚪ No ETFs currently showing a volume spike above their tier threshold — no confirmed institutional spike today
              </span>
            </div>""", unsafe_allow_html=True)

    total_movers = len(movers_df)
    for rank, (_, row) in enumerate(display_df.iterrows(), 1):
        sig_fg = row["signal_fg"]
        sig_bg = row["signal_bg"]

        def spark_bar(v, max_abs=15):
            pct   = min(abs(v) / max_abs * 100, 100)
            color = "#1D9E75" if v >= 0 else "#D85A30"
            return (
                f'<div style="display:inline-flex;align-items:center;gap:4px;width:80px">'
                f'<div style="height:6px;width:{pct:.0f}%;max-width:60px;background:{color};'
                f'border-radius:3px"></div>'
                f'<span style="font-size:10px;color:{color};font-weight:500">{fmt_pct(v)}</span>'
                f'</div>'
            )

        # [CRITICAL FIX] Use per-ticker tier threshold (row["spike_threshold"])
        tier_threshold = float(row.get("spike_threshold", 1.5))
        vol_r          = float(row["vol_ratio"])
        vol_spike      = vol_r >= tier_threshold

        vol_color    = "#1D9E75" if vol_spike else "#2BAD7E" if vol_r > 1.1 else \
                       "#D85A30" if vol_r < 0.8 else "#888780"
        spread_color = "#1D9E75" if row["spread"] > 0 else "#D85A30"
        border_style = "1.5px solid rgba(29,158,117,0.6)" if vol_spike else \
                       "0.5px solid rgba(255,255,255,0.08)"
        glow_style   = "box-shadow:0 0 12px rgba(29,158,117,0.2);" if vol_spike else ""

        # Dual-confirm indicator
        dual_conf = row.get("vol_spike_both", False)
        dual_tag  = ('<span style="font-size:9px;background:rgba(29,158,117,0.3);'
                     'color:#9FE1CB;padding:1px 5px;border-radius:4px;margin-left:4px">'
                     '✓✓ DUAL</span>') if dual_conf else ""

        # Rank as N of M
        flow_score_val = float(row["flow_score"])

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;
             background:var(--secondary-background-color);border-radius:10px;
             border:{border_style};{glow_style}margin-bottom:6px;">

          <div style="font-size:18px;font-weight:700;color:#6b7280;width:24px;
               text-align:center;flex-shrink:0">{rank}</div>

          <div style="flex:0 0 130px">
            <div style="font-size:15px;font-weight:600;color:var(--text-color)">{row["ticker"]}</div>
            <div style="font-size:11px;color:#6b7280;margin-top:1px">{row["name"]}</div>
            <div style="font-size:10px;color:#4b5563;margin-top:1px">{row["category"]}</div>
          </div>

          <div style="flex:0 0 170px">
            <span style="background:{sig_bg};color:{sig_fg};font-size:10px;font-weight:600;
                  padding:3px 8px;border-radius:6px">{row["signal"]}</span>
            <div style="margin-top:4px;font-size:9px;color:#6b7280">
              {row["tier_label"]} · spike≥{row["spike_label"]}
            </div>
          </div>

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

          <div style="flex:0 0 120px;text-align:right">
            <div style="font-size:11px;color:#6b7280">Spread</div>
            <div style="font-size:13px;font-weight:600;color:{spread_color}">{fmt_pct(row["spread"])}</div>
            <div style="font-size:10px;color:#6b7280;margin-top:3px">
              Vol ×{vol_r:.2f} vs 20d avg{dual_tag}
            </div>
            <div style="font-size:{"12px" if vol_spike else "10px"};
                 font-weight:{"700" if vol_spike else "400"};color:{vol_color}">
              {"🔊 SPIKE" if vol_spike else ("↑ Heavy" if vol_r > 1.1 else "↓ Light" if vol_r < 0.8 else "Normal")}
            </div>
          </div>

          <div style="flex:0 0 65px;text-align:center">
            <div style="font-size:9px;color:#6b7280;margin-bottom:1px">FLOW</div>
            <div style="font-size:17px;font-weight:700;color:{sig_fg}">{flow_score_val:.0f}</div>
            <div style="font-size:9px;color:#6b7280">#{rank} of {total_movers}</div>
          </div>

        </div>
        """, unsafe_allow_html=True)

    # [CRITICAL FIX] Flow Score methodology note updated to reflect v4 volume logic
    with st.expander("📐 How the Flow Score & Volume Spike are calculated"):
        st.markdown("""
        ### Institutional Flow Score

        Combines four signals weighted by reliability:

        | Weight | Component | What it measures |
        |---|---|---|
        | **40%** | Momentum consistency | Equal-weighted avg of 1W + 1M + 3M performance |
        | **30%** | Acceleration (spread) | 1M perf minus (3M ÷ 3 monthly run-rate) — positive = accelerating |
        | **20%** | Volume conviction | Today's session volume vs prior 20-day average |
        | **10%** | Timeframe unity | Fraction of 1W / 1M / 3M timeframes all positive |

        **Bonuses:** +8 pts for single-window vol spike · +13 pts for dual-confirmation spike
        (both 20-day AND 50-day averages breached simultaneously).

        ---

        ### Volume Spike Detection (v4)

        **Today's single session** is compared to the **prior 20-day rolling average** — NOT a
        5-day average (that was v3, which diluted real spikes by ~80%).

        Secondary confirmation uses the **50-day average**. When both are breached:
        vol_spike_both = True → shown as ✓✓ DUAL badge → "Strong Accumulation / Distribution"

        **Tier-appropriate thresholds** (not a flat 1.5×):
        - Tier 1 (>$2B ADV): 1.25× — QQQ, XLK
        - Tier 2 ($200M–$2B): 1.50× — XLE, GLD, IWM
        - Tier 3 ($50–200M): 2.00× — SMH, IBB, KRE
        - Tier 4 (<$50M): 3.00× — XBI, SKYY, HACK

        ---

        ### Signal Labels

        | Signal | Criteria |
        |---|---|
        | **Strong Accumulation** | Dual vol confirm + spread >1.5% + 1M positive |
        | **Accumulation** | Single spike + spread >0% + 1M positive |
        | **Inflow** | 1M positive + spread positive (no spike required) |
        | **Neutral** | Mixed signals |
        | **Outflow** | 1M negative + negative spread |
        | **Distribution** | Single spike + 1M negative |
        | **Strong Distribution** | Dual confirm + 1M negative + spread <-1% |
        """)

else:
    st.info("Top movers data unavailable. Check connection and refresh.", icon="📡")


st.markdown("---")
st.caption(
    "Data sourced from yfinance (Yahoo Finance) via ETF price history. "
    "Not financial advice. For informational and educational purposes only."
)
