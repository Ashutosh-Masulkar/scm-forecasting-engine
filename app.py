# app.py

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scripts.forecasting_engine import (
    load_walmart_data,
    get_store_data,
    run_arima_forecast,
    run_holtwinters_forecast,
    run_xgboost_forecast,
    build_forecast_results,
    calculate_inventory_metrics,
    calculate_safety_stock,
    calculate_forecast_accuracy_impact,
)


st.set_page_config(
    page_title="SCM Forecasting Engine",
    page_icon="📦",
    layout="wide",
)

PRIMARY = "#2563EB"
ACCENT = "#14B8A6"
WARN = "#F59E0B"
INK = "#0F172A"
SURFACE = "#FFFFFF"
BORDER = "#E2E8F0"

MODEL_COLORS = {"ARIMA": PRIMARY, "Holt-Winters": ACCENT, "XGBoost": WARN}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #F8FAFC; color: #0F172A; }
    .block-container { padding-top: 1.5rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: #0F172A; letter-spacing: -.01em; }

    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E2E8F0; }

    .hero { background: #0F172A; border-radius: 14px; padding: 28px 36px; margin-bottom: 24px; }
    .hero-tag { font-size: 11px; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 10px; }
    .hero-title { font-size: 30px; font-weight: 800; color: #F8FAFC; line-height: 1.15; margin-bottom: 8px; }
    .hero-sub { font-size: 14px; color: #94A3B8; line-height: 1.65; max-width: 700px; }

    .kpi { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px 20px;
           box-shadow: 0 1px 4px rgba(15,23,42,.05); margin-bottom: 12px; }
    .kpi-label { font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase;
                 letter-spacing: .06em; margin-bottom: 6px; }
    .kpi-value { font-size: 26px; font-weight: 800; color: #0F172A; line-height: 1.2; overflow-wrap: anywhere; }
    .kpi-note { font-size: 12px; color: #94A3B8; margin-top: 4px; }

    .insight { background: #EFF6FF; border-left: 4px solid #2563EB; border-radius: 0 10px 10px 0;
               padding: 12px 16px; color: #1E3A5F; font-size: 14px; line-height: 1.65; margin-bottom: 18px; }

    .winner { background: linear-gradient(135deg, #ECFDF5 0%, #EFF6FF 100%);
              border: 1px solid #A7F3D0; border-radius: 12px; padding: 18px 24px; margin-bottom: 18px; }
    .winner-label { font-size: 11px; font-weight: 700; color: #059669; text-transform: uppercase;
                    letter-spacing: .06em; margin-bottom: 4px; }
    .winner-name { font-size: 30px; font-weight: 800; color: #0F172A; }
    .winner-stat { font-size: 14px; color: #64748B; margin-top: 4px; }

    .formula { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px;
               padding: 16px 20px; font-size: 13px; color: #475569; line-height: 1.9; }

    .stButton > button { background: #2563EB !important; color: #FFFFFF !important;
                         border: none !important; border-radius: 8px !important;
                         font-weight: 600 !important; min-height: 42px !important; }
    .stButton > button:hover { background: #1D4ED8 !important;
                                box-shadow: 0 4px 12px rgba(37,99,235,.25) !important; }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
    .stTabs [data-baseweb="tab"] { background: #FFFFFF; border: 1px solid #E2E8F0;
                                    border-radius: 8px; padding: 8px 16px; font-weight: 500; color: #64748B; }
    .stTabs [aria-selected="true"] { background: #2563EB !important; color: #FFFFFF !important;
                                      border-color: #2563EB !important; }

    [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E2E8F0;
                                border-radius: 12px; padding: 16px 18px; }
    [data-testid="stMetricLabel"] { color: #64748B !important; font-size: 12px !important; }
    [data-testid="stMetricValue"] { color: #0F172A !important; font-size: 22px !important;
                                     white-space: normal; overflow-wrap: anywhere; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

DATA_PATH = os.path.join("data", "walmart-sales-dataset-of-45stores.csv")


@st.cache_data
def load_data():
    return load_walmart_data(DATA_PATH)


# ---------------------------------------------------------
# FORMATTERS
# ---------------------------------------------------------

def fmt_compact(v):
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:,.1f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:,.0f}K"
    return f"${v:,.2f}"


def fmt_number(v):
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:,.1f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:,.1f}K"
    return f"{v:,.0f}"


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def store_options(df):
    return ["All Stores"] + [f"Store {s}" for s in sorted(df["Store"].unique())]


def get_scope_label(sel):
    return "All Stores Network" if sel == "All Stores" else sel


def get_scope_data(df, sel):
    if sel == "All Stores":
        return (
            df.groupby("Date", as_index=False)
            .agg({"Weekly_Sales": "sum", "Holiday_Flag": "max",
                  "Temperature": "mean", "Fuel_Price": "mean",
                  "CPI": "mean", "Unemployment": "mean"})
            .sort_values("Date").reset_index(drop=True)
        )
    return get_store_data(df, int(sel.replace("Store ", "")))


def volatility_class(cv):
    if cv < 0.08:
        return "Stable"
    if cv < 0.16:
        return "Moderate"
    return "Volatile"


def style_fig(fig, height=450):
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=56, r=36, t=72, b=60),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=INK, family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False)
    return fig


def kpi(label, value, note=""):
    st.markdown(
        f'<div class="kpi">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def insight_card(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def forecast_config_key(sel, horizon):
    return f"{sel}|{horizon}"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Place `walmart-sales-dataset-of-45stores.csv` in the `data/` folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">Supply Chain Analytics · Portfolio Project</div>
        <div class="hero-title">SCM Demand Forecasting Engine</div>
        <div class="hero-sub">
            ARIMA · Holt-Winters · XGBoost with auto model selection, inventory policy sizing,
            and business impact quantification — built on Walmart's 45-store weekly sales dataset.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("Navigator")
page = st.sidebar.radio(
    "",
    ["Demand Review", "Forecast Engine", "Inventory Policy", "Business Impact"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.metric("Stores", df["Store"].nunique())
st.sidebar.metric("History", f"{df['Date'].min():%Y}–{df['Date'].max():%Y}")
st.sidebar.metric("Rows", f"{len(df):,}")


# ================================================================
# PAGE 1 — DEMAND REVIEW
# ================================================================

if page == "Demand Review":

    st.header("Demand Review")

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        sel = st.selectbox("Demand Scope", store_options(df))
    with c2:
        trend_view = st.radio("View", ["Weekly", "4-Week Rolling"], horizontal=True)
    with c3:
        compare_opts = ["None"] + [s for s in store_options(df) if s != sel]
        compare_sel = st.selectbox("Compare With", compare_opts)

    label = get_scope_label(sel)
    sdf = get_scope_data(df, sel)

    cv = sdf["Weekly_Sales"].std() / sdf["Weekly_Sales"].mean()
    vol = volatility_class(cv)
    total_avg = sdf["Weekly_Sales"].mean()
    recent_avg = sdf["Weekly_Sales"].tail(13).mean()
    bias = ((recent_avg - total_avg) / total_avg) * 100
    bias_dir = "above" if bias > 0 else "below"
    peak_val = sdf["Weekly_Sales"].max()
    peak_date = sdf.loc[sdf["Weekly_Sales"].idxmax(), "Date"]

    insight_card(
        f"<b>{label}</b> averaged <b>{fmt_compact(total_avg)}/week</b> over {len(sdf)} weeks. "
        f"Demand peaked at <b>{fmt_compact(peak_val)}</b> on {peak_date.strftime('%b %d, %Y')}. "
        f"Recent 13-week trend is <b>{abs(bias):.1f}% {bias_dir}</b> the full-history average — "
        f"<b>{vol}</b> demand profile (CV {cv:.1%})."
    )

    kc1, kc2, kc3 = st.columns(3)
    with kc1:
        kpi("Avg Weekly Sales", fmt_compact(total_avg), label)
    with kc2:
        kpi("Peak Week", fmt_compact(peak_val), peak_date.strftime("%b %d, %Y"))
    with kc3:
        kpi("Volatility", vol, f"CV {cv:.1%}")

    chart_df = sdf.copy()
    y_col = "Weekly_Sales"
    if trend_view == "4-Week Rolling":
        chart_df["Rolling"] = chart_df["Weekly_Sales"].rolling(4).mean()
        y_col = "Rolling"

    fig_trend = px.line(
        chart_df, x="Date", y=y_col,
        title=f"Weekly Sales Trend — {label}",
        markers=True, color_discrete_sequence=[PRIMARY],
    )

    if compare_sel != "None":
        cdf = get_scope_data(df, compare_sel).copy()
        if trend_view == "4-Week Rolling":
            cdf["Rolling"] = cdf["Weekly_Sales"].rolling(4).mean()
            cy = "Rolling"
        else:
            cy = "Weekly_Sales"
        fig_trend.add_trace(go.Scatter(
            x=cdf["Date"], y=cdf[cy], mode="lines",
            name=get_scope_label(compare_sel),
            line=dict(color=ACCENT, width=2, dash="dot"),
        ))

    fig_trend.update_layout(xaxis_title="Date", yaxis_title="Weekly Sales ($)")
    style_fig(fig_trend, height=460)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Seasonality")
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    sdf2 = sdf.copy()
    sdf2["Month"] = pd.Categorical(sdf2["Date"].dt.month_name(), categories=month_order, ordered=True)
    monthly = sdf2.groupby("Month", as_index=False, observed=True)["Weekly_Sales"].mean().sort_values("Month")

    peak_m = monthly.loc[monthly["Weekly_Sales"].idxmax(), "Month"]
    low_m = monthly.loc[monthly["Weekly_Sales"].idxmin(), "Month"]

    sc1, sc2 = st.columns([1.5, 1])
    with sc1:
        fig_m = px.bar(
            monthly, x="Month", y="Weekly_Sales",
            title=f"Average Monthly Sales Pattern — {label}",
            color="Weekly_Sales",
            color_continuous_scale=["#DBEAFE", PRIMARY],
        )
        fig_m.update_layout(xaxis_title="", yaxis_title="Avg Weekly Sales ($)", showlegend=False)
        style_fig(fig_m, height=380)
        st.plotly_chart(fig_m, use_container_width=True)
    with sc2:
        kpi("Peak Month", peak_m, "Highest average demand")
        kpi("Softest Month", low_m, "Lowest average demand")
        kpi("Recent Demand Bias", f"{bias:+.1f}%", "Last 13 weeks vs full history")

    with st.expander("Source rows"):
        st.dataframe(sdf, use_container_width=True)


# ================================================================
# PAGE 2 — FORECAST ENGINE
# ================================================================

elif page == "Forecast Engine":

    st.header("Forecast Engine")

    fc1, fc2, fc3 = st.columns([1.2, 1, 1])
    with fc1:
        sel_scope = st.selectbox("Demand Scope", store_options(df))
    with fc2:
        horizon = st.selectbox("Forecast Horizon (weeks)", [4, 8, 12], index=1)
    with fc3:
        sel_sdf = get_scope_data(df, sel_scope)
        st.metric("History Available", f"{len(sel_sdf)} weeks")

    current_key = forecast_config_key(sel_scope, horizon)
    if "forecast_key" in st.session_state and st.session_state["forecast_key"] != current_key:
        st.warning(
            "Scope or horizon changed since last run — results below are stale. "
            "Click **Run Models** to refresh."
        )

    if st.button("Run Models", type="primary"):
        results_list = []

        with st.status("Running forecast competition...", expanded=True) as status:
            st.write("Running ARIMA (statistical baseline)...")
            try:
                r = run_arima_forecast(sel_sdf, horizon)
                results_list.append(r)
                st.write(f"ARIMA complete — MAPE {r['mape']:.2f}%  |  RMSE ${r['rmse']:,.0f}")
            except Exception as e:
                st.write(f"ARIMA failed: {e}")

            st.write("Running Holt-Winters (seasonal exponential smoothing)...")
            try:
                r = run_holtwinters_forecast(sel_sdf, horizon)
                results_list.append(r)
                st.write(f"Holt-Winters complete — MAPE {r['mape']:.2f}%  |  RMSE ${r['rmse']:,.0f}")
            except Exception as e:
                st.write(f"Holt-Winters failed: {e}")

            st.write("Running XGBoost (ML with demand signals)...")
            try:
                r = run_xgboost_forecast(sel_sdf, horizon)
                results_list.append(r)
                st.write(f"XGBoost complete — MAPE {r['mape']:.2f}%  |  RMSE ${r['rmse']:,.0f}")
            except Exception as e:
                st.write(f"XGBoost failed: {e}")

            if not results_list:
                status.update(label="All models failed.", state="error")
            else:
                results = build_forecast_results(results_list)
                st.session_state.update({
                    "forecast_results": results,
                    "forecast_key": current_key,
                    "sel_scope": sel_scope,
                    "scope_label_val": get_scope_label(sel_scope),
                    "horizon": horizon,
                    "sel_sdf": sel_sdf,
                })
                status.update(
                    label=f"Done — {results['winning_model']} wins · MAPE {results['winning_mape']:.2f}%",
                    state="complete",
                )

        if not results_list:
            st.error("No models completed. Try a different store or horizon.")
            st.stop()

    if "forecast_results" in st.session_state:

        results = st.session_state["forecast_results"]
        win_df = results["winning_forecast_df"]
        all_val = results["all_validation_df"]
        slabel = st.session_state["scope_label_val"]
        h = st.session_state["horizon"]
        hist_sdf = st.session_state["sel_sdf"]

        avg_fc = win_df["Forecast"].mean()
        recent_baseline = hist_sdf["Weekly_Sales"].tail(h).mean()
        fc_change = ((avg_fc - recent_baseline) / recent_baseline) * 100
        direction = "above" if fc_change >= 0 else "below"

        st.markdown(
            f"""
            <div class="winner">
                <div class="winner-label">Winning Model</div>
                <div class="winner-name">{results['winning_model']}</div>
                <div class="winner-stat">
                    MAPE {results['winning_mape']:.2f}% &nbsp;·&nbsp;
                    avg forecast {fmt_compact(avg_fc)}/week &nbsp;·&nbsp;
                    {abs(fc_change):.1f}% {direction} recent baseline
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            kpi("Winning Model", results["winning_model"], "Lowest validation MAPE")
        with rc2:
            kpi("Winning MAPE", f"{results['winning_mape']:.2f}%", "Mean absolute % error")
        with rc3:
            kpi("Avg Forecast", fmt_compact(avg_fc), f"Next {h} weeks")

        tab_fut, tab_val, tab_race, tab_fi, tab_export = st.tabs(
            ["Future Forecast", "Validation", "Model Race", "Feature Importance", "Export"]
        )

        with tab_fut:
            hist_chart = hist_sdf.tail(52)
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=hist_chart["Date"], y=hist_chart["Weekly_Sales"],
                mode="lines+markers", name="Historical",
                line=dict(color=INK, width=2), marker=dict(size=4),
            ))
            fig_fc.add_trace(go.Scatter(
                x=win_df["Date"], y=win_df["Upper_Bound"],
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig_fc.add_trace(go.Scatter(
                x=win_df["Date"], y=win_df["Lower_Bound"],
                mode="lines", fill="tonexty",
                fillcolor="rgba(20,184,166,.14)",
                line=dict(width=0), name="95% Confidence",
            ))
            fig_fc.add_trace(go.Scatter(
                x=win_df["Date"], y=win_df["Forecast"],
                mode="lines+markers",
                name=f"{results['winning_model']} Forecast",
                line=dict(color=MODEL_COLORS.get(results["winning_model"], PRIMARY), width=3),
                marker=dict(size=7),
            ))
            fig_fc.update_layout(
                title=f"Demand Forecast — {slabel} (next {h} weeks)",
                xaxis_title="Date", yaxis_title="Weekly Sales ($)",
            )
            style_fig(fig_fc, height=500)
            st.plotly_chart(fig_fc, use_container_width=True)

        with tab_val:
            val_df = all_val[all_val["Model"] == results["winning_model"]]
            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(
                x=val_df["Date"], y=val_df["Actual"],
                mode="lines+markers", name="Actual",
                line=dict(color=INK, width=3),
            ))
            fig_val.add_trace(go.Scatter(
                x=val_df["Date"], y=val_df["Predicted"],
                mode="lines+markers", name="Predicted",
                line=dict(color=MODEL_COLORS.get(results["winning_model"], PRIMARY), width=3),
            ))
            fig_val.update_layout(
                title=f"Validation: Actual vs Predicted — {slabel}",
                xaxis_title="Date", yaxis_title="Weekly Sales ($)",
            )
            style_fig(fig_val, height=420)
            st.plotly_chart(fig_val, use_container_width=True)

        with tab_race:
            metrics = results["metrics_df"].copy()
            bar_colors = [MODEL_COLORS.get(m, PRIMARY) for m in metrics["Model"]]

            fig_race = go.Figure(go.Bar(
                y=metrics["Model"],
                x=metrics["MAPE (%)"],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.2f}%" for v in metrics["MAPE (%)"]],
                textposition="outside",
            ))
            fig_race.update_layout(
                title="Model Accuracy — Lower MAPE is Better",
                xaxis_title="MAPE (%)", yaxis_title="",
                showlegend=False,
            )
            style_fig(fig_race, height=260)
            st.plotly_chart(fig_race, use_container_width=True)

            winner_name = results["winning_model"]
            styled = metrics.style.apply(
                lambda row: [
                    "background-color: #ECFDF5; font-weight: 700;"
                    if row["Model"] == winner_name else ""
                    for _ in row
                ],
                axis=1,
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption("Winner highlighted. RMSE and MAE are in sales dollars — lower is better.")

        with tab_fi:
            fi = results.get("feature_importance")
            if fi:
                name_map = {
                    "Holiday_Flag": "Holiday Flag",
                    "Temperature": "Temperature",
                    "Fuel_Price": "Fuel Price",
                    "CPI": "CPI",
                    "Unemployment": "Unemployment",
                    "Week": "Week of Year",
                    "Month": "Month",
                    "Year": "Year",
                    "Lag_1": "Lag 1 Week",
                    "Lag_2": "Lag 2 Weeks",
                    "Lag_4": "Lag 4 Weeks",
                    "Lag_8": "Lag 8 Weeks",
                    "Rolling_Mean_4": "4-Week Rolling Mean",
                    "Rolling_Std_4": "4-Week Rolling Std",
                }
                fi_df = (
                    pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
                    .assign(Label=lambda d: d["Feature"].map(name_map).fillna(d["Feature"]))
                    .sort_values("Importance")
                )
                top_feat = fi_df.iloc[-1]["Label"]

                fig_fi = go.Figure(go.Bar(
                    y=fi_df["Label"],
                    x=fi_df["Importance"],
                    orientation="h",
                    marker=dict(
                        color=fi_df["Importance"],
                        colorscale=[[0, "#DBEAFE"], [1, PRIMARY]],
                        showscale=False,
                    ),
                    text=[f"{v:.3f}" for v in fi_df["Importance"]],
                    textposition="outside",
                ))
                fig_fi.update_layout(
                    title="XGBoost Feature Importance",
                    xaxis_title="Importance Score", yaxis_title="",
                    showlegend=False,
                )
                style_fig(fig_fi, height=460)
                st.plotly_chart(fig_fi, use_container_width=True)

                insight_card(
                    f"<b>{top_feat}</b> is the most influential signal in XGBoost for {slabel}. "
                    "Lag features capture recent sales momentum; economic signals add macroeconomic context. "
                    "Note: external features (CPI, Fuel Price, Unemployment) are held at last known values for future periods."
                )
            else:
                st.info("Feature importance is generated by XGBoost. Run a forecast where XGBoost completes successfully to see this chart.")

        with tab_export:
            st.subheader("Download Results")
            ec1, ec2 = st.columns(2)
            with ec1:
                st.download_button(
                    "Download Forecast (CSV)",
                    win_df.to_csv(index=False),
                    file_name=f"forecast_{slabel.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with ec2:
                val_export = all_val[all_val["Model"] == results["winning_model"]]
                st.download_button(
                    "Download Validation (CSV)",
                    val_export.to_csv(index=False),
                    file_name=f"validation_{slabel.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            st.dataframe(win_df, use_container_width=True, hide_index=True)

    else:
        st.info("Select a scope and horizon, then click **Run Models** to start the forecast competition.")


# ================================================================
# PAGE 3 — INVENTORY POLICY
# ================================================================

elif page == "Inventory Policy":

    st.header("Inventory Policy")

    if "forecast_results" not in st.session_state:
        st.warning("Run the **Forecast Engine** first to generate a planning forecast.")
        st.stop()

    results = st.session_state["forecast_results"]
    hist_sdf = st.session_state["sel_sdf"]
    win_df = results["winning_forecast_df"]
    slabel = st.session_state.get("scope_label_val", "Selected Scope")

    insight_card(
        f"Inventory policy derived from the <b>{results['winning_model']}</b> forecast for <b>{slabel}</b>. "
        "Adjust parameters below to explore how lead time and service level affect safety stock and holding cost."
    )

    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        unit_cost = st.slider("Unit Cost ($)", 1.0, 100.0, 25.0, 1.0)
    with pc2:
        hold_pct = st.slider("Holding Cost %", 1.0, 100.0, 20.0, 1.0)
    with pc3:
        lead_time = st.slider("Lead Time (weeks)", 1.0, 12.0, 4.0, 1.0)
    with pc4:
        service_level = st.slider("Service Level %", 90.0, 99.0, 95.0, 1.0)

    inv = calculate_inventory_metrics(
        forecast_df=win_df,
        historical_sales=hist_sdf["Weekly_Sales"],
        unit_cost=unit_cost,
        holding_cost_percentage=hold_pct,
        lead_time_weeks=lead_time,
        service_level=service_level,
    )

    policy_tab, tradeoff_tab, export_tab = st.tabs(["Policy Output", "Service Trade-Off", "Export"])

    with policy_tab:
        tc1, tc2 = st.columns(2)
        with tc1:
            kpi("Safety Stock", fmt_number(inv["safety_stock"]), "Buffer units for demand uncertainty")
            kpi("Inventory Value", fmt_compact(inv["inventory_value"]), "Safety stock × unit cost")
        with tc2:
            kpi("Reorder Point", fmt_number(inv["reorder_point"]), "Replenishment trigger level")
            kpi("Annual Holding Cost", fmt_compact(inv["annual_holding_cost"]), "Inventory value × holding %")

        if service_level >= 97 and lead_time >= 6:
            st.warning("High service level combined with long lead time is driving safety stock up significantly.")
        elif service_level <= 92 and lead_time <= 3:
            st.info("Lean posture — lower inventory exposure, but monitor stockout risk closely.")
        else:
            st.success("Balanced scenario — moderate inventory protection at reasonable carrying cost.")

        st.markdown(
            """
            <div class="formula">
                <b>Safety Stock</b> = Z × σ(demand) × √(lead time)<br>
                <b>Reorder Point</b> = avg weekly demand × lead time + safety stock<br>
                <b>Annual Holding Cost</b> = inventory value × holding cost %
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tradeoff_tab:
        tradeoff_rows = [
            {
                "Service Level": sl,
                "Safety Stock": calculate_safety_stock(inv["demand_std"], lead_time, sl),
                "Annual Carrying Cost": (
                    calculate_safety_stock(inv["demand_std"], lead_time, sl)
                    * unit_cost * (hold_pct / 100)
                ),
            }
            for sl in range(90, 100)
        ]
        tdf = pd.DataFrame(tradeoff_rows)

        fig_to = px.line(
            tdf, x="Service Level", y="Annual Carrying Cost",
            markers=True, title="Higher Service Level → Higher Inventory Cost",
            color_discrete_sequence=[ACCENT],
        )
        fig_to.add_vline(
            x=service_level, line_width=2, line_dash="dash",
            line_color=PRIMARY, annotation_text="Selected",
        )
        fig_to.update_layout(xaxis_title="Service Level (%)", yaxis_title="Annual Carrying Cost ($)")
        style_fig(fig_to, height=420)
        st.plotly_chart(fig_to, use_container_width=True)

    with export_tab:
        policy_df = pd.DataFrame([{
            "Scope": slabel,
            "Unit Cost ($)": unit_cost,
            "Holding Cost (%)": hold_pct,
            "Lead Time (weeks)": lead_time,
            "Service Level (%)": service_level,
            "Safety Stock": round(inv["safety_stock"], 0),
            "Reorder Point": round(inv["reorder_point"], 0),
            "Inventory Value ($)": round(inv["inventory_value"], 2),
            "Annual Holding Cost ($)": round(inv["annual_holding_cost"], 2),
        }])
        st.download_button(
            "Download Policy (CSV)",
            policy_df.to_csv(index=False),
            file_name="inventory_policy.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(policy_df.T.rename(columns={0: "Value"}), use_container_width=True)


# ================================================================
# PAGE 4 — BUSINESS IMPACT
# ================================================================

elif page == "Business Impact":

    st.header("Business Impact")

    if "forecast_results" not in st.session_state:
        st.warning("Run the **Forecast Engine** first to generate the planning forecast.")
        st.stop()

    results = st.session_state["forecast_results"]
    win_df = results["winning_forecast_df"]
    slabel = st.session_state.get("scope_label_val", "Selected Scope")

    insight_card(
        f"Quantifies the financial value of improving <b>{results['winning_model']}</b> forecast accuracy for <b>{slabel}</b>. "
        "Adjust the accuracy improvement slider to explore the sensitivity of inventory savings to forecast quality."
    )

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        imp_unit = st.slider("Avg Unit Cost ($)", 1.0, 100.0, 25.0, 1.0)
    with bc2:
        imp_hold = st.slider("Annual Holding Cost %", 1.0, 100.0, 20.0, 1.0)
    with bc3:
        imp_pct = st.slider("Forecast Accuracy Improvement %", 1.0, 50.0, 10.0, 1.0)

    avg_demand = win_df["Forecast"].mean()
    impact = calculate_forecast_accuracy_impact(
        current_mape=results["winning_mape"],
        average_weekly_demand=avg_demand,
        unit_cost=imp_unit,
        holding_cost_percentage=imp_hold,
        improvement_percentage=imp_pct,
    )

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        kpi("Current MAPE", f"{impact['current_mape']:.2f}%", "Winning model validation error")
    with ic2:
        kpi("Improved MAPE", f"{impact['improved_mape']:.2f}%", f"{imp_pct:.0f}% relative improvement")
    with ic3:
        kpi("Holding Cost Savings", fmt_compact(impact["annual_holding_savings"]), "Annual estimate")

    ic4, ic5 = st.columns(2)
    with ic4:
        kpi("Reduced Error Exposure", fmt_number(impact["reduced_error_units"]), "Weekly demand proxy reduction")
    with ic5:
        kpi("Inventory Value Reduction", fmt_compact(impact["inventory_value_reduction"]), "Working capital improvement")

    cc1, cc2 = st.columns(2)
    with cc1:
        err_df = pd.DataFrame({
            "Scenario": ["Current", "Improved"],
            "Forecast Error Exposure": [impact["current_error_units"], impact["improved_error_units"]],
        })
        fig_err = px.bar(
            err_df, x="Scenario", y="Forecast Error Exposure",
            title="Forecast Error Exposure: Before vs After",
            color="Scenario", color_discrete_sequence=[WARN, ACCENT],
        )
        fig_err.update_layout(xaxis_title="", yaxis_title="Weekly Demand Proxy", showlegend=False)
        style_fig(fig_err, height=360)
        st.plotly_chart(fig_err, use_container_width=True)

    with cc2:
        sav_df = pd.DataFrame({
            "Metric": ["Inventory Value Reduction", "Annual Holding Savings"],
            "Value": [impact["inventory_value_reduction"], impact["annual_holding_savings"]],
        })
        fig_sav = px.bar(
            sav_df, x="Metric", y="Value",
            title="Estimated Financial Impact",
            color="Metric", color_discrete_sequence=[PRIMARY, ACCENT],
        )
        fig_sav.update_layout(xaxis_title="", yaxis_title="Dollars ($)", showlegend=False)
        style_fig(fig_sav, height=360)
        st.plotly_chart(fig_sav, use_container_width=True)

    insight_card(
        f"A <b>{imp_pct:.0f}% improvement</b> in forecast accuracy "
        f"(MAPE <b>{impact['current_mape']:.1f}%</b> → <b>{impact['improved_mape']:.1f}%</b>) "
        f"reduces weekly error exposure by <b>{fmt_number(impact['reduced_error_units'])}</b> demand units, "
        f"freeing <b>{fmt_compact(impact['inventory_value_reduction'])}</b> in working capital and saving "
        f"<b>{fmt_compact(impact['annual_holding_savings'])}</b>/year in holding costs."
    )

    st.caption(
        "Directional portfolio estimate. Production use requires unit demand, item margin, "
        "stockout cost, and validated carrying cost assumptions."
    )
