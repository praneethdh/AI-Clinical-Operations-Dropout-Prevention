"""
utils/charts.py
Plotly chart builders for SRS trends and risk dashboards.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


COLORS = {
    "green": "#2D6A4F",
    "green_light": "#52B788",
    "amber": "#F4A261",
    "red": "#E63946",
    "bg": "#161B22",
    "grid": "#21262D",
    "text": "#E6EDF3",
    "muted": "#8B949E",
}


def srs_trend_chart(sessions_df: pd.DataFrame, patient_name: str) -> go.Figure:
    """Line chart showing all 4 SRS dimensions over sessions."""
    fig = go.Figure()

    dims = {
        "srs_overall": ("Overall", COLORS["green_light"], 3),
        "srs_relationship": ("Relationship", "#74C0FC", 1.5),
        "srs_goals": ("Goals", COLORS["amber"], 1.5),
        "srs_approach": ("Approach", "#DA77F2", 1.5),
    }

    for col, (label, color, width) in dims.items():
        fig.add_trace(
            go.Scatter(
                x=sessions_df["session_number"],
                y=sessions_df[col],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=width),
                marker=dict(size=6 if width > 2 else 4),
            )
        )

    # Add danger zone
    fig.add_hrect(y0=0, y1=5, fillcolor=COLORS["red"], opacity=0.07, line_width=0)
    fig.add_hline(y=5, line_dash="dot", line_color=COLORS["red"], opacity=0.4,
                  annotation_text="Risk threshold", annotation_font_color=COLORS["red"])

    fig.update_layout(
        title=dict(text=f"SRS Score Trend — {patient_name}", font=dict(color=COLORS["text"], size=14)),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(
            title="Session Number",
            gridcolor=COLORS["grid"],
            tickmode="linear",
            dtick=1,
        ),
        yaxis=dict(title="Score (1–10)", gridcolor=COLORS["grid"], range=[0, 10.5]),
        legend=dict(bgcolor=COLORS["bg"], bordercolor=COLORS["grid"]),
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
    )
    return fig


def risk_gauge(score: int) -> go.Figure:
    """Gauge chart for baseline flight risk score."""
    if score >= 60:
        color = COLORS["red"]
    elif score >= 35:
        color = COLORS["amber"]
    else:
        color = COLORS["green_light"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Baseline Flight Risk", "font": {"color": COLORS["text"], "size": 13}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
                "bar": {"color": color},
                "bgcolor": COLORS["grid"],
                "steps": [
                    {"range": [0, 35], "color": "#0D1117"},
                    {"range": [35, 60], "color": "#1A1F27"},
                    {"range": [60, 100], "color": "#1C1418"},
                ],
                "threshold": {
                    "line": {"color": COLORS["red"], "width": 2},
                    "thickness": 0.75,
                    "value": 60,
                },
            },
            number={"font": {"color": color, "size": 36}, "suffix": "/100"},
        )
    )
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def therapist_load_chart(patients_df: pd.DataFrame) -> go.Figure:
    """Bar chart of patient load per therapist."""
    counts = patients_df["assigned_therapist_name"].value_counts().reset_index()
    counts.columns = ["therapist", "patients"]

    fig = px.bar(
        counts,
        x="patients",
        y="therapist",
        orientation="h",
        color="patients",
        color_continuous_scale=[[0, COLORS["green"]], [1, COLORS["green_light"]]],
    )
    fig.update_layout(
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(title="Patients", gridcolor=COLORS["grid"]),
        yaxis=dict(title=""),
        coloraxis_showscale=False,
        margin=dict(l=10, r=20, t=10, b=30),
        height=220,
    )
    return fig


def insurance_donut(patients_df: pd.DataFrame) -> go.Figure:
    """Donut chart of insurance distribution."""
    counts = patients_df["insurance_status"].value_counts()
    colors_map = {
        "Insured": COLORS["green_light"],
        "Out-of-Pocket": COLORS["amber"],
        "Uninsured": COLORS["red"],
    }
    clrs = [colors_map.get(k, COLORS["muted"]) for k in counts.index]

    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.55,
            marker=dict(colors=clrs, line=dict(color=COLORS["bg"], width=2)),
            textfont=dict(color=COLORS["text"]),
        )
    )
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        legend=dict(bgcolor=COLORS["bg"]),
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
    )
    return fig
