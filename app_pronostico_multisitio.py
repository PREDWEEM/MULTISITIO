from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import app_agotamiento_balcarce as base
import app_detalle_1pct as detail
import app_zoom_operativo as zoom


principal = zoom.principal

FORECAST_FILL = "rgba(125, 211, 252, 0.22)"
FORECAST_LINE = "#0284C7"
FORECAST_BORDER = "rgba(2, 132, 199, 0.42)"

_ORIGINAL_EMERGENCE_FIGURE = zoom._ORIGINAL_EMERGENCE_FIGURE
_ORIGINAL_PLOTLY_CHART = zoom._ORIGINAL_PLOTLY_CHART
_ORIGINAL_LOW_EMERGENCE_FIGURE = detail._ORIGINAL_LOW_EMERGENCE_FIGURE

_FORECAST_PANELS: dict[int, tuple[pd.DataFrame, str, str]] = {}


def _forecast_frame(data: Any, today: Any) -> pd.DataFrame:
    """Extrae solo el tramo meteorológico marcado como pronóstico."""
    if not isinstance(data, pd.DataFrame):
        return pd.DataFrame()
    if "Fecha" not in data.columns or "EMERREL" not in data.columns:
        return pd.DataFrame()

    frame = data.copy()
    frame["Fecha"] = pd.to_datetime(frame["Fecha"], errors="coerce").dt.normalize()
    frame["EMERREL"] = pd.to_numeric(frame["EMERREL"], errors="coerce")
    frame = frame.dropna(subset=["Fecha", "EMERREL"])
    if frame.empty:
        return frame

    today_value = pd.Timestamp(today).normalize()
    mask = pd.Series(False, index=frame.index, dtype=bool)
    metadata_found = False
    for column in (
        "TipoDato",
        "TIPODATO",
        "TIPO",
        "Fuente",
        "FUENTE",
        "CalidadDato",
        "CALIDADDATO",
        "Calidad",
        "CALIDAD",
    ):
        if column not in frame.columns:
            continue
        metadata_found = True
        text = frame[column].astype("string").fillna("").str.lower()
        mask = mask | text.str.contains("pronost", regex=False)

    if metadata_found and bool(mask.any()):
        mask = mask & (frame["Fecha"] >= today_value)
    else:
        mask = frame["Fecha"] >= today_value

    forecast = frame.loc[mask, ["Fecha", "EMERREL"]].copy()
    if forecast.empty:
        return forecast

    forecast["Intensidad_relativa_emergencia_pct"] = (
        forecast["EMERREL"].clip(lower=0.0, upper=1.0) * 100.0
    )
    return (
        forecast.groupby("Fecha", as_index=False, sort=True)
        .agg(
            EMERREL=("EMERREL", "max"),
            Intensidad_relativa_emergencia_pct=(
                "Intensidad_relativa_emergencia_pct",
                "max",
            ),
        )
        .reset_index(drop=True)
    )


def _add_forecast_area(figure: go.Figure, forecast: pd.DataFrame) -> None:
    """Añade el área celeste que delimita el horizonte disponible."""
    if forecast.empty:
        return
    start = pd.Timestamp(forecast["Fecha"].min())
    end = pd.Timestamp(forecast["Fecha"].max())

    figure.add_vrect(
        x0=start,
        x1=end + pd.Timedelta(days=1),
        fillcolor=FORECAST_FILL,
        layer="below",
        line_width=0,
    )
    figure.add_annotation(
        x=start + (end - start) / 2,
        xref="x",
        y=0.995,
        yref="paper",
        text="<b>Pronóstico</b>",
        showarrow=False,
        xanchor="center",
        yanchor="top",
        bgcolor="rgba(240,249,255,0.94)",
        bordercolor=FORECAST_BORDER,
        borderwidth=1,
        borderpad=4,
        font={"size": 10, "color": "#075985"},
    )


def _forecast_figure(
    forecast: pd.DataFrame,
    site_name: str,
    model_name: str,
) -> go.Figure:
    """Gráfico del pronóstico usando la variable operativa en porcentaje."""
    start = pd.Timestamp(forecast["Fecha"].min())
    end = pd.Timestamp(forecast["Fecha"].max())
    values = forecast["Intensidad_relativa_emergencia_pct"].astype(float)
    labels = [
        f"{value:.3f}%" if value < 1.0 else f"{value:.1f}%"
        for value in values
    ]

    figure = go.Figure()
    figure.add_vrect(
        x0=start,
        x1=end + pd.Timedelta(days=1),
        fillcolor="rgba(125, 211, 252, 0.12)",
        layer="below",
        line_width=0,
    )
    figure.add_trace(
        go.Bar(
            x=forecast["Fecha"],
            y=values,
            name="Intensidad relativa de emergencia pronosticada",
            marker={
                "color": "rgba(2,132,199,0.64)",
                "line": {"color": "rgba(3,105,161,0.82)", "width": 0.5},
            },
            hovertemplate=(
                "<b>%{x|%d-%m-%Y}</b><br>"
                "Intensidad relativa de emergencia: %{y:.3f}%"
                "<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["Fecha"],
            y=values,
            mode="lines+markers+text",
            text=labels,
            textposition="top center",
            cliponaxis=False,
            name="Pronóstico diario",
            line={"color": FORECAST_LINE, "width": 2.4},
            marker={"size": 8, "color": FORECAST_LINE},
            hovertemplate=(
                "<b>%{x|%d-%m-%Y}</b><br>"
                "Intensidad relativa de emergencia: %{y:.3f}%"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                "<b>Pronóstico de intensidad relativa de emergencia</b><br>"
                "<span style='font-size:12px;color:#64748b'>"
                f"{site_name} · {model_name}"
                "</span>"
            ),
            "x": 0.0,
            "xanchor": "left",
            "font": {"size": 17, "color": "#0f172a"},
        },
        xaxis={
            "title": {"text": "Fecha", "standoff": 10},
            "tickmode": "array",
            "tickvals": forecast["Fecha"],
            "ticktext": forecast["Fecha"].dt.strftime("%d-%m"),
            "range": [
                start - pd.Timedelta(hours=12),
                end + pd.Timedelta(hours=12),
            ],
            "showgrid": False,
            "showline": True,
            "linecolor": "#94a3b8",
            "ticks": "outside",
            "fixedrange": False,
        },
        yaxis={
            "title": {
                "text": "Intensidad relativa de emergencia (%)",
                "standoff": 11,
            },
            "range": [0.0, 100.0],
            "tickmode": "array",
            "tickvals": [0, 20, 40, 60, 80, 100],
            "ticksuffix": "%",
            "showgrid": True,
            "gridcolor": "rgba(148,163,184,0.24)",
            "griddash": "dash",
            "zeroline": False,
            "fixedrange": False,
        },
        barmode="overlay",
        bargap=0.20,
        hovermode="x unified",
        height=390,
        margin={"l": 86, "r": 28, "t": 78, "b": 62},
        showlegend=False,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "color": "#334155"},
        dragmode="zoom",
    )
    return figure


def _emergence_figure_with_forecast(*args: Any, **kwargs: Any):
    """Añade banda celeste al gráfico principal y registra su panel futuro."""
    figure, x_range = _ORIGINAL_EMERGENCE_FIGURE(*args, **kwargs)

    data = args[0] if args else kwargs.get("data")
    site_name = str(args[2] if len(args) > 2 else kwargs.get("site_name", ""))
    model_name = str(args[3] if len(args) > 3 else kwargs.get("model_name", ""))
    today = args[9] if len(args) > 9 else kwargs.get("today")

    forecast = _forecast_frame(data, today)
    if forecast.empty:
        return figure, x_range

    _add_forecast_area(figure, forecast)
    _FORECAST_PANELS[id(figure)] = (forecast, site_name, model_name)

    end = pd.Timestamp(forecast["Fecha"].max())
    if isinstance(x_range, (list, tuple)) and len(x_range) == 2:
        current_end = pd.Timestamp(x_range[1])
        if end > current_end:
            x_range = [pd.Timestamp(x_range[0]), end]
            figure.update_xaxes(range=x_range)

    return figure, x_range


def _low_emergence_figure_with_forecast(
    data: Any,
    smooth: Any,
    x_range: Any,
    site_name: str,
    model_name: str,
    today: Any,
) -> go.Figure:
    """Mantiene el detalle 0–2 % e incorpora el mismo tramo celeste."""
    figure = _ORIGINAL_LOW_EMERGENCE_FIGURE(
        data,
        smooth,
        x_range,
        site_name,
        model_name,
        today,
    )
    forecast = _forecast_frame(data, today)
    _add_forecast_area(figure, forecast)
    return figure


def _plotly_chart_with_forecast(*args: Any, **kwargs: Any):
    """Renderiza el panel de pronóstico justo debajo del gráfico principal."""
    figure = args[0] if args else kwargs.get("figure_or_data")
    result = _ORIGINAL_PLOTLY_CHART(*args, **kwargs)

    panel = _FORECAST_PANELS.pop(id(figure), None)
    if panel is None:
        return result

    forecast, site_name, model_name = panel
    start = pd.Timestamp(forecast["Fecha"].min())
    end = pd.Timestamp(forecast["Fecha"].max())

    st.markdown("##### 🔭 Pronóstico de emergencia — horizonte disponible")
    st.caption(
        f"{site_name}: del {start.strftime('%d-%m-%Y')} "
        f"al {end.strftime('%d-%m-%Y')}. "
        "Variable Y: Intensidad relativa de emergencia (%). "
        "Se representan únicamente los días clasificados como pronóstico "
        "en la serie meteorológica activa."
    )
    _ORIGINAL_PLOTLY_CHART(
        _forecast_figure(forecast, site_name, model_name),
        width="stretch",
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "PREDWEEM_pronostico_intensidad_relativa_emergencia",
                "height": 760,
                "width": 2200,
                "scale": 2,
            },
        },
    )
    return result


def run() -> None:
    """Ejecuta MULTISITIO con sombreado y panel de pronóstico."""
    original_emergence = zoom._ORIGINAL_EMERGENCE_FIGURE
    original_plotly = zoom._ORIGINAL_PLOTLY_CHART
    original_low = detail._ORIGINAL_LOW_EMERGENCE_FIGURE
    original_st_plotly = st.plotly_chart
    original_principal_emergence = principal.emergence_figure

    _FORECAST_PANELS.clear()
    zoom._ORIGINAL_EMERGENCE_FIGURE = _emergence_figure_with_forecast
    zoom._ORIGINAL_PLOTLY_CHART = _plotly_chart_with_forecast
    detail._ORIGINAL_LOW_EMERGENCE_FIGURE = _low_emergence_figure_with_forecast

    try:
        base.run()
    finally:
        zoom._ORIGINAL_EMERGENCE_FIGURE = original_emergence
        zoom._ORIGINAL_PLOTLY_CHART = original_plotly
        detail._ORIGINAL_LOW_EMERGENCE_FIGURE = original_low
        principal.emergence_figure = original_principal_emergence
        st.plotly_chart = original_st_plotly
        _FORECAST_PANELS.clear()
