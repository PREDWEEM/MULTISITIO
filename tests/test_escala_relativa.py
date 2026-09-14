from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app_escala_relativa import (
    _normalize_figure_traces,
    _relative_forecast_frame,
    intensidad_relativa_pct,
)


def test_relative_intensity_reaches_100_without_changing_proportions():
    values = np.array([0.0, 0.095, 0.190, 0.0475])
    result = intensidad_relativa_pct(values)
    assert result.tolist() == pytest_approx([0.0, 50.0, 100.0, 25.0])


def pytest_approx(values):
    # Evita acoplar esta prueba simple a detalles internos de numpy.
    import pytest

    return pytest.approx(values)


def _figure_fixture():
    dates = pd.date_range("2026-02-17", periods=3, freq="D")
    data = pd.DataFrame(
        {
            "Fecha": dates,
            "EMERREL": [0.05, 0.19, 0.095],
        }
    )
    smooth = pd.DataFrame(
        {
            "Fecha": dates,
            "EMERREL_CAMPANA": [0.04, 0.18, 0.09],
        }
    )

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=dates,
            y=np.array(data["EMERREL"]) * 100.0,
            name="Emergencia diaria simulada",
            customdata=np.column_stack(
                [np.array(data["EMERREL"]) * 100.0, data["EMERREL"]]
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=np.array(smooth["EMERREL_CAMPANA"]) * 100.0,
            name="Tendencia · pulsos agrupados",
            customdata=smooth["EMERREL_CAMPANA"],
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[dates[1]],
            y=[19.0],
            name="Primer pico válido",
            customdata=[[0.19, 19.0]],
        )
    )
    return figure, data, smooth


def test_operational_figure_uses_campaign_peak_as_100_percent():
    figure, data, smooth = _figure_fixture()
    _normalize_figure_traces(
        figure,
        data,
        smooth,
        "Operativa (%)",
        pd.Timestamp("2026-02-18"),
    )

    bars = figure.data[0]
    assert max(bars.y) == pytest_approx(100.0)
    assert bars.y[0] == pytest_approx(0.05 / 0.19 * 100.0)
    assert bars.y[2] == pytest_approx(50.0)
    # EMERREL cruda permanece disponible para auditoría en customdata.
    assert float(bars.customdata[1][1]) == pytest_approx(0.19)
    assert float(bars.customdata[1][0]) == pytest_approx(100.0)

    meta = figure.layout.meta["predweem_relative_scale"]
    assert meta["campaign_max_emerrel"] == pytest_approx(0.19)
    assert meta["display_max_pct"] == pytest_approx(100.0)
    assert meta["model_signal_modified"] is False


def test_forecast_percent_is_relative_to_full_campaign_peak():
    data = pd.DataFrame(
        {
            "Fecha": pd.date_range("2026-02-17", periods=5, freq="D"),
            "EMERREL": [0.19, 0.095, 0.0475, 0.0, 0.0],
        }
    )
    forecast = _relative_forecast_frame(data, pd.Timestamp("2026-02-18"))
    assert not forecast.empty
    first = forecast.iloc[0]
    assert float(first["EMERREL"]) == pytest_approx(0.095)
    assert float(first["Intensidad_relativa_emergencia_pct"]) == pytest_approx(50.0)
