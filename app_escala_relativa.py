from __future__ import annotations

"""Escala visual relativa 0–100 % para PREDWEEM MULTISITIO.

La simulación conserva EMERREL en su escala biofísica original. Esta capa
normaliza únicamente las variables de PRESENTACIÓN para que el máximo diario
de cada campaña represente 100 % de intensidad relativa. De este modo no se
alteran umbrales, filtros ecofisiológicos, validaciones ni agotamiento de
cohorte.
"""

from typing import Any, Callable

import numpy as np
import pandas as pd

import app_pronostico_multisitio as forecast_layer
import app_sanpedro_final as app_base
import app_umbral_operativo as threshold_layer
import app_zoom_operativo as zoom_layer
from visualizacion_operativa import SCIENTIFIC_SCALE


_BASE_EMERGENCE_FIGURE = zoom_layer._ORIGINAL_EMERGENCE_FIGURE
_BASE_FORECAST_FRAME = forecast_layer._forecast_frame
_BASE_ZOOM_STATUS = zoom_layer._seven_day_emergence_forecast
_BASE_THRESHOLD_STATUS = threshold_layer._seven_day_emergence_forecast


def intensidad_relativa_pct(values, reference_max: float | None = None) -> np.ndarray:
    """Normaliza una señal a 0–100 % preservando ceros y NaN como cero.

    El denominador es el máximo positivo de la campaña completa, salvo que se
    suministre explícitamente ``reference_max``. La función se usa sólo para
    presentación; EMERREL permanece intacta en el motor.
    """

    numeric = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0)
    array = np.clip(numeric.to_numpy(float), 0.0, None)
    maximum = float(np.nanmax(array)) if array.size else 0.0
    if reference_max is not None:
        maximum = max(float(reference_max), 0.0)
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros_like(array, dtype=float)
    return np.clip(array / maximum * 100.0, 0.0, 100.0)


def _arg(args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str):
    return args[index] if len(args) > index else kwargs.get(name)


def _campaign_max(data: Any) -> float:
    if not isinstance(data, pd.DataFrame) or "EMERREL" not in data:
        return 0.0
    values = pd.to_numeric(data["EMERREL"], errors="coerce").fillna(0.0)
    return max(float(values.clip(lower=0.0).max()), 0.0) if len(values) else 0.0


def _replace_customdata_column(customdata: Any, column: int, values: np.ndarray):
    if customdata is None:
        return customdata
    matrix = np.asarray(customdata, dtype=object).copy()
    if matrix.ndim != 2 or matrix.shape[0] != len(values) or matrix.shape[1] <= column:
        return customdata
    matrix[:, column] = values
    return matrix


def _normalize_figure_traces(
    figure: Any,
    data: pd.DataFrame,
    smooth: pd.DataFrame,
    scale_mode: str,
    peak: Any,
):
    """Reescala exclusivamente las trazas visuales a máximo diario = 100 %."""

    campaign_max = _campaign_max(data)
    if campaign_max <= 0.0:
        return figure

    raw_daily = pd.to_numeric(data["EMERREL"], errors="coerce").fillna(0.0).to_numpy(float)
    relative_daily = intensidad_relativa_pct(raw_daily, campaign_max)
    log_daily = np.log10(relative_daily + 1.0)

    if isinstance(smooth, pd.DataFrame) and "EMERREL_CAMPANA" in smooth:
        raw_smooth = pd.to_numeric(
            smooth["EMERREL_CAMPANA"], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        relative_smooth = intensidad_relativa_pct(raw_smooth, campaign_max)
        log_smooth = np.log10(relative_smooth + 1.0)
    else:
        raw_smooth = np.array([], dtype=float)
        relative_smooth = np.array([], dtype=float)
        log_smooth = np.array([], dtype=float)

    scientific = scale_mode == SCIENTIFIC_SCALE

    for trace in getattr(figure, "data", ()):
        name = str(getattr(trace, "name", "") or "")

        if name == "Emergencia diaria simulada" and len(relative_daily) == len(trace.y):
            trace.y = log_daily if scientific else relative_daily
            if scientific:
                custom = _replace_customdata_column(trace.customdata, 0, log_daily)
                custom = _replace_customdata_column(custom, 1, relative_daily)
            else:
                custom = _replace_customdata_column(trace.customdata, 0, relative_daily)
            trace.customdata = custom

        elif name == "Tendencia · pulsos agrupados" and len(relative_smooth) == len(trace.y):
            trace.y = log_smooth if scientific else relative_smooth
            if scientific:
                trace.customdata = _replace_customdata_column(
                    trace.customdata, 0, relative_smooth
                )

        elif name == "Primer pico válido" and peak is not None:
            peak_rows = data.loc[pd.to_datetime(data["Fecha"]) == pd.Timestamp(peak)]
            if peak_rows.empty:
                continue
            raw_peak = max(float(peak_rows.iloc[0]["EMERREL"]), 0.0)
            relative_peak = min(raw_peak / campaign_max * 100.0, 100.0)
            trace.y = [np.log10(relative_peak + 1.0) if scientific else relative_peak]
            trace.customdata = _replace_customdata_column(
                trace.customdata, 1, np.array([relative_peak], dtype=float)
            )

    metadata = dict(getattr(figure.layout, "meta", None) or {})
    metadata["predweem_relative_scale"] = {
        "definition": "100 * EMERREL / max(EMERREL de la campaña)",
        "campaign_max_emerrel": campaign_max,
        "display_max_pct": 100.0,
        "model_signal_modified": False,
    }
    figure.update_layout(meta=metadata)
    return figure


def _relative_emergence_figure(*args: Any, **kwargs: Any):
    """Envuelve el gráfico original sin alterar los datos del motor."""

    figure, x_range = _BASE_EMERGENCE_FIGURE(*args, **kwargs)
    data = _arg(args, kwargs, 0, "data")
    smooth = _arg(args, kwargs, 1, "smooth")
    peak = _arg(args, kwargs, 5, "peak")
    scale_mode = str(_arg(args, kwargs, 8, "scale_mode") or "Operativa (%)")
    if isinstance(data, pd.DataFrame) and isinstance(smooth, pd.DataFrame):
        _normalize_figure_traces(figure, data, smooth, scale_mode, peak)
    return figure, x_range


def _relative_forecast_frame(data: Any, today: Any) -> pd.DataFrame:
    """Normaliza el panel de pronóstico contra el máximo de toda la campaña."""

    frame = _BASE_FORECAST_FRAME(data, today)
    if frame.empty:
        return frame
    campaign_max = _campaign_max(data)
    frame = frame.copy()
    frame["Intensidad_relativa_emergencia_pct"] = intensidad_relativa_pct(
        frame["EMERREL"], campaign_max
    )
    return frame


def _relative_status(
    original: Callable[[Any, Any], dict[str, Any]],
    data: Any,
    today: Any,
) -> dict[str, Any]:
    """Mantiene el criterio EMERREL del semáforo y corrige sólo su porcentaje."""

    status = dict(original(data, today))
    campaign_max = _campaign_max(data)
    if campaign_max <= 0.0 or not isinstance(data, pd.DataFrame):
        status["max_intensity_pct"] = 0.0
        return status

    today_date = pd.Timestamp(today).normalize()
    start = today_date + pd.Timedelta(days=1)
    end = today_date + pd.Timedelta(days=7)
    subset = data.loc[:, ["Fecha", "EMERREL"]].copy()
    subset["Fecha"] = pd.to_datetime(subset["Fecha"], errors="coerce").dt.normalize()
    subset["EMERREL"] = pd.to_numeric(subset["EMERREL"], errors="coerce")
    subset = subset.dropna(subset=["Fecha", "EMERREL"])
    subset = subset.loc[(subset["Fecha"] >= start) & (subset["Fecha"] <= end)]
    if subset.empty:
        status["max_intensity_pct"] = 0.0
    else:
        status["max_intensity_pct"] = float(
            intensidad_relativa_pct(subset["EMERREL"], campaign_max).max()
        )
    return status


def _zoom_status_relative(data: Any, today: Any) -> dict[str, Any]:
    return _relative_status(_BASE_ZOOM_STATUS, data, today)


def _threshold_status_relative(data: Any, today: Any) -> dict[str, Any]:
    return _relative_status(_BASE_THRESHOLD_STATUS, data, today)


def run() -> None:
    """Ejecuta MULTISITIO con escala visual relativa correctamente normalizada."""

    original_figure = zoom_layer._ORIGINAL_EMERGENCE_FIGURE
    original_forecast = forecast_layer._forecast_frame
    original_zoom_status = zoom_layer._seven_day_emergence_forecast
    original_threshold_status = threshold_layer._seven_day_emergence_forecast

    zoom_layer._ORIGINAL_EMERGENCE_FIGURE = _relative_emergence_figure
    forecast_layer._forecast_frame = _relative_forecast_frame
    zoom_layer._seven_day_emergence_forecast = _zoom_status_relative
    threshold_layer._seven_day_emergence_forecast = _threshold_status_relative

    try:
        app_base.run()
    finally:
        zoom_layer._ORIGINAL_EMERGENCE_FIGURE = original_figure
        forecast_layer._forecast_frame = original_forecast
        zoom_layer._seven_day_emergence_forecast = original_zoom_status
        threshold_layer._seven_day_emergence_forecast = original_threshold_status
