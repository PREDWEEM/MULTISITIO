from __future__ import annotations

"""Capa de integración de la versión final San Pedro 2025–2026 en MULTISITIO.

Las restantes localidades continúan usando el motor multisitio vigente. Para
San Pedro se reproduce la capa local congelada del repositorio geográfico:
termohidria continua + agotamiento causal de cohorte, sin modificar la ANN.
"""

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import app_agotamiento_balcarce as balcarce_layer
import app_pronostico_multisitio as base
import config_multisitio as calibration_registry
from predweem_core import (
    SimulationResult,
    calculate_et0_hargreaves,
    canonicalize_weather,
    cumulative_thermal_time_from_peak,
    first_peak_index,
    shift_signal,
    surface_parameters,
    surface_water_balance,
    thermal_time_scalar,
)
from sanpedro_calibracion_final import (
    ALPHA_HIDRICA_FINAL,
    CALIBRACION,
    CHOQUE_HIDRICO_FINAL,
    COBERTURA_FINAL,
    ESCALA_LLUVIA_TH_FINAL,
    EXPONENTE_KR_FINAL,
    FIN_CHOQUE_HIDRICO_JD_FINAL,
    K_COHORTE_FINAL,
    LATENCIA_JD_FINAL,
    PENDIENTE_TERMOHIDRICA_FINAL,
    REPOSITORIO_REFERENCIA,
    TECHO_CHOQUE_HIDRICO_FINAL,
    T0_TERMOHIDRICO_FINAL,
    UMBRAL_PRIMER_PICO_FINAL,
    VENTANA_TERMOHIDRICA_FINAL,
    VERSION_MOTOR,
    WMAX_FINAL,
    aplicar_agotamiento_cohorte,
    aplicar_interaccion_termohidrica,
)


principal = balcarce_layer.principal
_ORIGINAL_CORE_SIMULATE = balcarce_layer._ORIGINAL_SIMULATE_DUAL
_ORIGINAL_CORE_BUILD = balcarce_layer._ORIGINAL_BUILD_OPERATIONAL_DATA
_ORIGINAL_SLIDER = st.slider


def _install_runtime_profile() -> None:
    """Sincroniza los campos comunes del registro con la calibración final."""
    current = calibration_registry.SITE_CALIBRATIONS["san-pedro"]
    calibration_registry.SITE_CALIBRATIONS["san-pedro"] = replace(
        current,
        cobertura_predeterminada_pct=COBERTURA_FINAL,
        wmax_predeterminado_mm=WMAX_FINAL,
        exponente_kr_predeterminado=EXPONENTE_KR_FINAL,
        latencia_jd=LATENCIA_JD_FINAL,
        ventana_termica_dias=VENTANA_TERMOHIDRICA_FINAL,
        umbral_termoinhibicion_c=T0_TERMOHIDRICO_FINAL,
        umbral_termoinhibicion_con_lag_c=T0_TERMOHIDRICO_FINAL,
        ventana_lluvia_dias=3,
        umbral_choque_hidrico_mm=CHOQUE_HIDRICO_FINAL,
        fin_choque_hidrico_jd=FIN_CHOQUE_HIDRICO_JD_FINAL,
        techo_choque_hidrico=TECHO_CHOQUE_HIDRICO_FINAL,
        umbral_primer_pico=UMBRAL_PRIMER_PICO_FINAL,
        modelo_referencia_local="sin_lag",
        repositorio_referencia=REPOSITORIO_REFERENCIA,
        archivo_motor_referencia="sanpedro_calibracion_final.py",
    )


_install_runtime_profile()


def _is_san_pedro(config: Any | None = None) -> bool:
    selected = str(st.session_state.get("selected_lolium_site", "")).strip().lower()
    if selected == "san-pedro":
        return True
    configured = str(getattr(config, "nombre_sitio", "")).strip().lower()
    return configured.startswith("san pedro")


def _days_since_peak(length: int, peak_index: int | None) -> np.ndarray:
    if peak_index is None:
        return np.zeros(length, dtype=float)
    return np.maximum(np.arange(length, dtype=float) - int(peak_index), 0.0)


def _shift_aux(values, lag_days: int, fill: float = 0.0) -> np.ndarray:
    values = np.asarray(values)
    if int(lag_days) == 0:
        return values.copy()
    output = np.full(values.shape, fill, dtype=values.dtype)
    lag = int(lag_days)
    if lag > 0 and lag < len(values):
        output[lag:] = values[:-lag]
    elif lag < 0 and abs(lag) < len(values):
        output[:lag] = values[-lag:]
    return output


def _simulate_san_pedro_final(
    raw_weather,
    ann,
    *,
    coverage_percent: float,
    wmax: float,
    lag_days: int,
    kr_exponent: float = 0.0,
    config,
) -> SimulationResult:
    """Reproduce en MULTISITIO el motor final del repositorio San Pedro."""
    data = canonicalize_weather(raw_weather)
    data["Julian_days"] = data["Fecha"].dt.dayofyear
    data["Tmedia_aire"] = (data["TMAX"] + data["TMIN"]) / 2.0

    coverage = float(COBERTURA_FINAL)
    wmax_local = float(WMAX_FINAL)
    kr_local = float(EXPONENTE_KR_FINAL)
    ke, thermal_modulator = surface_parameters(coverage)
    data["Cobertura_Rastrojo"] = coverage
    data["Ke_Suelo"] = ke
    data["Exponente_Kr"] = kr_local

    inputs = data[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    raw_ann = np.clip(ann.predict(inputs), 0.0, 1.0)
    data["EMERREL_RAW_ANN"] = raw_ann
    data["EMERREL_RAW"] = raw_ann

    emer_base = raw_ann.copy()
    data["Prec_3d"] = data["Prec"].rolling(window=3, min_periods=1).sum()
    hydric_shock = (
        (data["Julian_days"] > LATENCIA_JD_FINAL)
        & (data["Julian_days"] <= FIN_CHOQUE_HIDRICO_JD_FINAL)
        & (data["Prec_3d"] >= CHOQUE_HIDRICO_FINAL)
    )
    shock_values = hydric_shock.to_numpy(bool)
    emer_base[shock_values] = np.maximum(
        emer_base[shock_values],
        TECHO_CHOQUE_HIDRICO_FINAL,
    )
    data["Choque_Hidrico"] = hydric_shock

    data["ET0"] = calculate_et0_hargreaves(
        data["Julian_days"],
        data["TMAX"],
        data["TMIN"],
        float(config.latitud),
    )
    water, kr_daily = surface_water_balance(
        data["Prec"],
        data["ET0"],
        wmax_local,
        ke,
        kr_local,
    )
    data["W_superficial"] = water
    data["Kr_Diario"] = kr_daily
    relative_water = water / max(wmax_local, 1e-12)
    data["Humedad_Relativa"] = relative_water

    hydric_factor = 1.0 / (
        1.0
        + np.exp(
            -float(config.pendiente_hidrica)
            * (relative_water - float(config.p50_hidrico))
        )
    )
    data["Hydric_Factor"] = hydric_factor
    emer_base *= hydric_factor
    emer_base[relative_water < float(config.corte_hidrico)] = 0.0

    recharge = (data["Prec"] >= wmax_local).cummax().to_numpy(bool)
    data["Lluvia_Recarga"] = recharge
    emer_base[~recharge] = 0.0

    data["EMERREL_ANTES_TERMOHIDRIA"] = emer_base.copy()
    (
        thermohydric_signal,
        thermohydric_factor,
        temperature_window,
        hydric_index,
        effective_tcrit,
        thermal_diagnostic,
    ) = aplicar_interaccion_termohidrica(
        emer_base,
        temperatura_media=data["Tmedia_aire"],
        humedad_relativa=relative_water,
        precipitacion_3d=data["Prec_3d"],
        t0_base=T0_TERMOHIDRICO_FINAL,
        alivio_hidrico=ALPHA_HIDRICA_FINAL,
        pendiente_c=PENDIENTE_TERMOHIDRICA_FINAL,
        ventana_dias=VENTANA_TERMOHIDRICA_FINAL,
        escala_lluvia_mm=ESCALA_LLUVIA_TH_FINAL,
    )
    data["Tmedia_TH"] = temperature_window
    data["Tmedia_5d"] = temperature_window
    data["Indice_Hidrico_Termico"] = hydric_index
    data["Tcrit_Efectiva"] = effective_tcrit
    data["Factor_TermoHidrico"] = thermohydric_factor
    data["Termoinhibida"] = thermal_diagnostic
    data["Termoinhibida_SIN_LAG"] = thermal_diagnostic
    data["Termoinhibida_CON_LAG"] = thermal_diagnostic
    data["Umbral_Termoinhibicion_SIN_LAG_C"] = effective_tcrit
    data["Umbral_Termoinhibicion_CON_LAG_C"] = effective_tcrit

    julian = data["Julian_days"].to_numpy(float)
    potential = np.clip(thermohydric_signal, 0.0, 1.0)
    data["EMERREL_ANTES_FILTRO_PRIMER_PICO"] = potential.copy()
    potential[julian <= LATENCIA_JD_FINAL] = 0.0
    idx0 = first_peak_index(potential, UMBRAL_PRIMER_PICO_FINAL)
    if idx0 is None:
        potential[:] = 0.0
    else:
        potential[:idx0] = 0.0

    enabled = np.zeros(len(data), dtype=bool)
    peak_marker = np.zeros(len(data), dtype=bool)
    if idx0 is not None:
        enabled[idx0:] = True
        peak_marker[idx0] = True
    data["Primer_Pico_Habilitado"] = enabled
    data["Primer_Pico_Calibrado_SIN_LAG"] = peak_marker

    (
        emer_no_lag,
        reserve_no_lag,
        released_no_lag,
        cohort_factor_no_lag,
    ) = aplicar_agotamiento_cohorte(
        potential,
        idx0,
        k_cohorte=K_COHORTE_FINAL,
    )

    lag = int(lag_days)
    emer_lag = shift_signal(emer_no_lag, lag)
    if idx0 is None:
        idx_lag = None
    else:
        candidate = int(idx0) + lag
        idx_lag = candidate if 0 <= candidate < len(data) else None

    potential_lag = shift_signal(potential, lag)
    reserve_lag = _shift_aux(reserve_no_lag, lag, fill=1.0)
    released_lag = _shift_aux(released_no_lag, lag, fill=0.0)
    cohort_factor_lag = _shift_aux(cohort_factor_no_lag, lag, fill=1.0)
    peak_marker_lag = np.zeros(len(data), dtype=bool)
    if idx_lag is not None:
        peak_marker_lag[idx_lag] = True
    data["Primer_Pico_Calibrado_CON_LAG"] = peak_marker_lag

    data["EMERREL_SIN_LAG_ANTES_DECAIMIENTO"] = potential
    data["FACTOR_DECAIMIENTO_SIN_LAG"] = cohort_factor_no_lag
    data["DIAS_DESDE_PICO_SIN_LAG"] = _days_since_peak(len(data), idx0)
    data["EMERREL_SIN_LAG"] = emer_no_lag
    data["Reserva_Cohorte_SIN_LAG"] = reserve_no_lag
    data["Fraccion_Liberada_Cohorte_SIN_LAG"] = released_no_lag
    data["Factor_Cohorte_SIN_LAG"] = cohort_factor_no_lag
    data["EMERREL_SIN_LAG_ANTES_COHORTE"] = potential

    data["EMERREL_CON_LAG_ANTES_DESPLAZAMIENTO"] = emer_no_lag
    data["EMERREL_CON_LAG_ANTES_DECAIMIENTO"] = potential_lag
    data["FACTOR_DECAIMIENTO_CON_LAG"] = cohort_factor_lag
    data["DIAS_DESDE_PICO_CON_LAG"] = _days_since_peak(len(data), idx_lag)
    data["EMERREL_CON_LAG"] = emer_lag
    data["Reserva_Cohorte_CON_LAG"] = reserve_lag
    data["Fraccion_Liberada_Cohorte_CON_LAG"] = released_lag
    data["Factor_Cohorte_CON_LAG"] = cohort_factor_lag
    data["EMERREL_CON_LAG_ANTES_COHORTE"] = potential_lag

    data["Decaimiento_Activo"] = False
    data["Decaimiento_Tau_Dias"] = np.nan
    data["Decaimiento_Beta"] = np.nan
    data["Decaimiento_Intensidad"] = np.nan
    data["Termohidria_Continua_Activa"] = True
    data["Agotamiento_Cohorte_Causal_Activo"] = True
    data["K_Cohorte_Causal"] = float(K_COHORTE_FINAL)
    data["Motor_Local_Version"] = VERSION_MOTOR
    data["Calibracion_Local"] = CALIBRACION
    data["Modelo_Referencia_Local"] = "sin_lag"

    data["EMERAC_SIN_LAG"] = np.cumsum(emer_no_lag)
    data["EMERAC_CON_LAG"] = np.cumsum(emer_lag)

    data["GD_Tb2"] = [
        thermal_time_scalar(
            temperature,
            float(config.t_base_c),
            float(config.t_optima_c),
            float(config.t_critica_c),
        )
        for temperature in data["Tmedia_aire"]
    ]
    data["TT_ACUM"] = data["GD_Tb2"].cumsum()
    data["TT_DESDE_PICO_SIN_LAG"] = cumulative_thermal_time_from_peak(
        data["GD_Tb2"], idx0
    )
    data["TT_DESDE_PICO_CON_LAG"] = cumulative_thermal_time_from_peak(
        data["GD_Tb2"], idx_lag
    )

    peak0 = pd.Timestamp(data.loc[idx0, "Fecha"]) if idx0 is not None else None
    peak_lag = (
        pd.Timestamp(data.loc[idx_lag, "Fecha"])
        if idx_lag is not None
        else None
    )
    return SimulationResult(data, peak0, peak_lag, ke, thermal_modulator)


def _simulate_dual_with_san_pedro(*args: Any, **kwargs: Any):
    config = kwargs.get("config")
    if config is None or not _is_san_pedro(config):
        return _ORIGINAL_CORE_SIMULATE(*args, **kwargs)
    return _simulate_san_pedro_final(*args, **kwargs)


def _build_operational_data_with_san_pedro(
    data: pd.DataFrame,
    model_mode: str,
    lag_days: int,
):
    is_local = (
        "Motor_Local_Version" in data
        and data["Motor_Local_Version"].astype(str).eq(VERSION_MOTOR).any()
    )
    if not is_local:
        return _ORIGINAL_CORE_BUILD(data, model_mode, lag_days)

    suffix = "CON_LAG" if model_mode == "con_lag" else "SIN_LAG"
    audit_columns = {
        "Reserva_Cohorte": f"Reserva_Cohorte_{suffix}",
        "Fraccion_Liberada_Cohorte": f"Fraccion_Liberada_Cohorte_{suffix}",
        "Factor_Cohorte": f"Factor_Cohorte_{suffix}",
        "EMERREL_ANTES_COHORTE": f"EMERREL_{suffix}_ANTES_COHORTE",
    }
    audit = {
        name: data[column].to_numpy(copy=True)
        for name, column in audit_columns.items()
        if column in data
    }
    peak_column = f"Primer_Pico_Calibrado_{suffix}"
    calibrated_peak = None
    if peak_column in data:
        positions = np.flatnonzero(data[peak_column].to_numpy(bool))
        if positions.size:
            calibrated_peak = pd.Timestamp(data.iloc[int(positions[0])]["Fecha"])

    output, _, generic_peak = _ORIGINAL_CORE_BUILD(data, model_mode, lag_days)
    for name, values in audit.items():
        output[name] = values
    output["Motor_Local_Version"] = VERSION_MOTOR
    output["Calibracion_Local"] = CALIBRACION
    output["Termohidria_Continua_Activa"] = True
    output["Agotamiento_Cohorte_Causal_Activo"] = True
    output["Cobertura_Calibrada_Pct"] = float(COBERTURA_FINAL)
    output["Wmax_Calibrado_mm"] = float(WMAX_FINAL)
    output["K_Cohorte_Causal"] = float(K_COHORTE_FINAL)
    model_name = f"Sin lag · {VERSION_MOTOR}" if model_mode == "sin_lag" else f"Con lag · {VERSION_MOTOR}"
    output["Modelo_Operativo"] = model_name
    return output, model_name, calibrated_peak or generic_peak


def _slider_with_frozen_san_pedro(*args: Any, **kwargs: Any):
    label = str(args[0] if args else kwargs.get("label", ""))
    if label == "Cobertura de rastrojo (%)" and _is_san_pedro():
        st.number_input(
            "Cobertura efectiva calibrada (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(COBERTURA_FINAL),
            step=0.01,
            format="%.2f",
            disabled=True,
            key="coverage_final_san_pedro",
            help=(
                "Parámetro congelado por calibración conjunta San Pedro 2025–2026. "
                "MULTISITIO ignora ajustes manuales de cobertura para esta localidad."
            ),
        )
        st.caption(
            f"San Pedro usa {COBERTURA_FINAL:.2f}% de cobertura efectiva y "
            f"Wmax={WMAX_FINAL:.2f} mm, congelados en {CALIBRACION}."
        )
        return float(COBERTURA_FINAL)
    return _ORIGINAL_SLIDER(*args, **kwargs)


def run() -> None:
    """Ejecuta MULTISITIO con la versión final local de San Pedro integrada."""
    original_balcarce_simulate = balcarce_layer._ORIGINAL_SIMULATE_DUAL
    original_balcarce_build = balcarce_layer._ORIGINAL_BUILD_OPERATIONAL_DATA
    original_slider = st.slider

    balcarce_layer._ORIGINAL_SIMULATE_DUAL = _simulate_dual_with_san_pedro
    balcarce_layer._ORIGINAL_BUILD_OPERATIONAL_DATA = _build_operational_data_with_san_pedro
    st.slider = _slider_with_frozen_san_pedro
    try:
        base.run()
    finally:
        balcarce_layer._ORIGINAL_SIMULATE_DUAL = original_balcarce_simulate
        balcarce_layer._ORIGINAL_BUILD_OPERATIONAL_DATA = original_balcarce_build
        st.slider = original_slider
