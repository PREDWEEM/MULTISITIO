from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app_sanpedro_final import _simulate_san_pedro_final
from sanpedro_calibracion_final import (
    COBERTURA_FINAL,
    K_COHORTE_FINAL,
    T0_TERMOHIDRICO_FINAL,
    VERSION_MOTOR,
    WMAX_FINAL,
    aplicar_agotamiento_cohorte,
    aplicar_interaccion_termohidrica,
)


class ConstantANN:
    def __init__(self, value: float = 0.8):
        self.value = float(value)

    def predict(self, values):
        return np.full(len(values), self.value, dtype=float)


def _config():
    return SimpleNamespace(
        latitud=-33.7328,
        pendiente_hidrica=10.0,
        p50_hidrico=0.30,
        corte_hidrico=0.20,
        t_base_c=2.0,
        t_optima_c=20.0,
        t_critica_c=30.0,
        nombre_sitio="San Pedro (Buenos Aires)",
    )


def _weather(periods: int = 80):
    return pd.DataFrame(
        {
            "Fecha": pd.date_range("2026-01-01", periods=periods, freq="D"),
            "TMAX": np.full(periods, 18.0),
            "TMIN": np.full(periods, 8.0),
            "Prec": np.full(periods, 30.0),
        }
    )


def test_termohydric_response_is_continuous_not_binary():
    signal = np.ones(5, dtype=float)
    output, factor, *_ = aplicar_interaccion_termohidrica(
        signal,
        temperatura_media=pd.Series(np.full(5, 25.0)),
        humedad_relativa=np.ones(5),
        precipitacion_3d=pd.Series(np.full(5, 45.0)),
    )

    assert np.all((factor > 0.0) & (factor < 1.0))
    assert np.all(output > 0.0)
    assert np.all(output < signal)
    assert T0_TERMOHIDRICO_FINAL > 24.0


def test_causal_cohort_exhaustion_reduces_the_remaining_reserve():
    signal = np.array([0.0, 0.8, 0.8, 0.8, 0.8], dtype=float)
    flow, reserve, released, factor = aplicar_agotamiento_cohorte(
        signal,
        peak_index=1,
    )

    assert K_COHORTE_FINAL == pytest.approx(0.4302565796601996)
    assert flow[1] > flow[2] > flow[3] > flow[4] > 0.0
    assert reserve[1] == pytest.approx(1.0)
    assert reserve[-1] < reserve[2] < reserve[1]
    assert (released[1:] > 0.0).all()
    assert (factor[1:] < 1.0).all()


def test_multisite_san_pedro_uses_frozen_final_parameters():
    result = _simulate_san_pedro_final(
        _weather(),
        ConstantANN(),
        coverage_percent=90.0,
        wmax=18.81,
        lag_days=0,
        kr_exponent=1.0,
        config=_config(),
    )
    data = result.data

    assert VERSION_MOTOR == "SP-FINAL-2025-2026"
    assert data["Cobertura_Rastrojo"].iat[0] == pytest.approx(COBERTURA_FINAL)
    assert data["Exponente_Kr"].iat[0] == 0.0
    assert data["W_superficial"].max() <= WMAX_FINAL + 1e-12
    assert data["Termohidria_Continua_Activa"].all()
    assert data["Agotamiento_Cohorte_Causal_Activo"].all()
    assert data["K_Cohorte_Causal"].iat[0] == pytest.approx(K_COHORTE_FINAL)
    assert result.first_peak_no_lag == pd.Timestamp("2026-01-26")
    assert data.loc[25, "EMERREL_SIN_LAG"] > 0.20
    assert data.loc[30, "Reserva_Cohorte_SIN_LAG"] < data.loc[25, "Reserva_Cohorte_SIN_LAG"]
