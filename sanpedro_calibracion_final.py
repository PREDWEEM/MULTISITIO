from __future__ import annotations

"""Parámetros y funciones de la versión final PREDWEEM San Pedro 2026.

La parametrización fue calibrada de manera conjunta con las campañas completas
2025–2026 del repositorio geográfico ``PREDWEEM/lolium_sanpedro2026``.
La ANN original no se modifica. La capa local reemplaza la termoinhibición
binaria por una respuesta termohídrica continua y agrega agotamiento causal de
la cohorte germinable.
"""

import numpy as np
import pandas as pd


COBERTURA_FINAL = 17.117654787448153
WMAX_FINAL = 21.835296520312887
T0_TERMOHIDRICO_FINAL = 24.554776252844984
ALPHA_HIDRICA_FINAL = 0.05239929751806027
PENDIENTE_TERMOHIDRICA_FINAL = 0.36599312676568485
K_COHORTE_FINAL = 0.4302565796601996
VENTANA_TERMOHIDRICA_FINAL = 7
CHOQUE_HIDRICO_FINAL = 45.0
EXPONENTE_KR_FINAL = 0.0
ESCALA_LLUVIA_TH_FINAL = 45.0
LATENCIA_JD_FINAL = 25
FIN_CHOQUE_HIDRICO_JD_FINAL = 110
TECHO_CHOQUE_HIDRICO_FINAL = 0.75
UMBRAL_PRIMER_PICO_FINAL = 0.20
CALIBRACION = "San Pedro 2025-2026"
VERSION_MOTOR = "SP-FINAL-2025-2026"
REPOSITORIO_REFERENCIA = "PREDWEEM/lolium_sanpedro2026"


def aplicar_interaccion_termohidrica(
    signal,
    *,
    temperatura_media: pd.Series,
    humedad_relativa,
    precipitacion_3d: pd.Series,
    t0_base: float = T0_TERMOHIDRICO_FINAL,
    alivio_hidrico: float = ALPHA_HIDRICA_FINAL,
    pendiente_c: float = PENDIENTE_TERMOHIDRICA_FINAL,
    ventana_dias: int = VENTANA_TERMOHIDRICA_FINAL,
    escala_lluvia_mm: float = ESCALA_LLUVIA_TH_FINAL,
):
    """Devuelve señal modulada por temperatura × humedad y variables de auditoría.

    H_int = HR * clip(Prec_3d / escala_lluvia, 0, 1)
    Tcrit_eff = T0 + alpha * H_int
    F_TH = 1 / (1 + exp((Tmedia_window - Tcrit_eff) / pendiente))
    """

    base = np.asarray(signal, dtype=float).copy()
    temperatura = pd.to_numeric(temperatura_media, errors="coerce")
    ventana = max(1, int(ventana_dias))
    pendiente = max(float(pendiente_c), 1e-6)
    escala = max(float(escala_lluvia_mm), 1e-6)

    temperatura_ventana = (
        temperatura.rolling(window=ventana, min_periods=1).mean().to_numpy(float)
    )
    hr = np.clip(np.asarray(humedad_relativa, dtype=float), 0.0, 1.0)
    lluvia_relativa = np.clip(
        pd.to_numeric(precipitacion_3d, errors="coerce").fillna(0.0).to_numpy(float)
        / escala,
        0.0,
        1.0,
    )
    indice_hidrico = hr * lluvia_relativa
    tcrit_efectiva = float(t0_base) + float(alivio_hidrico) * indice_hidrico
    z = np.clip(
        (temperatura_ventana - tcrit_efectiva) / pendiente,
        -50.0,
        50.0,
    )
    factor = 1.0 / (1.0 + np.exp(z))
    salida = np.clip(base * factor, 0.0, 1.0)
    diagnostico = factor < 0.50
    return salida, factor, temperatura_ventana, indice_hidrico, tcrit_efectiva, diagnostico


def aplicar_agotamiento_cohorte(
    signal,
    peak_index: int | None,
    *,
    k_cohorte: float = K_COHORTE_FINAL,
):
    """Aplica el reservorio causal de cohorte usado por San Pedro final.

    f_t = 1 - exp(-k * potencial_t)
    flujo_t = reserva_t * f_t
    reserva_(t+1) = reserva_t - flujo_t
    """

    potencial = np.asarray(signal, dtype=float).copy()
    flujo = np.zeros(len(potencial), dtype=float)
    reserva_previa = np.ones(len(potencial), dtype=float)
    fraccion_liberada = np.zeros(len(potencial), dtype=float)
    factor_cohorte = np.ones(len(potencial), dtype=float)

    if peak_index is None or len(potencial) == 0:
        return flujo, reserva_previa, fraccion_liberada, factor_cohorte

    peak = int(peak_index)
    if peak < 0 or peak >= len(potencial):
        raise IndexError("El índice del pico está fuera de la señal.")

    k = max(float(k_cohorte), 0.0)
    reserva = 1.0
    factor_cohorte[:peak] = 0.0

    for index in range(peak, len(potencial)):
        reserva_previa[index] = reserva
        valor = float(np.clip(potencial[index], 0.0, 1.0))
        if valor <= 0.0 or reserva <= 0.0:
            factor_cohorte[index] = 1.0 if valor <= 0.0 else 0.0
            continue

        fraccion = 1.0 - np.exp(-k * valor)
        liberado = reserva * fraccion
        fraccion_liberada[index] = fraccion
        flujo[index] = liberado
        factor_cohorte[index] = liberado / valor if valor > 0.0 else 0.0
        reserva = float(np.clip(reserva - liberado, 0.0, 1.0))

    return np.clip(flujo, 0.0, 1.0), reserva_previa, fraccion_liberada, factor_cohorte
