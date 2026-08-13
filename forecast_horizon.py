from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import update_meteo_core as core

MIN_FORECAST_DAYS = 7
FILL_SOURCE = "MULTISITIO_OPEN_METEO_ECMWF_IFS_FORECAST_FILL"
FILL_QUALITY = "Pronostico_relleno_horizonte_7d"


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    index = {str(c).upper().strip(): str(c) for c in frame.columns}
    for name in names:
        found = index.get(name.upper())
        if found is not None:
            return found
    return None


def _required(frame: pd.DataFrame, site_name: str):
    date_col = _column(frame, "Fecha", "Date", "Datetime")
    tmax_col = _column(frame, "TMAX")
    tmin_col = _column(frame, "TMIN")
    prec_col = _column(frame, "Prec", "Precipitacion", "Precipitación", "Lluvia")
    missing = [name for name, col in (("Fecha", date_col), ("TMAX", tmax_col), ("TMIN", tmin_col), ("Prec", prec_col)) if col is None]
    if missing:
        raise RuntimeError(f"{site_name}: faltan columnas: {', '.join(missing)}")
    return date_col, tmax_col, tmin_col, prec_col


def target_dates(site) -> pd.DatetimeIndex:
    today = pd.Timestamp(datetime.now(ZoneInfo(site.timezone)).date())
    return pd.date_range(today, periods=MIN_FORECAST_DAYS, freq="D")


def _set_metadata(frame: pd.DataFrame, emission: str) -> None:
    for candidates, value in (
        (("Fuente", "Source"), FILL_SOURCE),
        (("TipoDato", "Tipo", "Type"), "Pronostico"),
        (("CalidadDato", "Calidad", "Quality"), FILL_QUALITY),
        (("Emision", "Emision_UTC", "Fecha_Emision", "FECHA_EMISION"), emission),
        (("Fuente_TMAX",), FILL_SOURCE),
        (("Fuente_TMIN",), FILL_SOURCE),
        (("Fuente_Prec",), FILL_SOURCE),
    ):
        col = _column(frame, *candidates)
        if col is not None:
            frame[col] = value


def complete_site(site) -> dict:
    path = site.meteo_path(".")
    data = pd.read_csv(path)
    date_col, tmax_col, tmin_col, prec_col = _required(data, site.nombre)
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce").dt.normalize()
    data = data.dropna(subset=[date_col]).copy()

    target = target_dates(site)
    existing = pd.DatetimeIndex(data[date_col].unique())
    missing = target.difference(existing)
    original_sha = core.sha256_bytes(path.read_bytes())
    added: list[str] = []

    if len(missing):
        emission = datetime.now(ZoneInfo(site.timezone)).isoformat(timespec="seconds")
        fallback = core.fetch_open_meteo_forecast(site, emission)
        fallback["Fecha"] = pd.to_datetime(fallback["Fecha"], errors="coerce").dt.normalize()
        fallback = fallback.loc[fallback["Fecha"].isin(missing)].drop_duplicates("Fecha", keep="last")
        still_missing = missing.difference(pd.DatetimeIndex(fallback["Fecha"].unique()))
        if len(still_missing):
            raise RuntimeError(
                f"{site.nombre}: ECMWF no completó el horizonte: "
                + ", ".join(day.strftime("%Y-%m-%d") for day in still_missing)
            )

        additions = pd.DataFrame(index=range(len(fallback)), columns=data.columns)
        additions[date_col] = fallback["Fecha"].to_numpy()
        additions[tmax_col] = fallback["TMAX"].to_numpy()
        additions[tmin_col] = fallback["TMIN"].to_numpy()
        additions[prec_col] = fallback["Prec"].to_numpy()
        tmean_col = _column(additions, "TMEDIA", "TMEAN")
        if tmean_col is not None:
            additions[tmean_col] = (
                pd.to_numeric(additions[tmax_col], errors="coerce")
                + pd.to_numeric(additions[tmin_col], errors="coerce")
            ) / 2.0
        _set_metadata(additions, emission)

        data = pd.concat([data, additions], ignore_index=True)
        data = data.sort_values(date_col).drop_duplicates(date_col, keep="last").reset_index(drop=True)
        core.atomic_csv(data, path)
        added = [day.strftime("%Y-%m-%d") for day in missing]

    final = pd.read_csv(path)
    fdate, ftmax, ftmin, fprec = _required(final, site.nombre)
    final[fdate] = pd.to_datetime(final[fdate], errors="coerce").dt.normalize()
    horizon = final.loc[final[fdate].isin(target)].copy()
    available = pd.DatetimeIndex(horizon[fdate].dropna().unique())
    missing_final = target.difference(available)
    if len(missing_final):
        raise RuntimeError(f"{site.nombre}: horizonte final inferior a 7 días.")

    for col in (ftmax, ftmin, fprec):
        if pd.to_numeric(horizon[col], errors="coerce").isna().any():
            raise RuntimeError(f"{site.nombre}: nulos en {col} dentro del horizonte.")
    if (pd.to_numeric(horizon[ftmax], errors="coerce") < pd.to_numeric(horizon[ftmin], errors="coerce")).any():
        raise RuntimeError(f"{site.nombre}: TMAX < TMIN dentro del horizonte.")
    if (pd.to_numeric(horizon[fprec], errors="coerce") < 0).any():
        raise RuntimeError(f"{site.nombre}: precipitación negativa dentro del horizonte.")

    return {
        "horizonte_validado": True,
        "horizonte_minimo_dias": MIN_FORECAST_DAYS,
        "horizonte_desde": target[0].date().isoformat(),
        "horizonte_hasta": target[-1].date().isoformat(),
        "dias_disponibles_horizonte": int(len(available)),
        "dias_agregados_multisitio": added,
        "relleno_ecmwf_aplicado": bool(added),
        "fuente_relleno": FILL_SOURCE if added else None,
        "sha256_origen": original_sha,
        "sha256_final": core.sha256_bytes(path.read_bytes()),
    }


def ensure_all_sites(state_path=None) -> None:
    state_path = state_path or core.STATE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    site_states = state.setdefault("sitios", {})

    for site in core.ordered_sites():
        result = complete_site(site)
        current = site_states.setdefault(site.slug, {})
        current.update(result)
        if result["relleno_ecmwf_aplicado"]:
            current["copia_exacta"] = False
            current["modo"] = "copia_origen_mas_relleno_ecmwf_7d"
            current["sha256"] = result["sha256_final"]
            current["bytes"] = site.meteo_path(".").stat().st_size
        print(
            f"{site.nombre}: {result['horizonte_desde']} → {result['horizonte_hasta']} "
            f"· relleno={result['relleno_ecmwf_aplicado']}"
        )

    state["horizonte_pronostico"] = {
        "politica": "7_fechas_consecutivas_desde_hoy_inclusive",
        "dias": MIN_FORECAST_DAYS,
        "relleno_seguridad": FILL_SOURCE,
    }
    core.atomic_json(state, state_path)
