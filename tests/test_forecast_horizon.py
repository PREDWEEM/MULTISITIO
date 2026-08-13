from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from sitios_lolium import ordered_sites


def test_all_sites_have_seven_consecutive_days_from_today():
    for site in ordered_sites():
        data = pd.read_csv(site.meteo_path('.'))
        dates = pd.to_datetime(data['Fecha'], errors='coerce').dt.normalize()
        today = pd.Timestamp(datetime.now(ZoneInfo(site.timezone)).date())
        expected = pd.date_range(today, periods=7, freq='D')
        missing = expected.difference(pd.DatetimeIndex(dates.dropna().unique()))
        assert len(missing) == 0, (
            f"{site.nombre}: faltan fechas en el horizonte de 7 días: "
            + ', '.join(day.strftime('%Y-%m-%d') for day in missing)
        )
