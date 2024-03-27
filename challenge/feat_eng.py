import pandas as pd
from copy import deepcopy
from datetime import datetime

PERIOD_BORDERS = {
    "morning":   "05:00",  # "max": "11:59"
    "afternoon": "12:00",  # "max": "18:59"
    "evening":   "19:00",  # "max": "23:59"
    "night":     "00:00"   # "max": "04:59"
}
PERIOD_BORDERS = {
    k: datetime.strptime(v, '%H:%M').time()
    for k, v in PERIOD_BORDERS.items()
}

HIGH_SEASON_RANGES = {
    "range1": {"min": "15-Dec", "max": "31-Dec"},
    "range2": {"min": "1-Jan",  "max":  "3-Mar"},
    "range3": {"min": "15-Jul", "max": "31-Jul"},
    "range4": {"min": "11-Sep", "max": "30-Sep"},
}
HIGH_SEASON_RANGES = {
    k1: {
        k2: datetime.strptime(v2, '%d-%b')
        for k2, v2 in v1.items()
    }
    for k1, v1 in HIGH_SEASON_RANGES.items()
}


def get_period_day(date) -> str:
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()

    if date_time >= PERIOD_BORDERS["morning"] and date_time < PERIOD_BORDERS["afternoon"]:
        return "mañana"
    elif date_time >= PERIOD_BORDERS["afternoon"] and date_time < PERIOD_BORDERS["evening"]:
        return "tarde"
    else:
        return "noche"


def is_high_season(fecha: str) -> int:
    fecha_anio = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')

    hsr = deepcopy(HIGH_SEASON_RANGES)
    hsr = {
        k1: {
            k2: v2.replace(year=fecha_anio)
            for k2, v2 in v1.items()
        }
        for k1, v1 in hsr.items()
    }

    cond = (
        (fecha >= hsr["range2"]["min"] and fecha <= hsr["range2"]["max"]) or
        (fecha >= hsr["range4"]["min"] and fecha <= hsr["range4"]["max"]) or
        (fecha >= hsr["range1"]["min"] and fecha <= hsr["range1"]["max"]) or
        (fecha >= hsr["range3"]["min"] and fecha <= hsr["range3"]["max"])
    )  # order based on number of days (for efficiency)
    if cond:
        return 1
    else:
        return 0


def get_min_diff(data: pd.DataFrame) -> float:
    fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
    fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
    return ((fecha_o - fecha_i).total_seconds()) / 60.0
