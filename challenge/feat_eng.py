from typing import TYPE_CHECKING
from datetime import datetime as dt
if TYPE_CHECKING:
    from pandas import DataFrame


def get_min_diff(data: "DataFrame") -> float:
    fecha_o = dt.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
    fecha_i = dt.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
    return ((fecha_o - fecha_i).total_seconds()) / 60.0
