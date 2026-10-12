"""Вспомогательные функции для обработки данных работы 3.3.4.

Содержит:
  * пути к каталогам работы (data/raw, data/processed, report/images);
  * чтение сырых и запись обработанных CSV;
  * МНК (с свободным членом и через начало координат) с оценкой погрешностей;
  * округление результата по погрешности.

Модуль не зависит от конкретной работы и при желании может быть перенесён
в common/scripts/ репозитория.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Пути. Скрипты лежат в <работа>/scripts/, данные -- в <работа>/data/
# --------------------------------------------------------------------------
LAB_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = LAB_DIR / "data" / "raw"
PROC_DIR = LAB_DIR / "data" / "processed"
FIG_DIR = LAB_DIR / "report" / "images"


# --------------------------------------------------------------------------
# Ввод-вывод
# --------------------------------------------------------------------------
def read_raw(name: str) -> pd.DataFrame:
    """Читает сырой CSV из data/raw/.

    Строки, начинающиеся с '#', считаются комментариями. Полностью пустые
    строки отбрасываются. Сырые данные НИКОГДА не перезаписываются.
    """
    path = RAW_DIR / name
    if not path.exists():
        print(f"  [!] файл {path.relative_to(LAB_DIR)} не найден")
        return pd.DataFrame()
    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    if len(df) == 0:
        return df
    return df.dropna(how="all").reset_index(drop=True)


def write_processed(df: pd.DataFrame, name: str, columns: list[str] | None = None) -> None:
    """Пишет обработанные данные в data/processed/.

    Если данных нет, файл всё равно создаётся -- только с заголовком,
    чтобы отчёт LaTeX собирался и до проведения измерений.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        if df.empty:
            df = pd.DataFrame(columns=columns)
        else:
            df = df.reindex(columns=columns)
    df.to_csv(PROC_DIR / name, index=False)
    print(f"  -> data/processed/{name} ({len(df)} строк)")


def read_params() -> dict[str, float]:
    """Читает data/raw/setup_params.csv в словарь {param: value}."""
    df = read_raw("setup_params.csv")
    if df.empty or "param" not in df.columns:
        return {}
    params: dict[str, float] = {}
    for _, row in df.iterrows():
        try:
            params[str(row["param"]).strip()] = float(row["value"])
        except (TypeError, ValueError):
            continue
    return params


def get_param(params: dict[str, float], name: str, default: float) -> float:
    """Значение параметра установки; если не задано -- рекомендованное методичкой."""
    value = params.get(name)
    if value is None or not np.isfinite(value):
        return default
    return value


# --------------------------------------------------------------------------
# Метод наименьших квадратов
# --------------------------------------------------------------------------
def lsq(x, y):
    """МНК для y = k*x + b.

    Возвращает (k, sigma_k, b, sigma_b); погрешности -- случайные,
    по стандартным формулам для парной линейной регрессии.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return (np.nan,) * 4

    mxy, mx, my = np.mean(x * y), np.mean(x), np.mean(y)
    mxx, myy = np.mean(x * x), np.mean(y * y)
    d = mxx - mx**2
    k = (mxy - mx * my) / d
    b = my - k * mx

    sigma_k = np.sqrt(((myy - my**2) / d - k**2) / (n - 2))
    sigma_b = sigma_k * np.sqrt(mxx)
    return k, sigma_k, b, sigma_b


def lsq_origin(x, y):
    """МНК для y = k*x (прямая через начало координат).

    Возвращает (k, sigma_k).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return np.nan, np.nan

    k = np.sum(x * y) / np.sum(x * x)
    sigma_k = np.sqrt((np.mean(y**2) / np.mean(x**2) - k**2) / (n - 1))
    return k, sigma_k


# --------------------------------------------------------------------------
# Оформление результатов
# --------------------------------------------------------------------------
def fmt(value: float, sigma: float | None = None, unit: str = "") -> str:
    """Строка вида '1.23 +- 0.05 кГц' с разумным числом знаков."""
    if value is None or not np.isfinite(value):
        return "???"
    if sigma is None or not np.isfinite(sigma) or sigma == 0:
        text = f"{value:.4g}"
    else:
        digits = max(0, int(-np.floor(np.log10(abs(sigma)))) + 1)
        text = f"{value:.{digits}f} +- {sigma:.{digits}f}"
        if value != 0:
            text += f"  ({abs(sigma / value) * 100:.1f} %)"
    return f"{text} {unit}".strip()


class Results:
    """Накопитель итоговых чисел -> data/processed/fit_results.csv."""

    COLUMNS = ["quantity", "value", "sigma", "unit", "comment"]

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, quantity: str, value: float, sigma: float = np.nan,
            unit: str = "", comment: str = "") -> None:
        self.rows.append({
            "quantity": quantity,
            "value": None if not np.isfinite(value) else round(float(value), 6),
            "sigma": None if not np.isfinite(sigma) else round(float(sigma), 6),
            "unit": unit,
            "comment": comment,
        })
        print(f"    {quantity} = {fmt(value, sigma, unit)}")

    def save(self) -> None:
        df = pd.DataFrame(self.rows, columns=self.COLUMNS)
        write_processed(df, "fit_results.csv", self.COLUMNS)


# --------------------------------------------------------------------------
# Вывод чисел в LaTeX (десятичная запятая, общий множитель 10^p)
# --------------------------------------------------------------------------
def _tex_decimal(text: str) -> str:
    return text.replace(".", "{,}")


def tex_num(value: float, digits: int = 3) -> str:
    """Число для математического режима LaTeX: '6{,}25\\cdot10^{-3}'."""
    if value is None or not np.isfinite(value):
        return r"\textbf{???}"
    if value == 0:
        return "0"
    p = int(np.floor(np.log10(abs(value))))
    if -2 <= p <= 3:
        decimals = max(0, digits - 1 - p)
        return _tex_decimal(f"{value:.{decimals}f}")
    mant = value / 10**p
    return _tex_decimal(f"{mant:.{digits - 1}f}") + rf"\cdot10^{{{p}}}"


def tex_pm(value: float, sigma: float) -> str:
    """'(6{,}2 \\pm 0{,}3)\\cdot10^{-3}' -- значение с погрешностью.

    Погрешность округляется до 2 значащих цифр, если первая цифра 1 или 2,
    иначе до одной; значение -- до того же разряда.
    """
    if value is None or not np.isfinite(value):
        return r"\textbf{???}"
    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        return tex_num(value)
    p = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
    if -2 <= p <= 3:
        p = 0
    v, s = value / 10**p, sigma / 10**p
    ps = int(np.floor(np.log10(s)))
    lead = int(s / 10**ps)
    decimals = -(ps - 1) if lead < 3 else -ps
    if decimals >= 0:
        body = (_tex_decimal(f"{v:.{decimals}f}") + r" \pm "
                + _tex_decimal(f"{s:.{decimals}f}"))
    else:  # округление до десятков/сотен: 3473 +- 270 -> 3470 +- 270
        q = 10 ** (-decimals)
        body = f"{round(v / q) * q:.0f}" + r" \pm " + f"{round(s / q) * q:.0f}"
    if p == 0:
        return body
    return rf"({body})\cdot10^{{{p}}}"
