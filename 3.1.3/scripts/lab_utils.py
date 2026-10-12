"""Вспомогательные функции для обработки данных лабораторной работы.

Содержит:
  * пути к каталогам работы (data/raw, data/processed, report/img);
  * чтение сырых и запись обработанных CSV;
  * чтение параметров установки (setup_params.csv) вместе с погрешностями;
  * МНК (со свободным членом и через начало координат) с оценкой погрешностей;
  * округление результата по погрешности и вывод итогов:
      - data/processed/fit_results.csv  -- таблица всех итоговых чисел;
      - data/processed/results.tex      -- те же числа в виде макросов LaTeX,
        которые отчёт подставляет сам (\\res{ключ}, \\resv, \\ress, \\rese).

Модуль не зависит от конкретной работы (такой же лежит в соседних работах)
и при желании может быть перенесён в common/scripts/ репозитория.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Пути. Скрипты лежат в <работа>/scripts/, данные -- в <работа>/data/
# --------------------------------------------------------------------------
LAB_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = LAB_DIR / "data" / "raw"
PROC_DIR = LAB_DIR / "data" / "processed"
FIG_DIR = LAB_DIR / "report" / "img"


# --------------------------------------------------------------------------
# Ввод-вывод
# --------------------------------------------------------------------------
def read_raw(name: str) -> pd.DataFrame:
    """Читает сырой CSV из data/raw/.

    Строки, начинающиеся с '#', -- комментарии. Полностью пустые строки
    отбрасываются. Сырые данные НИКОГДА не перезаписываются.
    """
    path = RAW_DIR / name
    if not path.exists():
        print(f"  [!] файл {path.relative_to(LAB_DIR)} не найден")
        return pd.DataFrame()
    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    return df.dropna(how="all").reset_index(drop=True)


def col(df: pd.DataFrame, name: str) -> np.ndarray:
    """Столбец как массив float (NaN, если столбца нет или ячейка пустая)."""
    if name not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def _round_sig(x, sig: int = 4):
    if x is None or not isinstance(x, (int, float, np.floating)) or not np.isfinite(x) or x == 0:
        return x
    return round(float(x), sig - 1 - int(math.floor(math.log10(abs(x)))))


def write_processed(df: pd.DataFrame, name: str, columns: list[str],
                    sig: int = 4) -> None:
    """Пишет обработанные данные в data/processed/.

    Числа округляются до `sig` значащих цифр (для таблиц отчёта).
    Если данных нет, файл всё равно создаётся -- только с заголовком,
    чтобы отчёт LaTeX собирался и до проведения измерений.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(columns=columns) if df.empty else df.reindex(columns=columns)
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: _round_sig(v, sig))
    df.to_csv(PROC_DIR / name, index=False)
    print(f"  -> data/processed/{name} ({len(df)} строк)")


def read_params() -> dict[str, tuple[float, float]]:
    """Читает data/raw/setup_params.csv в словарь {param: (value, sigma)}.

    Пустые значения -> NaN; пустая погрешность -> 0.
    """
    df = read_raw("setup_params.csv")
    if df.empty or "param" not in df.columns:
        return {}
    params: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        value = pd.to_numeric(row.get("value"), errors="coerce")
        sigma = pd.to_numeric(row.get("sigma"), errors="coerce")
        params[str(row["param"]).strip()] = (
            float(value) if pd.notna(value) else np.nan,
            float(sigma) if pd.notna(sigma) else 0.0,
        )
    return params


def get_param(params: dict, name: str) -> tuple[float, float]:
    """(value, sigma) параметра установки; (NaN, 0), если он не заполнен."""
    value, sigma = params.get(name, (np.nan, 0.0))
    if not np.isfinite(value):
        print(f"  [!] параметр {name} не заполнен в setup_params.csv")
    return value, sigma


# --------------------------------------------------------------------------
# Статистика
# --------------------------------------------------------------------------
def mean_with_error(x, sigma_instr: float = 0.0) -> tuple[float, float]:
    """Среднее и полная погрешность: случайная (СКО среднего) + приборная."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    m = float(np.mean(x))
    s_rand = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return m, float(np.hypot(s_rand, sigma_instr))


def lsq(x, y):
    """МНК для y = k*x + b. Возвращает (k, sigma_k, b, sigma_b)."""
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
    sigma_k = np.sqrt(max((myy - my**2) / d - k**2, 0.0) / (n - 2))
    sigma_b = sigma_k * np.sqrt(mxx)
    return k, sigma_k, b, sigma_b


def lsq_origin(x, y):
    """МНК для y = k*x (прямая через начало координат). Возвращает (k, sigma_k)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    k = np.sum(x * y) / np.sum(x * x)
    sigma_k = np.sqrt(max(np.mean(y**2) / np.mean(x**2) - k**2, 0.0) / (n - 1))
    return k, sigma_k


def fit_line_csv(name: str, k: float, b: float, x_min: float, x_max: float) -> None:
    """Две точки прямой y = k x + b -> data/processed/<name> (столбцы x,y).

    Отчёт рисует аппроксимирующую прямую прямо из этого файла, поэтому
    коэффициенты не нужно переписывать в main.tex руками.
    """
    if np.isfinite(k) and np.isfinite(b):
        df = pd.DataFrame({"x": [x_min, x_max], "y": [k * x_min + b, k * x_max + b]})
    else:
        df = pd.DataFrame(columns=["x", "y"])
    write_processed(df, name, ["x", "y"], sig=6)


# --------------------------------------------------------------------------
# Округление и оформление результатов
# --------------------------------------------------------------------------
def round_pm(value: float, sigma: float | None) -> tuple[str, str, int]:
    """Округление по погрешности: 1 значащая цифра (2, если первая -- 1).

    Возвращает (значение, погрешность, число знаков после запятой).
    """
    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        return f"{value:.3g}", "", -1
    exp = int(math.floor(math.log10(sigma)))
    lead = int(sigma / 10**exp)
    digits = -exp + (1 if lead == 1 else 0)
    if digits >= 0:
        return f"{value:.{digits}f}", f"{sigma:.{digits}f}", digits
    q = 10 ** (-digits)
    return f"{round(value / q) * q:.0f}", f"{round(sigma / q) * q:.0f}", digits


def fmt(value: float, sigma: float | None = None, unit: str = "") -> str:
    """Строка для консоли: '1.23 +- 0.05 кГц  (4.1 %)'."""
    if value is None or not np.isfinite(value):
        return "???"
    v, s, _ = round_pm(value, sigma)
    text = v if not s else f"{v} +- {s}"
    if s and value != 0:
        text += f"  ({abs(sigma / value) * 100:.1f} %)"
    return f"{text} {unit}".strip()


def _tex_num(text: str) -> str:
    """'1.23e-05' -> '1{,}23\\cdot10^{-5}', '1.23' -> '1{,}23'."""
    if "e" in text:
        mant, exp = text.split("e")
        return f"{mant.replace('.', '{,}')}\\cdot10^{{{int(exp)}}}"
    return text.replace(".", "{,}")


class Results:
    """Накопитель итоговых чисел.

    save() пишет data/processed/fit_results.csv и data/processed/results.tex.
    В results.tex для каждого ключа определяются макросы, которыми пользуется
    отчёт:  \\res{key} -> «v \\pm s»,  \\resv{key} -> v,  \\ress{key} -> s,
    \\rese{key} -> относительная погрешность в процентах.
    Ключ -- латинские буквы и цифры (без «_»).
    """

    COLUMNS = ["key", "quantity", "value", "sigma", "unit", "comment"]

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, key: str, quantity: str, value: float, sigma: float = np.nan,
            unit: str = "", comment: str = "") -> None:
        ok = value is not None and np.isfinite(value)
        self.rows.append({
            "key": key,
            "quantity": quantity,
            "value": _round_sig(float(value), 6) if ok else None,
            "sigma": _round_sig(float(sigma), 6) if ok and np.isfinite(sigma) else None,
            "unit": unit,
            "comment": comment,
        })
        print(f"    {quantity} = {fmt(value, sigma, unit)}")

    def save(self) -> None:
        df = pd.DataFrame(self.rows, columns=self.COLUMNS)
        write_processed(df, "fit_results.csv", self.COLUMNS, sig=6)

        lines = ["% Сгенерировано scripts/process_data.py -- вручную не править.",
                 "% Незаполненные величины не определены и в отчёте выводятся как ???."]
        for r in self.rows:
            if r["value"] is None:
                continue
            value = float(r["value"])
            sigma = float(r["sigma"]) if r["sigma"] is not None else np.nan
            v, s, _ = round_pm(value, sigma)
            key = r["key"]
            lines.append(rf"\expandafter\def\csname res@{key}@v\endcsname{{{_tex_num(v)}}}")
            if s:
                eps = abs(sigma / value) * 100 if value != 0 else np.nan
                eps_txt = f"{eps:.1g}" if eps < 1 else f"{eps:.0f}" if eps >= 10 else f"{eps:.1f}"
                lines.append(rf"\expandafter\def\csname res@{key}@s\endcsname{{{_tex_num(s)}}}")
                lines.append(rf"\expandafter\def\csname res@{key}@e\endcsname{{{_tex_num(eps_txt)}}}")
                lines.append(rf"\expandafter\def\csname res@{key}\endcsname"
                             rf"{{{_tex_num(v)} \pm {_tex_num(s)}}}")
            else:
                lines.append(rf"\expandafter\def\csname res@{key}\endcsname{{{_tex_num(v)}}}")
        (PROC_DIR / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("  -> data/processed/results.tex")


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
