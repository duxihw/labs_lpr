"""Снятие отсчётов со скриншотов PicoScope (report/img/*.png) -> data/raw/screens/.

Скриншоты спектров -- первичный результат измерений этой работы. Скрипт
переводит изображение в числа так же, как это делается курсорами
осциллографа: по калибровке осей (положение линий сетки и подписи к ним)
находит частоты и амплитуды спектральных линий.

Алгоритм
--------
* Линии сетки имеют цвет (173, 216, 230); по ним и по подписям осей
  (занесены вручную в словарь SCREENS ниже) строится линейная калибровка
  «пиксель -> кГц, мВ».
* Кривая спектра рисуется синим и чёрным; для каждого столбца пикселей
  берётся самая верхняя точка кривой (пунктирные горизонтальные курсоры
  исключаются).
* Амплитуда линии -- максимум кривой в окне вокруг ожидаемой частоты,
  частота -- середина столбцов, в которых кривая поднимается не ниже чем
  на 1 пиксель от максимума.
* Если вершина линии упирается в верхнюю границу экрана, линия помечается
  clipped = 1 (амплитуда известна лишь как нижняя граница).

Погрешность отсчёта -- 1 пиксель; её значение по каждой оси записывается
в столбцы sigma_*.

Сырые данные не перезаписываются: если выходной файл уже существует,
скрипт его пропускает (перезапись -- только с ключом --force).

Запуск (из каталога работы):
    python scripts/digitize_screens.py [--force]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from lab_utils import FIG_DIR, RAW_DIR

OUT_DIR = RAW_DIR / "screens"

GRID_RGB = (173, 216, 230)

# ---------------------------------------------------------------------------
# Калибровка осей: (x_px0, x0, x_px1, x1), (y_px0, y0, y_px1, y1)
# x_px/y_px -- положения крайних линий сетки, x0, y0 ... -- подписи к ним.
# ---------------------------------------------------------------------------
RECT_X = (45, 0.0, 1000, 6.0)            # кГц
RECT_Y = (584, 0.0, 85, 20.1)           # мВ
SCREENS = {
    "spectrum_rect_nu1.png":  ((45, 0.0, 1048, 3.5), (584, 0.0, 85, 4.705)),
    "spectrum_rect_nu2.png":  (RECT_X, RECT_Y),
    "spectrum_rect_nu3.png":  (RECT_X, RECT_Y),
    "spectrum_rect_nu4.png":  (RECT_X, RECT_Y),
    "spectrum_rect_tau100.png": (RECT_X, RECT_Y),
    "spectrum_rect_tau200.png": (RECT_X, RECT_Y),
    "spectrum_rect_tau250.png": (RECT_X, RECT_Y),
    "spectrum_burst_T1_N5.png":  ((170, 30.0, 1033, 80.0), (650, 0.0, 85, 13.42)),
    "spectrum_burst_T2_N5.png":  ((170, 30.0, 1033, 80.0), (650, 0.0, 85, 13.42)),
    "spectrum_burst_T1_N10.png": ((170, 30.0, 1033, 80.0), (650, 0.0, 85, 13.42)),
    "spectrum_burst_nu100_N10.png": ((170, 30.0, 1033, 80.0), (650, 0.0, 85, 13.42)),
    "spectrum_am.png":        ((36, 25.0, 1110, 75.0), (650, 19.17, 85, 843.6)),
    "spectrum_am_wide_1.png": ((45, 0.0, 985, 700.0), (650, 0.0, 85, 50.0)),
    "spectrum_am_wide_2.png": ((45, 0.0, 985, 700.0), (650, 0.0, 85, 50.0)),
    "spectrum_am_wide_3.png": ((45, 0.0, 985, 700.0), (650, 0.0, 85, 50.0)),
    "filter_spectrum.png":    ((45, 0.0, 1062, 3.5), (650, 0.0, 85, 53.68)),   # МГц
}

# Параметры сигналов прямоугольных импульсов на скриншотах части А
# (период T в мс, длительность tau в мкс). Период проверяется по спектру.
RECT_SCREENS = {
    "spectrum_rect_nu1.png":    (1.0, 50.0),
    "spectrum_rect_nu2.png":    (0.5, 50.0),
    "spectrum_rect_nu3.png":    (1.0 / 3.0, 50.0),
    "spectrum_rect_nu4.png":    (0.2, 50.0),
    "spectrum_rect_tau100.png": (0.5, 100.0),
    "spectrum_rect_tau200.png": (0.5, 200.0),
    "spectrum_rect_tau250.png": (0.5, 250.0),
}


class Screen:
    """Оцифрованная кривая одного скриншота спектра."""

    def __init__(self, name: str) -> None:
        (xp0, x0, xp1, x1), (yp0, y0, yp1, y1) = SCREENS[name]
        self.name = name
        self.kx = (x1 - x0) / (xp1 - xp0)
        self.ky = (y1 - y0) / (yp1 - yp0)
        self.x0, self.xp0, self.y0, self.yp0 = x0, xp0, y0, yp0
        self.y_top = y1                       # верхняя граница экрана

        rgb = np.asarray(Image.open(FIG_DIR / name).convert("RGB")).astype(int)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        blue = (r < 60) & (g < 60) & (b > 200)
        black = (r < 60) & (g < 60) & (b < 60)
        mask = blue | black
        for i in range(mask.shape[0]):        # пунктирные курсоры
            if black[i].sum() > 250:
                mask[i] = False

        top, bottom = min(yp0, yp1), max(yp0, yp1)
        cols, vals = [], []
        # левый край области графика (слева от него -- подписи оси Y)
        left = 40 if name == "spectrum_am.png" else 48
        for j in range(left, rgb.shape[1] - 30):
            idx = np.nonzero(mask[top:bottom + 1, j])[0]
            if len(idx):
                cols.append(j)
                vals.append(top + idx[0])
        self.cols = np.array(cols)
        self.rows = np.array(vals)
        self.x = x0 + (self.cols - xp0) * self.kx
        self.y = y0 + (self.rows - yp0) * self.ky

    @property
    def sigma_x(self) -> float:
        return abs(self.kx)

    @property
    def sigma_y(self) -> float:
        return abs(self.ky)

    def line(self, nu: float, half_window: float) -> tuple[float, float, int]:
        """Частота, амплитуда и признак «зашкаливания» линии около nu."""
        m = (self.x > nu - half_window) & (self.x < nu + half_window)
        if not m.any():
            return np.nan, np.nan, 0
        y = self.y[m]
        x = self.x[m]
        ymax = y.max()
        top = np.abs(y - ymax) <= self.sigma_y * 1.01
        clipped = int(ymax >= self.y_top - 0.5 * self.sigma_y)
        return float(np.mean(x[top])), float(ymax), clipped

    def peaks(self, threshold: float, min_sep: float) -> list[tuple[float, float]]:
        """Все локальные максимумы выше threshold, разнесённые на min_sep."""
        out: list[tuple[float, float]] = []
        for k in np.argsort(-self.y):
            if self.y[k] < threshold:
                break
            if all(abs(self.x[k] - p[0]) > min_sep for p in out):
                out.append((float(self.x[k]), float(self.y[k])))
        return sorted(out)


# ---------------------------------------------------------------------------
def rect_lines() -> pd.DataFrame:
    rows = []
    for name, (T, tau) in RECT_SCREENS.items():
        s = Screen(name)
        nu_rep = 1.0 / T
        n = 1
        while n * nu_rep < s.x.max() - 0.1:
            nu, a, clip = s.line(n * nu_rep, 0.15)
            rows.append({
                "screen": name, "T_ms": round(T, 5), "tau_us": tau, "n": n,
                "nu_kHz": round(nu, 4), "a_mV": round(a, 3), "clipped": clip,
                "sigma_nu_kHz": round(s.sigma_x, 4), "sigma_a_mV": round(s.sigma_y, 3),
            })
            n += 1
    return pd.DataFrame(rows)


def burst_lines() -> pd.DataFrame:
    params = {  # nu0, T, N -- по надписям на генераторе (N для 100 кГц см. отчёт)
        "spectrum_burst_T1_N5.png": (50.0, 1.0, 5),
        "spectrum_burst_T2_N5.png": (50.0, 2.0, 5),
        "spectrum_burst_T1_N10.png": (50.0, 1.0, 10),
        "spectrum_burst_nu100_N10.png": (100.0, 1.0, None),
    }
    rows = []
    for name, (nu0, T, N) in params.items():
        s = Screen(name)
        step = 1.0 / T
        for nu in np.arange(23.0, 83.0 + 1e-9, step):
            x, a, clip = s.line(nu, 0.3 * step)
            rows.append({
                "screen": name, "nu0_set_kHz": nu0, "T_ms": T,
                "N_set": "" if N is None else N,
                "nu_grid_kHz": round(nu, 3), "nu_kHz": round(x, 3),
                "a_mV": round(a, 3), "sigma_nu_kHz": round(s.sigma_x, 3),
                "sigma_a_mV": round(s.sigma_y, 3),
            })
    return pd.DataFrame(rows)


def am_lines() -> pd.DataFrame:
    s = Screen("spectrum_am.png")
    rows = []
    for label, nu in [("left", 48.0), ("main", 50.0), ("right", 52.0)]:
        x, a, clip = s.line(nu, 0.8)
        rows.append({"line": label, "nu_kHz": round(x, 3), "a_mV": round(a, 1),
                     "sigma_nu_kHz": round(s.sigma_x, 3),
                     "sigma_a_mV": round(s.sigma_y, 2)})
    return pd.DataFrame(rows)


def meander_lines() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Спектры в диапазоне 0-780 кГц: гармоники и «лишние» линии."""
    screens = {"spectrum_am_wide_1.png": 50.0, "spectrum_am_wide_2.png": 50.0,
               "spectrum_am_wide_3.png": 100.0}
    harm, other = [], []
    for name, nu1 in screens.items():
        s = Screen(name)
        used = []
        n = 1
        while n * nu1 < s.x.max() - 3:
            x, a, clip = s.line(n * nu1, 4.0)
            harm.append({"screen": name, "nu1_set_kHz": nu1, "n": n,
                         "nu_kHz": round(x, 2), "a_mV": round(a, 3),
                         "sigma_nu_kHz": round(s.sigma_x, 3),
                         "sigma_a_mV": round(s.sigma_y, 3)})
            used.append(n * nu1)
            n += 1
        for x, a in s.peaks(0.9, 4.0):
            if all(abs(x - u) > 4 for u in used):
                other.append({"screen": name, "nu1_set_kHz": nu1,
                              "nu_kHz": round(x, 2), "a_mV": round(a, 3),
                              "sigma_nu_kHz": round(s.sigma_x, 3),
                              "sigma_a_mV": round(s.sigma_y, 3)})
    return pd.DataFrame(harm), pd.DataFrame(other)


def filter_lines() -> pd.DataFrame:
    s = Screen("filter_spectrum.png")
    rows = []
    for x, a in s.peaks(1.0, 0.1):
        rows.append({"nu_MHz": round(x, 4), "a_mV": round(a, 2),
                     "sigma_nu_MHz": round(s.sigma_x, 4),
                     "sigma_a_mV": round(s.sigma_y, 2)})
    df = pd.DataFrame(rows)
    df.insert(0, "n", np.arange(1, len(df) + 1))
    return df


def filter_waveform() -> pd.DataFrame:
    """Осциллограмма на выходе RC-цепочки (ось времени в мкс, AC-вход)."""
    rgb = np.asarray(Image.open(FIG_DIR / "filter_signal.png").convert("RGB")).astype(int)
    blue = (rgb[..., 0] < 90) & (rgb[..., 1] < 90) & (rgb[..., 2] > 180)
    # сетка: столбцы 35 ... 1262 <-> -3.805 ... 6.195 мкс; строки 5 ... 682 <-> 100 ... -100 мВ
    t, u = [], []
    for j in range(36, 1265):
        idx = np.nonzero(blue[6:682, j])[0]
        if len(idx):
            t.append(-3.805 + (j - 35) / (1262 - 35) * 10.0)
            u.append(100.0 - (6 + idx.mean() - 5) / (682 - 5) * 200.0)
    return pd.DataFrame({"t_us": np.round(t, 4), "U_mV": np.round(u, 2)})


def main() -> None:
    force = "--force" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meander, other = meander_lines()
    tables = {
        "rect_lines.csv": rect_lines(),
        "burst_lines.csv": burst_lines(),
        "am_lines.csv": am_lines(),
        "meander_lines.csv": meander,
        "meander_other_lines.csv": other,
        "filter_lines.csv": filter_lines(),
        "filter_waveform.csv": filter_waveform(),
    }
    for name, df in tables.items():
        path = OUT_DIR / name
        if path.exists() and not force:
            print(f"  [=] {path.relative_to(RAW_DIR.parent.parent)} уже есть, пропуск")
            continue
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# Отсчёты со скриншотов report/img/ (scripts/digitize_screens.py)\n")
            df.to_csv(fh, index=False)
        print(f"  -> {path.relative_to(RAW_DIR.parent.parent)} ({len(df)} строк)")


if __name__ == "__main__":
    main()
