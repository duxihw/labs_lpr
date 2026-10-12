"""Построение графиков лабораторной работы 3.6.1 в «латеховском» стиле.

Читает data/processed/*.csv (их готовит process_data.py) и сохраняет
векторные графики в report/img/.

Запуск (из каталога работы):
    python scripts/plot_data.py

Графики оформлены по требованиям к отчёту: подписанные оси с размерностями,
ноль на осях, кресты погрешностей, теоретические кривые тонкой линией,
легенда с параметрами аппроксимации.

В самом отчёте (report/main.tex) те же зависимости строятся средствами
pgfplots прямо из CSV -- так график автоматически обновляется при сборке.
Графики из этого скрипта удобны для лабораторного журнала и для быстрой
проверки данных прямо во время работы, без пересборки LaTeX.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lab_utils import FIG_DIR, PROC_DIR  # noqa: E402

# --------------------------------------------------------------------------
# Стиль «как в LaTeX»
# --------------------------------------------------------------------------
# Если в системе установлен LaTeX с поддержкой кириллицы, можно включить
# честный рендеринг подписей средствами LaTeX: USE_TEX = True.
USE_TEX = False

# Шрифт подписей. CMU Serif (Computer Modern Unicode) выглядит в точности как
# шрифт LaTeX и содержит кириллицу; если он не установлен, используется
# DejaVu Serif (есть всегда). Математика рисуется тем же шрифтом, иначе
# кириллица в строках с формулами заменяется «квадратиками».
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
SERIF = "CMU Serif" if "CMU Serif" in _available else "DejaVu Serif"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [SERIF],
    "mathtext.fontset": "custom",
    "mathtext.rm": SERIF,
    "mathtext.it": f"{SERIF}:italic",
    "mathtext.bf": f"{SERIF}:bold",
    "mathtext.cal": SERIF,
    "mathtext.sf": SERIF,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth": 1.0,
    "lines.markersize": 4.5,
    "figure.figsize": (6.3, 4.0),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

if USE_TEX:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T2A]{fontenc}"
                               r"\usepackage[utf8]{inputenc}"
                               r"\usepackage[russian]{babel}",
    })

MARK = dict(fmt="o", capsize=3, elinewidth=0.8, markeredgewidth=0.8,
            color="#1f4e79", label="Эксперимент")


def load(name: str) -> pd.DataFrame:
    path = PROC_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep=";").dropna(how="all")


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"  -> report/img/{name}")


def result(name: str) -> float:
    """Число из fit_results.csv по началу названия величины."""
    df = load("fit_results.csv")
    row = df[df["quantity"].str.startswith(name)]
    return float(row["value"].iloc[0])


# --------------------------------------------------------------------------
def plot_harmonics() -> None:
    """Относительные амплитуды гармоник (T = 1 мс, tau = 50 мкс) и огибающая sinc."""
    df = load("harmonics_processed.csv")
    fig, ax = plt.subplots()
    ax.plot(df["nu_exp_kHz"], df["ratio_exp"], "o", color="#1f4e79", label="Эксперимент")
    nu = np.linspace(0.01, 21, 500)
    env = np.abs(np.sinc(nu * 0.05)) / np.sinc(0.05)
    ax.plot(nu, env, color="crimson", label=r"$|\mathrm{sinc}(\pi\nu\tau)|$, $\tau=50$ мкс")
    ax.set_xlabel(r"$\nu$, кГц")
    ax.set_ylabel(r"$|a_n/a_1|$")
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 1.05)
    ax.legend()
    save(fig, "fig_harmonics.pdf")


def plot_meander() -> None:
    df = load("meander_processed.csv")
    fig, ax = plt.subplots()
    for nu1, mk in [(50, "o"), (100, "s")]:
        d = df[df["nu1_kHz"] == nu1]
        ax.plot(d["inv_n"], d["ratio"], mk, mfc="none", label=rf"$\nu_1={nu1}$ кГц")
    ax.plot([0, 1.05], [0, 1.05], "k--", lw=0.8, label=r"теория: $1/n$")
    ax.set_xlabel(r"$1/n$")
    ax.set_ylabel(r"$|a_n/a_1|$")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    save(fig, "fig_meander.pdf")


def plot_width() -> None:
    df = load("spectrum_width_processed.csv")
    k = result("k_Dnu (взвеш")
    fig, ax = plt.subplots()
    ax.errorbar(df["x"], df["y"], yerr=df["ey"], **MARK)
    xx = np.linspace(0, 55, 10)
    ax.plot(xx, k * xx, color="crimson", label=rf"МНК: $k={k:.3f}$")
    ax.set_xlabel(r"$1/\tau$, кГц")
    ax.set_ylabel(r"$\Delta\nu$, кГц")
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 55)
    ax.legend()
    save(fig, "fig_width.pdf")


def plot_spacing() -> None:
    df = load("harmonics_spacing_processed.csv")
    df = df[df["use"] == 1]
    k = result("k_dnu (МНК")
    fig, ax = plt.subplots()
    ax.plot(df["x"], df["y"], "o", color="#1f4e79", label="Эксперимент (без $T=2\\tau$)")
    xx = np.linspace(0, 1.1, 10)
    ax.plot(xx, k * xx, color="crimson", label=rf"МНК: $k={k:.4f}$")
    ax.set_xlabel(r"$1/T$, кГц")
    ax.set_ylabel(r"$\delta\nu$, кГц")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.legend()
    save(fig, "fig_spacing.pdf")


def plot_bursts() -> None:
    df = load("bursts_width_points.csv")
    k = result("цуги: k (Dnu")
    fig, ax = plt.subplots()
    ax.errorbar(df["x"], df["y"], yerr=df["ey"], **MARK)
    xx = np.linspace(0, 11, 10)
    ax.plot(xx, k * xx, color="crimson", label=rf"МНК: $k={k:.3f}$")
    ax.set_xlabel(r"$1/\tau = \nu_0/N$, кГц")
    ax.set_ylabel(r"$\Delta\nu$, кГц")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.legend()
    save(fig, "fig_bursts.pdf")


def plot_filter() -> None:
    df = load("filter_processed.csv")
    rc = result("tau_RC")
    nu1 = result("nu_1 (спектр)")
    fig, ax = plt.subplots()
    ax.plot(df["nu_kHz"], df["K_exp"], "o", color="#1f4e79", label="Эксперимент")
    xx = np.linspace(nu1, 3600, 300)
    th = np.sqrt(1 + (2 * np.pi * nu1 * rc / 1000) ** 2) / np.sqrt(1 + (2 * np.pi * xx * rc / 1000) ** 2)
    ax.plot(xx, th, color="crimson", label=rf"теория, $\tau_{{RC}}={rc:.2f}$ мкс")
    ax.set_xlabel(r"$\nu$, кГц")
    ax.set_ylabel(r"$K_n/K_1$")
    ax.set_xlim(0, 3600)
    ax.set_ylim(0, 1.1)
    ax.legend()
    save(fig, "fig_filter.pdf")


def main() -> None:
    print("Построение графиков в report/img/:")
    plot_harmonics()
    plot_meander()
    plot_width()
    plot_spacing()
    plot_bursts()
    plot_filter()
    print("Готово.")


if __name__ == "__main__":
    main()
