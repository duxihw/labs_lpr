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

from lab_utils import FIG_DIR, PROC_DIR, get_param, lsq, lsq_origin, read_params  # noqa: E402

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
    df = pd.read_csv(path)
    return df.dropna(how="all")


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"  -> report/img/{name}")


def skip(name: str) -> None:
    print(f"  [-] {name}: нет данных, график не построен")


# --------------------------------------------------------------------------
def plot_harmonics(params: dict) -> None:
    """Относительные амплитуды гармоник и теоретическая огибающая sinc."""
    df = load("harmonics_processed.csv")
    if df.empty:
        return skip("гармоники")

    T_ms = get_param(params, "T_rect", 1.0)
    tau_us = get_param(params, "tau_rect", 50.0)
    tau_ms = tau_us / 1000.0

    fig, ax = plt.subplots()
    nu = df["nu_exp_kHz"].to_numpy(dtype=float)
    ax.errorbar(nu, df["ratio_exp"], yerr=df.get("sigma_ratio_exp"), **MARK)

    # огибающая |sinc(pi*nu*tau)| / |sinc(pi*tau/T)|, нормированная на первую гармонику
    nu_grid = np.linspace(0, np.nanmax(nu) * 1.05, 600)
    q = np.pi * tau_ms
    norm = abs(np.sin(np.pi * tau_ms / T_ms) / (np.pi * tau_ms / T_ms))
    with np.errstate(invalid="ignore", divide="ignore"):
        env = np.abs(np.sin(q * nu_grid) / (q * nu_grid)) / norm
    env[0] = 1 / norm
    ax.plot(nu_grid, env, color="crimson", lw=0.9,
            label=r"теория: $|\mathrm{sinc}(\pi\nu\tau)|$")

    ax.set_xlabel(r"$\nu$, кГц")
    ax.set_ylabel(r"$|a_n/a_1|$, отн. ед.")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set_title(rf"Спектр прямоугольных импульсов, $T={T_ms:g}$ мс, $\tau={tau_us:g}$ мкс")
    save(fig, "fig_harmonics.pdf")


def plot_line(name: str, xcol: str, ycol: str, sxcol: str, sycol: str,
              xlabel: str, ylabel: str, title: str, outfile: str,
              theory_k: float | None = None, theory_label: str = "") -> None:
    """Общий график линейной зависимости с МНК-аппроксимацией."""
    df = load(name)
    if df.empty:
        return skip(outfile)

    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)

    fig, ax = plt.subplots()
    ax.errorbar(x, y, xerr=df.get(sxcol), yerr=df.get(sycol), **MARK)

    k0, sk0 = lsq_origin(x, y)
    k, sk, b, sb = lsq(x, y)
    xx = np.linspace(0, np.nanmax(x) * 1.05, 100)
    if np.isfinite(k0):
        ax.plot(xx, k0 * xx, color="#c00000", lw=1.0,
                label=rf"МНК: $k={k0:.3f}\pm{sk0:.3f}$")
    if np.isfinite(k):
        ax.plot(xx, k * xx + b, color="#c00000", lw=0.8, ls="-.",
                label=rf"МНК: $k={k:.3f}\pm{sk:.3f}$, $b={b:.3f}$")
    if theory_k is not None:
        ax.plot(xx, theory_k * xx, color="k", lw=0.8, ls="--", label=theory_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set_title(title)
    save(fig, outfile)


def plot_filter(params: dict) -> None:
    """K(nu) с теоретической кривой и линеаризация 1/K^2 от nu^2."""
    df = load("filter_processed.csv")
    if df.empty:
        return skip("фильтрация")

    nu = df["nu_kHz"].to_numpy(dtype=float)
    K = df["K"].to_numpy(dtype=float)
    invK2 = df["invK2"].to_numpy(dtype=float)

    slope, sigma_slope = lsq_origin(nu**2, invK2 - 1.0)
    tau_rc = np.sqrt(slope) / (2 * np.pi) if np.isfinite(slope) and slope > 0 else np.nan

    # --- K(nu)
    fig, ax = plt.subplots()
    ax.errorbar(nu, K, yerr=df.get("sigma_K"), **MARK)
    if np.isfinite(tau_rc):
        xx = np.linspace(0, np.nanmax(nu) * 1.05, 400)
        ax.plot(xx, 1 / np.sqrt(1 + (2 * np.pi * xx * tau_rc) ** 2),
                color="crimson", lw=1.0,
                label=rf"$K=1/\sqrt{{1+(2\pi\nu\tau_{{RC}})^2}}$, "
                      rf"$\tau_{{RC}}={tau_rc * 1e3:.1f}$ мкс")
        ax.axhline(1 / np.sqrt(2), color="k", lw=0.7, ls=":")
        ax.annotate(r"$K=1/\sqrt{2}$", xy=(0.02, 1 / np.sqrt(2) + 0.02),
                    xycoords=("axes fraction", "data"), fontsize=9)
    ax.set_xlabel(r"$\nu$, кГц")
    ax.set_ylabel(r"$K$, отн. ед.")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Амплитудный коэффициент фильтрации RC-цепочки")
    save(fig, "fig_filter.pdf")

    # --- линеаризация
    fig, ax = plt.subplots()
    ax.errorbar(nu**2, invK2, yerr=df.get("sigma_invK2"), **MARK)
    if np.isfinite(slope):
        xx = np.linspace(0, np.nanmax(nu**2) * 1.05, 100)
        ax.plot(xx, 1 + slope * xx, color="#c00000", lw=1.0,
                label=rf"МНК: $1/K^2=1+({2 * np.pi:.3f}\tau_{{RC}})^2\nu^2$, "
                      rf"наклон $={slope:.4f}\pm{sigma_slope:.4f}$ мс$^2$")
    ax.set_xlabel(r"$\nu^2$, кГц$^2$")
    ax.set_ylabel(r"$1/K^2$, отн. ед.")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set_title("Линеаризованная АЧХ RC-фильтра")
    save(fig, "fig_filter_lin.pdf")


def main() -> None:
    params = read_params()
    print("Построение графиков в report/img/:")
    plot_harmonics(params)
    plot_line("spectrum_width_processed.csv", "inv_tau_kHz", "Dnu_kHz",
              "sigma_inv_tau_kHz", "sigma_Dnu_kHz",
              r"$1/\tau$, кГц", r"$\Delta\nu$, кГц",
              r"Ширина спектра прямоугольных импульсов ($T=\mathrm{const}$)",
              "fig_width.pdf", theory_k=1.0, theory_label=r"теория: $\Delta\nu=1/\tau$")
    plot_line("harmonics_spacing_processed.csv", "inv_T_kHz", "dnu_kHz",
              "sigma_inv_T_kHz", "sigma_dnu_kHz",
              r"$1/T$, кГц", r"$\delta\nu$, кГц",
              r"Расстояние между гармониками ($\tau=\mathrm{const}$)",
              "fig_spacing.pdf", theory_k=1.0, theory_label=r"теория: $\delta\nu=1/T$")
    plot_line("am_processed.csv", "m_used", "ratio", "sigma_m_exp", "sigma_ratio",
              r"$m$, отн. ед.", r"$a_{\mathrm{бок}}/a_{\mathrm{осн}}$, отн. ед.",
              "Амплитудная модуляция", "fig_am.pdf",
              theory_k=0.5, theory_label=r"теория: $a_{бок}/a_{осн}=m/2$")
    plot_filter(params)
    print("Готово.")


if __name__ == "__main__":
    main()
