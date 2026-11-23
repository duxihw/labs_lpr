"""Векторные графики 3.3.4. Запуск после scripts/process_data.py."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab_utils import FIG_DIR, PROC_DIR

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "font.size": 10, "axes.grid": True, "grid.alpha": .18,
                     "axes.axisbelow": True, "figure.figsize": (6.6, 3.75),
                     "pdf.fonttype": 42})
STYLE = dict(linestyle="none", ms=3.4, capsize=1.6, elinewidth=.65, markeredgewidth=.65)


def load(name):
    return pd.read_csv(PROC_DIR / (name + ".csv"))


def save(fig, ax, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ax.spines[["right","top"]].set_visible(False)
    fig.tight_layout(pad=.6)
    fig.savefig(FIG_DIR / (name + ".pdf"), metadata={"CreationDate": None})
    plt.close(fig)


def calibration():
    d = load("calibration_processed")
    fig, ax = plt.subplots()
    ax.errorbar(d.I_M_A, d.B_T, xerr=d.dI_A, yerr=d.dB_T, fmt="o", color="#22568a", **STYLE)
    ax.set(xlabel=r"$I_{\mathrm{М}}$, А", ylabel=r"$B$, Тл", xlim=(0,1.72), ylim=(0,1.31))
    save(fig, ax, "calibration")


def family():
    d = load("hall_processed")
    slopes = load("slopes").set_index("series")
    fig, ax = plt.subplots(figsize=(7.1,4.5))
    colors = ["#234e77", "#a04c12", "#16855f", "#b73440", "#8051aa", "#8b6741", "#b34186", "#16788b"]
    markers = ["o","^","s","D","v","<",">","p"]
    for (series, g), c, m in zip(d.groupby("series"), colors, markers):
        row = slopes.loc[series]
        x = np.array([0, max(g.B_T)])
        ax.plot(x, row.k_mV_T*1000*x, color=c, lw=.8)
        ax.errorbar(g.B_T, g.U_perp_uV, xerr=g.dB_T, yerr=g.dU_perp_uV,
                    fmt=m, color=c, **STYLE, label=f"{row.I_mA:.1f}".replace(".",","))
    ax.set(xlabel=r"$B$, Тл", ylabel=r"$U_{34}-U_0$, мкВ")
    ax.legend(title=r"$I$, мА", ncol=2, loc="upper left", frameon=False,
              columnspacing=.8, handletextpad=.3, fontsize=9)
    save(fig, ax, "hall_family")


def slopes():
    d = load("slopes")
    K = load("fit_results").set_index("quantity").loc["K","value"]
    fig, ax = plt.subplots()
    x = np.array([0,1])
    ax.plot(x, K*x, lw=1, color="#ae4b22", label="МНК через начало координат")
    ax.errorbar(d.I_mA, d.k_mV_T, xerr=d.dI_mA, yerr=d.sigma_k, fmt="o", color="#22568a", **STYLE)
    ax.set(xlabel=r"$I$, мА", ylabel=r"$K(I)$, мВ/Тл")
    ax.legend(frameon=False)
    save(fig, ax, "k_vs_I")


def reversal():
    d = load("reverse_processed")
    k = load("fit_results").set_index("quantity").loc["Krev","value"] * 1000
    fig, ax = plt.subplots()
    x = np.array([0,d.B_T.max()])
    ax.plot(x, k*x, color="#314e71", lw=.9)
    ax.errorbar(d.B_T, d.Udirect_uV, xerr=d.dB_T, yerr=d.dUdirect_uV, fmt="s",
                mfc="white", color="#ae4b22", **STYLE, label="Прямая серия")
    ax.errorbar(d.B_T, d.Ucorr_uV, xerr=d.dB_T, yerr=d.dUcorr_uV, fmt="o",
                color="#22568a", **STYLE, label="Полуразность с поправкой на нули")
    ax.set(xlabel=r"$|B|$, Тл", ylabel="Холловское напряжение, мкВ")
    ax.legend(frameon=False)
    save(fig, ax, "reverse")


def main():
    calibration()
    family()
    slopes()
    reversal()


if __name__ == "__main__":
    main()
