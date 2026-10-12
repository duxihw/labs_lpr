"""Графики работы 3.3.4 по данным из data/processed/ -> report/images/*.pdf.

Запуск (после process_data.py):  python scripts/plot_data.py
Если каких-то данных ещё нет, соответствующий график пропускается,
а в отчёте на его месте остаётся рамка-заглушка.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab_utils import FIG_DIR, PROC_DIR, lsq

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "figure.figsize": (6.4, 4.2),
    "savefig.bbox": "tight",
})


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
    print(f"  -> report/images/{name}")


def line(ax, x, k, b, **kw):
    xs = np.linspace(min(0, np.min(x)), np.max(x) * 1.05, 50)
    ax.plot(xs, k * xs + b, **kw)


def plot_calibration() -> None:
    df = load("calibration_processed.csv")
    if df.empty:
        print("  [-] calibration: нет данных")
        return
    fig, ax = plt.subplots()
    ax.plot(df["I_M_A"], df["B_mT"], "o-", ms=4, lw=1)
    ax.set_xlabel(r"$I_\mathrm{M}$, А")
    ax.set_ylabel(r"$B$, мТл")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    save(fig, "calibration.pdf")


def plot_family() -> None:
    df = load("hall_processed.csv")
    sl = load("slopes.csv")
    if df.empty or sl.empty:
        print("  [-] hall_family: нет данных")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    markers = "osD^v<>ph*"
    for i, (s, g) in enumerate(df.groupby("series")):
        r = sl[sl["series"] == s].iloc[0]
        B = g["B_mT"].to_numpy() / 1e3
        c = f"C{i % 10}"
        ax.plot(B, g["U_perp_mV"], markers[i % len(markers)], color=c, ms=4,
                label=rf"$I = {r['I_mA']:g}$ мА")
        line(ax, B, r["k_mV_T"], r["b_mV"], color=c, lw=0.9)
    ax.set_xlabel(r"$B$, Тл")
    ax.set_ylabel(r"$U_\perp$, мВ")
    ax.legend(fontsize=9, ncol=2)
    save(fig, "hall_family.pdf")


def plot_k_vs_I() -> None:
    sl = load("slopes.csv")
    if len(sl) < 2:
        print("  [-] k_vs_I: нет данных")
        return
    K, sK, b, sb = lsq(sl["I_mA"], sl["k_mV_T"])
    fig, ax = plt.subplots()
    ax.errorbar(sl["I_mA"], sl["k_mV_T"], yerr=sl["sigma_k"], fmt="o", ms=4, capsize=3,
                label="эксперимент")
    if np.isfinite(K):
        line(ax, sl["I_mA"].to_numpy(), K, b, color="C1", lw=1,
             label=rf"МНК: $k = ({K:.3g})\,I {'+' if b >= 0 else '-'} {abs(b):.2g}$")
    ax.set_xlabel(r"$I$, мА")
    ax.set_ylabel(r"$k = dU_\perp/dB$, мВ/Тл")
    ax.legend()
    save(fig, "k_vs_I.pdf")


def plot_reverse() -> None:
    rv = load("reverse_processed.csv")
    if rv.empty:
        print("  [-] reverse: нет данных")
        return
    fig, ax = plt.subplots()
    B = rv["B_mT"].to_numpy() / 1e3
    ax.plot(B, rv["U34_plus_mV"], "^", ms=5, label=r"$U_{34}^{(+)}$")
    ax.plot(B, rv["U34_minus_mV"], "v", ms=5, label=r"$U_{34}^{(-)}$")
    ax.plot(B, rv["U_perp_mV"], "o", ms=4, label=r"$U_\perp = (U^{(+)}_{34}-U^{(-)}_{34})/2$")
    k, sk, b, sb = lsq(B, rv["U_perp_mV"])
    if np.isfinite(k):
        line(ax, B, k, b, color="C2", lw=0.9)
    ax.set_xlabel(r"$|B|$, Тл")
    ax.set_ylabel(r"$U$, мВ")
    ax.legend(fontsize=9)
    save(fig, "reverse.pdf")


def main() -> None:
    plot_calibration()
    plot_family()
    plot_k_vs_I()
    plot_reverse()


if __name__ == "__main__":
    main()
