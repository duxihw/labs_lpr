"""Быстрые графики лабораторной работы 3.7.3 «Длинные линии».

Читает data/processed/*.csv (их готовит process_data.py) и сохраняет
PNG-графики в report/img/check_*.png -- для проверки данных прямо во время
работы, без пересборки LaTeX. В самом отчёте те же зависимости строятся
pgfplots прямо из CSV.

Запуск (из каталога работы):
    python scripts/plot_data.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from lab_utils import FIG_DIR, PROC_DIR  # noqa: E402

plt.rcParams.update({"font.family": "serif", "axes.grid": True, "font.size": 11})


def load(name: str) -> pd.DataFrame:
    path = PROC_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  -> report/img/{name}")


def xy(ax, df, x, y, fit=None, label=None, **kw):
    if df.empty:
        return
    ax.plot(df[x], df[y], "o", label=label, **kw)
    if fit and fit in df:
        ax.plot(df[x], df[fit], "-", lw=1)


def main() -> None:
    # Резонансы кабеля
    fig, ax = plt.subplots()
    for sig in ["sine", "pulse"]:
        for load_ in ["matched", "open"]:
            xy(ax, load(f"cable_res_{sig}_{load_}.csv"), "n", "nu_MHz",
               "nu_fit_MHz", f"{sig}, {load_}")
    ax.set(xlabel="$n$", ylabel=r"$\nu_n$, МГц")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    save(fig, "check_cable_res.png")

    afc = load("afc_processed.csv")
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    xy(axs[0, 0], afc, "nu_MHz", "Kamp")
    axs[0, 0].set(xlabel=r"$\nu$, МГц", ylabel=r"$U_н/U_0$", title="АЧХ")
    xy(axs[0, 1], afc, "nu_MHz", "dphi_rad")
    axs[0, 1].set(xlabel=r"$\nu$, МГц", ylabel=r"$\Delta\varphi$, рад", title="ФЧХ")
    xy(axs[1, 0], afc, "x2", "y2", "y2_fit")
    axs[1, 0].set(xlabel=r"$\sqrt{\nu}$, $10^3$ Гц$^{1/2}$",
                  ylabel=r"$\alpha$, $10^{-3}$ м$^{-1}$", title="Метод А")
    xy(axs[1, 1], afc, "x3", "y3", "y3_fit")
    axs[1, 1].set(xlabel=r"$\nu^{3/2}$, $10^9$ Гц$^{3/2}$",
                  ylabel=r"$\alpha k$, $10^{-3}$ м$^{-2}$", title="Метод Б")
    save(fig, "check_afc.png")

    fig, ax = plt.subplots()
    xy(ax, afc, "x1", "y1", "y1_fit")
    ax.set(xlabel=r"$\omega^2$, $10^{15}$ с$^{-2}$", ylabel=r"$k^2-\alpha^2$, м$^{-2}$")
    save(fig, "check_y1.png")

    fig, ax = plt.subplots()
    th = load("model_phase_theory.csv")
    if not th.empty:
        ax.plot(th["nu_kHz"], th["dphi_chain"], "-", label="цепочка")
        ax.plot(th["nu_kHz"], th["dphi_cont"], "--", label="непрерывная линия")
    xy(ax, load("model_phase_processed.csv"), "nu_kHz", "dphi_rad", label="эксперимент")
    ax.set(xlabel=r"$\nu$, кГц", ylabel=r"$\Delta\varphi$, рад", ylim=(0, 3.5))
    ax.legend()
    save(fig, "check_model_phase.png")

    dist = load("model_distribution_processed.csv")
    fig, ax = plt.subplots()
    for i in range(1, 5):
        if f"U{i}_mV" in dist and dist[f"U{i}_mV"].notna().any():
            ax.plot(dist["cell"], dist[f"U{i}_mV"], "o-", label=f"серия {i}")
    ax.set(xlabel="номер клеммы", ylabel="$U$, мВ")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    save(fig, "check_model_dist.png")


if __name__ == "__main__":
    main()
