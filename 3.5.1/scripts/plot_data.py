"""Графики лабораторной работы 3.5.1 для лабораторного журнала и отчёта.

Читает data/raw/probe_iv.csv (обработка -- функциями process_data.py)
и сохраняет в report/img/:
    probe_1.pdf, probe_2.pdf, ...  -- зондовые ВАХ по отдельности
                                      с асимптотами и касательной в нуле
                                      (номер = порядок по убыванию I_р,
                                      как в data/processed/probe_centered_*.csv).

Остальные графики (ВАХ разряда, семейство зондовых ВАХ, T_e(I_р), n_e(I_р))
строятся в отчёте средствами pgfplots прямо из data/processed/*.csv.

Запуск (из каталога работы):
    python scripts/plot_data.py
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lab_utils import FIG_DIR, read_params, read_raw  # noqa: E402
from process_data import process_one_probe  # noqa: E402

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
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.figsize": (6.3, 4.2),
    "savefig.bbox": "tight",
})


def plot_probe(k: int, ip: float, part, params) -> None:
    centered, row = process_one_probe(ip, part, params)
    u, i = centered["U_c_V"].to_numpy(), centered["I_c_uA"].to_numpy()
    umax = np.nanmax(np.abs(u)) * 1.05
    uu = np.linspace(-umax, umax, 200)

    fig, ax = plt.subplots()
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.plot(u, i, "o", ms=4, color="#1f4e79", label="эксперимент (центрировано)")

    up = uu[uu > 0]
    un = uu[uu < 0]
    ax.plot(up, row["asym_slope_pos"] * up + row["asym_b_pos"], "--", color="#c00000", lw=0.9,
            label="асимптоты")
    ax.plot(un, row["asym_slope_neg"] * un + row["asym_b_neg"], "--", color="#c00000", lw=0.9)
    ul = np.linspace(-1.6 * row["U_lin_V"], 1.6 * row["U_lin_V"], 50)
    ax.plot(ul, row["dIdU_uA_V"] * ul, "-", color="#2e7d32", lw=1.0,
            label=rf"касательная в нуле: $dI/dU = {row['dIdU_uA_V']:.3g}$ мкА/В")
    for s in (+1, -1):
        ax.plot(0, s * row["Iin_uA"], "s", color="#c00000", ms=5)
    ax.set_title(rf"$I_р = {ip:g}$ мА: $I_{{iн}} = {row['Iin_uA']:.3g}$ мкА, "
                 rf"$kT_e = {row['Te_eV']:.3g}$ эВ")
    ax.set_xlabel(r"$U$, В")
    ax.set_ylabel(r"$I$, мкА")
    ax.legend(loc="upper left")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"probe_{k}.pdf")
    plt.close(fig)
    print(f"  -> report/img/probe_{k}.pdf")


def main() -> None:
    params = read_params()
    df = read_raw("probe_iv.csv")
    if df.empty:
        print("Нет данных зондовых характеристик -- графики не построены.")
        return
    groups = sorted(df.groupby("Ip_mA"), key=lambda g: -g[0])
    for k, (ip, part) in enumerate(groups, start=1):
        plot_probe(k, float(ip), part, params)
    print("Готово.")


if __name__ == "__main__":
    main()
