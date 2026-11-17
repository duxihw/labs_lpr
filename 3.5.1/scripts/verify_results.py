"""Независимый пересчёт опубликованных результатов без импорта process_data."""
from math import pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.constants import Boltzmann, atomic_mass, e, electron_mass, epsilon_0, torr
from scipy.stats import linregress

DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    raw = pd.read_csv(DATA / "raw/probe_iv.csv", comment="#")
    output = pd.read_csv(DATA / "processed/probe_results.csv").set_index("Ip_mA")
    assert len(raw) == 83 and set(output.index) == set(raw.Ip_mA)
    for current, group in raw.groupby("Ip_mA"):
        u, i = group.U_V, group.I_uA
        def line(mask):
            return np.poly1d(np.polyfit(u[mask], i[mask], 1))
        upper = line(u >= 16)
        lower = line(u <= -16)
        tangent = line(abs(u) <= 4.6)
        center = (2 * tangent - upper - lower).r.item()
        ion = (upper(center) - lower(center)) / 2       # мкА
        energy = ion / (2 * tangent[1])                # эВ
        speed = sqrt(2 * e * energy / (22 * atomic_mass))
        area = pi * 0.0002 * 0.0052
        density = ion * 1e-6 / (e * area * 0.4 * speed)
        omega = sqrt(density / (epsilon_0 * electron_mass)) * e
        # Радиусы через тепловую скорость и электронную плазменную частоту.
        re = sqrt(energy * e / electron_mass) / omega
        rd = sqrt(300 * Boltzmann / electron_mass) / omega
        expected = dict(U_off_V=center, I_off_uA=tangent(center), Iin_uA=ion,
                        dIdU_uA_V=tangent[1], Te_eV=energy, Te_K=energy*e/Boltzmann,
                        n_1e10_cm3=density/1e16, wp_1e9_rad_s=omega/1e9,
                        fp_MHz=omega/(2*pi*1e6), rDe_um=re*1e6, rD_um=rd*1e6,
                        ND=density*4*pi*rd**3/3, alpha_1e7=density*Boltzmann*300/(2*torr)*1e7)
        np.testing.assert_allclose(output.loc[current, list(expected)].to_numpy(float),
                                   list(expected.values()), rtol=1e-8)

    raw = pd.read_csv(DATA / "raw/discharge_iv.csv", comment="#")
    assert len(raw) == 39
    extrema = pd.read_csv(DATA / "processed/discharge_extrema.csv").set_index(["branch", "criterion"])
    for branch, group in raw.groupby("branch"):
        g = group.groupby("I_mA").U_V1_V.mean().sort_index()
        fits = [linregress(g.index[j:j+3], 10*g.iloc[j:j+3]) for j in range(len(g)-2)]
        for criterion, key in (("max_abs", lambda f: abs(f.slope)), ("max_signed", lambda f: f.slope)):
            best = max(fits, key=key)
            row = extrema.loc[branch, criterion]
            np.testing.assert_allclose([best.slope, best.stderr], [row.R_kOhm, row.sR_kOhm])
    print("OK: 39 discharge points, 83 probe points, 3 plasma regimes, 4 resistance extrema")


if __name__ == "__main__":
    main()
