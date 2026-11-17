"""Проверка бюджета погрешностей независимым возмущением исходных показаний."""
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.constants import e, Boltzmann, atomic_mass, epsilon_0, electron_mass, torr

DATA = Path(__file__).resolve().parents[1] / "data"


def step_sizes(series):
    return np.array([10.0 ** Decimal(x).as_tuple().exponent for x in series])


def calculate(u, i, masks):
    lines = [np.poly1d(np.polyfit(u[m], i[m], 1)) for m in masks]
    positive, negative, center = lines
    u0 = (2 * center - positive - negative).r.item()
    i0 = center(u0)
    ion = (positive(u0) - negative(u0)) / 2
    te = ion / (2 * center[1])
    ne = ion * 1e-6 / (0.4 * e * np.pi * .0002 * .0052 * np.sqrt(2 * te * e / (22 * atomic_mass)))
    wp = np.sqrt(ne * e**2 / (epsilon_0 * electron_mass))
    re = np.sqrt(te * e / electron_mass) / wp
    rd = np.sqrt(Boltzmann * 300 / electron_mass) / wp
    return dict(U_off_V=u0, I_off_uA=i0, Iin_uA=ion, dIdU_uA_V=center[1],
                Te_eV=te, Te_K=te*e/Boltzmann, n_1e10_cm3=ne/1e16,
                wp_1e9_rad_s=wp/1e9, fp_MHz=wp/(2*np.pi*1e6),
                rDe_um=re*1e6, rD_um=rd*1e6, ND=4*np.pi*rd**3*ne/3,
                alpha_1e7=ne*Boltzmann*300/(2*torr)*1e7)


def check_probes():
    raw = pd.read_csv(DATA / "raw/probe_iv.csv", dtype=str, comment="#")
    budget = pd.read_csv(DATA / "processed/ai_probe_budget.csv")
    for current, group in raw.groupby("Ip_mA"):
        u, i = group.U_V.to_numpy(float), group.I_uA.to_numpy(float)
        masks = (u >= 16, u <= -16, abs(u) <= 4.6)
        nominal = calculate(u, i, masks)
        z = np.r_[u, i]
        n = len(u)
        digits = np.r_[step_sizes(group.U_V), step_sizes(group.I_uA)] / np.sqrt(3)
        # Каждый столбец — отдельный источник ошибки в исходных координатах.
        directions = np.column_stack((np.diag(digits),
            np.r_[u, np.zeros(n)]*.03/np.sqrt(3),
            np.r_[np.zeros(n), i]*.03/np.sqrt(3)))
        derivatives = []
        for direction in directions.T:
            h = .001
            a, b = z + h*direction, z - h*direction
            plus, minus = calculate(a[:n], a[n:], masks), calculate(b[:n], b[n:], masks)
            derivatives.append([(plus[k]-minus[k])/(2*h) for k in nominal])
        derivative = np.array(derivatives).T
        expected_digits = np.linalg.norm(derivative[:, :-2], axis=1)
        expected_scale = np.linalg.norm(derivative[:, -2:], axis=1)
        table = budget[budget.Ip_mA == float(current)].set_index("quantity").loc[list(nominal)]
        np.testing.assert_allclose(table.value, list(nominal.values()), rtol=1e-8)
        np.testing.assert_allclose(table.s_digits, expected_digits, rtol=2e-5, atol=1e-8)
        np.testing.assert_allclose(table.s_scale, expected_scale, rtol=2e-5, atol=1e-8)
        np.testing.assert_allclose(table.s_total**2,
                                   table.s_fit**2+table.s_digits**2+table.s_scale**2)
        # Общая шкала A2 сокращается в энергии I/(2g).
        np.testing.assert_allclose(derivative[list(nominal).index("Te_eV"), -1], 0, atol=1e-8)


def check_resistance():
    raw = pd.read_csv(DATA / "raw/discharge_iv.csv", dtype=str, comment="#")
    results = pd.read_csv(DATA / "processed/ai_discharge_extrema.csv")
    for row in results.itertuples():
        g = raw[(raw.branch == row.branch) & raw.I_mA.astype(float).between(row.I_low, row.I_high)]
        x, y = g.I_mA.to_numpy(float), 10*g.U_V1_V.to_numpy(float)
        z = np.r_[x, y]
        digits = np.r_[step_sizes(g.I_mA), 10*step_sizes(g.U_V1_V)]/np.sqrt(3)
        deriv = []
        for delta in np.diag(digits):
            a, b = z+.001*delta, z-.001*delta
            deriv.append((np.polyfit(a[:len(x)],a[len(x):],1)[0]
                          -np.polyfit(b[:len(x)],b[len(x):],1)[0])/.002)
        np.testing.assert_allclose(np.linalg.norm(deriv), row.sR_digits, rtol=1e-5)
        np.testing.assert_allclose(abs(row.R_kOhm)*.03*np.sqrt(2/3), row.sR_scale, rtol=1e-7)
    ign = pd.read_csv(DATA / "processed/ai_ignition.csv").iloc[0]
    np.testing.assert_allclose(ign.sU_kV, np.sqrt(1+6.3**2)/np.sqrt(3)*.01)


def check_between_regimes():
    summary = pd.read_csv(DATA / "processed/ai_plasma_params.csv").set_index("Ip_mA", drop=False)
    covariance = pd.read_csv(DATA / "processed/ai_between_regimes.csv")
    # Законы масштабирования: Te ~ V2, ne ~ A2/sqrt(V2), Ip ~ A1.
    factors = {"Te_eV": .03**2/3, "n_1e10_cm3": 1.25*.03**2/3, "Ip_mA": .03**2/3}
    assert len(covariance) == 27
    for row in covariance.itertuples():
        a, b = summary.loc[row.I1_mA], summary.loc[row.I2_mA]
        expected = (a["s"+row.quantity]**2 if row.I1_mA == row.I2_mA
                    else factors[row.quantity]*a[row.quantity]*b[row.quantity])
        np.testing.assert_allclose(row.covariance, expected, rtol=1e-7)


if __name__ == "__main__":
    check_probes()
    check_resistance()
    check_between_regimes()
    print("OK: independent input perturbation, 39 budgets, 4 slopes, ignition, 27 covariances")
