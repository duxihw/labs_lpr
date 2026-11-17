"""Обработка работы 3.5.1. Запуск: python scripts/process_data.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.constants import atomic_mass, Boltzmann, elementary_charge, electron_mass, epsilon_0, torr
from scipy.linalg import block_diag

from lab_utils import PROC_DIR, read_params, read_raw, write_processed

# Одинаковые окна для всех трёх режимов. Внутри окон напряжение задано в В.
CENTRAL = 4.6
TAIL = 16.0


def number(value, places=2):
    return f"{value:.{places}f}".replace(".", "{,}")


def measurement(value, sigma):
    """Округление по двум значащим цифрам стандартной ошибки."""
    if not np.isfinite(sigma) or sigma <= 0:
        return "$" + number(value) + "$"
    digits = 1 - int(np.floor(np.log10(sigma)))
    value, sigma = round(value, digits), round(sigma, digits)
    return "$" + number(value, max(0, digits)) + r"\pm" + number(sigma, max(0, digits)) + "$"


def rows(name, values):
    text = "\n".join(" & ".join(row) + r" \\" for row in values)
    (PROC_DIR / name).write_text(text + "\n", encoding="utf-8")


def straight_line(x, y):
    if len(x) < 3 or len(np.unique(x)) < 2:
        raise ValueError("Для МНК с оценкой дисперсии нужно минимум три точки и два разных напряжения/тока")
    design = np.column_stack((x, np.ones_like(x)))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    variance = (residual @ residual) / (len(x) - 2)
    covariance = variance * np.linalg.inv(design.T @ design)
    return coefficients, covariance


def process_discharge(params):
    frame = read_raw("discharge_iv.csv")
    if frame.empty:
        return
    frame["U_r_V"] = params["k_div"] * frame["U_V1_V"]
    summary, table, windows = [], [], []
    for branch, group in frame.groupby("branch", sort=False):
        group = group.sort_values("I_mA")
        write_processed(group, f"discharge_iv_{branch}.csv")
        # Несколько измерений при одном токе могут входить в одну оценку напряжения.
        means = group.groupby("I_mA")["U_r_V"].mean()
        current, voltage = means.index.to_numpy(float), means.to_numpy(float)
        line_results = []
        for j in range(len(current) - 2):
            block = slice(j, j + 3)
            (r, _), cov = straight_line(current[block], voltage[block])
            line_results.append(dict(branch=branch, I_low=current[j], I_high=current[j + 2],
                                     R_kOhm=r, sR_kOhm=np.sqrt(cov[0, 0])))
        windows.extend(line_results)
        steep = max(line_results, key=lambda row: abs(row["R_kOhm"]))
        highest = max(line_results, key=lambda row: row["R_kOhm"])
        summary.extend([dict(criterion=key, **row) for key, row in
                        [("max_abs", steep), ("max_signed", highest)]])
        cells = [{"up": "Рост", "down": "Спад"}.get(branch, branch)]
        for row in (steep, highest):
            cells += ["$" + number(row["I_low"]) + r"\ldots " + number(row["I_high"]) + "$",
                      measurement(row["R_kOhm"], row["sR_kOhm"])]
        table.append(cells)
    write_processed(frame, "discharge_iv_processed.csv")
    write_processed(pd.DataFrame(summary), "discharge_extrema.csv")
    write_processed(pd.DataFrame(windows), "discharge_slopes.csv")
    rows("discharge_rows.tex", table)


def fit_regions(u, i, center, tail):
    """Три независимые прямые по непересекающимся выборкам."""
    masks = (u >= tail, u <= -tail, np.abs(u) <= center)
    lines, covariances = zip(*(straight_line(u[m], i[m]) for m in masks))
    return np.concatenate(lines), block_diag(*covariances)


def evaluate(coefficients, params):
    pos, bp, neg, bn, slope, intercept = coefficients
    # В новом начале центральная прямая пересекает среднюю линию асимптот.
    u0 = ((bp + bn) / 2 - intercept) / (slope - (pos + neg) / 2)
    i0 = slope * u0 + intercept
    ion = ((pos - neg) * u0 + bp - bn) / 2
    te = ion / (2 * slope)
    if ion <= 0 or te <= 0:
        raise ValueError("Неположительная энергия или ток насыщения: проверьте полярность и окна")
    energy = elementary_charge * te
    area = np.pi * params["d_probe"] * params["l_probe"] * 1e-6
    density = ion * 1e-6 / (0.4 * elementary_charge * area * np.sqrt(2 * energy / (params["A_ion"] * atomic_mass)))
    omega = np.sqrt(density * elementary_charge**2 / (epsilon_0 * electron_mass))
    rde = np.sqrt(epsilon_0 * energy / (density * elementary_charge**2))
    ti = params["T_gas"]
    rd = np.sqrt(epsilon_0 * Boltzmann * ti / (density * elementary_charge**2))
    nd = (4 * np.pi / 3) * density * rd**3
    alpha = density * Boltzmann * ti / (params["P_gas"] * torr)
    return dict(U_off_V=u0, I_off_uA=i0, Iin_uA=ion, dIdU_uA_V=slope,
                Te_eV=te, Te_K=energy / Boltzmann, n_1e10_cm3=density / 1e16,
                wp_1e9_rad_s=omega / 1e9, fp_MHz=omega / (2 * np.pi * 1e6),
                rDe_um=rde * 1e6, rD_um=rd * 1e6, ND=nd, alpha_1e7=alpha * 1e7)


def process_one_probe(ip, part, params):
    u = part["U_V"].to_numpy(float)
    i = part["I_uA"].to_numpy(float)
    coefs, cov = fit_regions(u, i, CENTRAL, TAIL)
    result = evaluate(coefs, params)
    # Полная ковариация наклонов и свободных членов. Центральная разность
    # берётся по каждому из шести коэффициентов в его исходных единицах.
    gradients = []
    for index, value in enumerate(coefs):
        step = 1e-5 * max(1, abs(value))
        delta = np.zeros_like(coefs)
        delta[index] = step
        above, below = evaluate(coefs + delta, params), evaluate(coefs - delta, params)
        gradients.append([(above[key] - below[key]) / (2 * step) for key in result])
    jacobian = np.array(gradients).T
    sigmas = np.sqrt(np.maximum(0, np.diag(jacobian @ cov @ jacobian.T)))
    errors = {"s" + key: sigma for key, sigma in zip(result, sigmas)}
    pos, bp, neg, bn, slope, _ = coefs
    row = dict(Ip_mA=ip, **result, **errors, asym_slope_pos=pos, asym_slope_neg=neg,
               asym_b_pos=result["Iin_uA"], asym_b_neg=-result["Iin_uA"], U_lin_V=CENTRAL,
               U_asym_V=TAIL, n_pos=int(sum(u >= TAIL)), n_neg=int(sum(u <= -TAIL)),
               n_central=int(sum(abs(u) <= CENTRAL)))
    frame = pd.DataFrame(dict(Ip_mA=ip, U_c_V=u - result["U_off_V"], I_c_uA=i - result["I_off_uA"]))
    return frame.sort_values("U_c_V"), row


def plot_lines(index, frame, row):
    ion, slope = row["Iin_uA"], row["dIdU_uA_V"]
    u_min, u_max = frame["U_c_V"].min(), frame["U_c_V"].max()
    lines = []
    for start, end, a, b, style in [
        (0, u_max, row["asym_slope_pos"], ion, "dashed"),
        (u_min, 0, row["asym_slope_neg"], -ion, "dashed"),
        (-ion / slope, ion / slope, slope, 0, "solid"),
    ]:
        lines.append(fr"\addplot[thin,black,{style},no marks,forget plot] coordinates "
                     f"{{({start:.8f},{a * start + b:.8f}) ({end:.8f},{a * end + b:.8f})}};")
    (PROC_DIR / f"probe_lines_{index}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    params = read_params()
    process_discharge(params)
    probe = read_raw("probe_iv.csv")
    if probe.empty:
        return
    results, checks, diagnostics, family = [], [], [], []
    probe_rows, base_rows, debye_rows, plasma_rows = [], [], [], []
    for index, (ip, group) in enumerate(sorted(probe.groupby("Ip_mA"), reverse=True), 1):
        frame, row = process_one_probe(ip, group, params)
        results.append(row)
        filename = f"probe_centered_{index}.csv"
        write_processed(frame, filename)
        plot_lines(index, frame, row)
        mark = ("*", "square*", "triangle*")[(index - 1) % 3]
        family.append(fr"\addplot+[only marks,mark={mark},mark size=1.4pt] "
                      fr"table[col sep=comma,x=U_c_V,y=I_c_uA] {{../data/processed/{filename}}};")
        family.append(r"\addlegendentry{$" + number(ip, 2) + r"$ мА}")
        ip_cell = "$" + number(ip) + "$"
        def cells(keys):
            return [measurement(row[k], row["s" + k]) for k in keys]
        probe_rows.append([ip_cell] + cells(["Iin_uA", "dIdU_uA_V", "Te_eV"]) +
                          ["$" + number(100 * row["sTe_eV"] / row["Te_eV"], 1) + "$"])
        base = [ip_cell, measurement(row["Te_K"] / 1e3, row["sTe_K"] / 1e3)] + cells(
            ["n_1e10_cm3", "wp_1e9_rad_s", "fp_MHz"])
        debye = [ip_cell] + cells(["rDe_um", "rD_um", "ND", "alpha_1e7"])
        base_rows.append(base)
        debye_rows.append(debye)
        plasma_rows.append(base[:-1] + debye[1:])
        for center, tail in [(3.0, TAIL), (6.5, TAIL), (CENTRAL, 15), (CENTRAL, 18)]:
            cc, _ = fit_regions(group["U_V"].to_numpy(), group["I_uA"].to_numpy(), center, tail)
            checks.append(dict(Ip_mA=ip, center_V=center, tail_V=tail, Te_eV=evaluate(cc, params)["Te_eV"]))
        if {"quality", "polarity"}.issubset(group.columns):
            clear = group.loc[group["quality"] != "uncertain"]
            coefs, _ = fit_regions(clear["U_V"].to_numpy(), clear["I_uA"].to_numpy(), CENTRAL, TAIL)
            central = group.loc[group["U_V"].abs() <= CENTRAL]
            # Один наклон и два независимых пересечения с осью тока.
            design = np.column_stack((central["U_V"], central["polarity"] == "+", central["polarity"] == "-"))
            (slope, plus, minus), _, rank, _ = np.linalg.lstsq(design, central["I_uA"], rcond=None)
            if rank != 3 or slope <= 0:
                raise ValueError("Не определён общий наклон двух полярностей")
            diagnostics.append(dict(Ip_mA=ip, n_uncertain=len(group) - len(clear),
                                    Te_clear_eV=evaluate(coefs, params)["Te_eV"],
                                    slope_separate_uA_V=slope, intercept_plus_uA=plus, intercept_minus_uA=minus,
                                    Te_separate_eV=row["Iin_uA"] / (2 * slope)))
    (PROC_DIR / "probe_family.tex").write_text("\n".join(family) + "\n", encoding="utf-8")
    full = pd.DataFrame(results)
    write_processed(full, "probe_results.csv")
    write_processed(full[["Ip_mA"] + [field for key in ("Te_eV", "Te_K", "n_1e10_cm3",
        "wp_1e9_rad_s", "fp_MHz", "rDe_um", "rD_um", "ND", "alpha_1e7")
        for field in (key, "s" + key)]], "plasma_params.csv")
    write_processed(pd.DataFrame(checks), "probe_window_check.csv")
    write_processed(pd.DataFrame(diagnostics), "probe_quality_check.csv")
    for name, table in (("probe_rows", probe_rows), ("plasma_base_rows", base_rows),
                        ("plasma_debye_rows", debye_rows), ("plasma_rows", plasma_rows)):
        rows(name + ".tex", table)


if __name__ == "__main__":
    main()
    import sys
    from ai_uncertainty import generate
    generate(read_params(), sys.modules[__name__])
