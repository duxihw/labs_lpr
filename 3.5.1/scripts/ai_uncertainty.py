"""Условный бюджет погрешностей для варианта 351-ai.

У каждого из четырёх приборов один общий относительный множитель ±3 %
(все точки, обе полярности и все режимы). Каждое отдельное показание имеет
дополнительную независимую погрешность ±1 единица последнего записанного
разряда. Пределы прямоугольные. Последовательное линейное распространение
сохраняет ковариации величин. Дополнительный разброс регрессии учитывается
независимо от приборного бюджета, что может давать консервативную оценку.
"""

from decimal import Decimal

import numpy as np
import pandas as pd
from scipy.linalg import block_diag

from lab_utils import RAW_DIR, PROC_DIR, write_processed

SCALE = 0.03
ROOT3 = np.sqrt(3)


def readings(strings):
    """Последний разряд зависит от записи: 1.5, 1.50 и 150 различаются."""
    decimals = [Decimal(str(value)) for value in strings]
    values = np.array([float(value) for value in decimals])
    digits = np.array([float(Decimal(1).scaleb(value.as_tuple().exponent)) for value in decimals])
    return values, digits


def read_strings(filename):
    return pd.read_csv(RAW_DIR / filename, dtype=str, comment="#", keep_default_na=False)


def point_sigma(value, digit):
    return np.hypot(SCALE * value, digit) / ROOT3


def inputs(x, y, dx, dy):
    """Коэффициенты ошибок [детали отсчётов x и y, шкала x, шкала y], u=1."""
    n = len(x)
    independent = np.diag(np.r_[dx, dy] / ROOT3)
    gain_x = np.r_[x, np.zeros(n)] * SCALE / ROOT3
    gain_y = np.r_[np.zeros(n), y] * SCALE / ROOT3
    return np.column_stack((independent, gain_x, gain_y))


def line_jacobian(x, y):
    """Производные наклона и свободного члена МНК по x_1...x_N,y_1...y_N."""
    xc, yc = x - x.mean(), y - y.mean()
    sxx = xc @ xc
    slope = (xc @ yc) / sxx
    dk_dx, dk_dy = (yc - 2 * slope * xc) / sxx, xc / sxx
    db_dx = -slope / len(x) - x.mean() * dk_dx
    db_dy = 1 / len(x) - x.mean() * dk_dy
    return np.stack([np.r_[dk_dx, dk_dy], np.r_[db_dx, db_dy]])


def quantity_jacobian(evaluate, coefs, params):
    nominal = evaluate(coefs, params)
    columns = []
    for j, coefficient in enumerate(coefs):
        step = 1e-5 * max(1, abs(coefficient))
        change = np.zeros_like(coefs)
        change[j] = step
        above, below = evaluate(coefs + change, params), evaluate(coefs - change, params)
        columns.append([(above[key] - below[key]) / (2 * step) for key in nominal])
    return nominal, np.asarray(columns).T


def variances(jacobian, covariance):
    return np.einsum("ij,jk,ik->i", jacobian, covariance, jacobian)


def axis_data(group, xkey, ykey, xmult=1, ymult=1):
    x, dx = readings(group[xkey])
    y, dy = readings(group[ykey])
    return x * xmult, y * ymult, dx * xmult, dy * ymult


def discharge(params, label, rows, measurement, number):
    raw = read_strings("discharge_iv.csv")
    if raw.empty:
        return
    nominal = pd.read_csv(PROC_DIR / "discharge_extrema.csv")
    records, formatted = [], []
    for branch, group in raw.groupby("branch", sort=False):
        x, y, dx, dy = axis_data(group, "I_mA", "U_V1_V", ymult=params["k_div"])
        write_processed(pd.DataFrame(dict(I_mA=x, U_r_V=y, sI_mA=point_sigma(x, dx),
                                         sU_r_V=point_sigma(y, dy))).sort_values("I_mA"),
                        f"ai_discharge_iv_{branch}.csv")
        cells = [label.get(branch, branch)]
        for item in nominal[nominal.branch == branch].itertuples():
            keep = (x >= item.I_low) & (x <= item.I_high)
            # Идентичные значения тока усреднены тем же способом, что и в основном расчёте.
            unique, assignment, count = np.unique(x[keep], return_inverse=True, return_counts=True)
            mean = (np.arange(len(unique))[:, None] == assignment) / count[:, None]
            derivative = line_jacobian(unique, mean @ y[keep])[0] @ block_diag(mean, mean)
            effect = derivative @ inputs(x[keep], y[keep], dx[keep], dy[keep])
            digit = np.linalg.norm(effect[:-2])
            scale = np.linalg.norm(effect[-2:])
            sigma = np.sqrt(digit**2 + scale**2 + item.sR_kOhm**2)
            records.append(dict(branch=branch, criterion=item.criterion, I_low=item.I_low,
                                I_high=item.I_high, R_kOhm=item.R_kOhm, sR_kOhm=sigma,
                                sR_fit=item.sR_kOhm, sR_digits=digit, sR_scale=scale))
            cells += ["$" + number(item.I_low) + r"\ldots " + number(item.I_high) + "$",
                      measurement(item.R_kOhm, sigma)]
        formatted.append(cells)
    write_processed(pd.DataFrame(records), "ai_discharge_extrema.csv")
    rows("ai_discharge_rows.tex", formatted)


def probe_budget(group, params, model):
    u, i, du, di = axis_data(group, "U_V", "I_uA")
    n = len(u)
    coefs, cov_fit = model.fit_regions(u, i, model.CENTRAL, model.TAIL)
    fit_jac = np.zeros((6, 2 * n))
    for region, mask in enumerate((u >= model.TAIL, u <= -model.TAIL, np.abs(u) <= model.CENTRAL)):
        fit_jac[2 * region:2 * region + 2, np.r_[mask, mask]] = line_jacobian(u[mask], i[mask])
    values, jac = quantity_jacobian(model.evaluate, coefs, params)
    loading = inputs(u, i, du, di)
    instrument = (jac @ fit_jac) @ loading
    fit = np.sqrt(variances(jac, cov_fit))
    digits = np.linalg.norm(instrument[:, :-2], axis=1)
    scales = np.linalg.norm(instrument[:, -2:], axis=1)
    total = np.sqrt(fit**2 + digits**2 + scales**2)
    keys = list(values)
    details = pd.DataFrame(dict(quantity=keys, value=list(values.values()), s_fit=fit, s_digits=digits,
                                s_scale=scales, s_total=total))
    errors = {"s" + key: s for key, s in zip(keys, total)}

    # Центрирование: отсчёты и оценка нового начала зависят от одних измерений.
    # Идентичный общий сдвиг обеих координат корректно сокращается в разности.
    centered = pd.DataFrame()
    for col, key, shift, original in [("U_c_V", "U_off_V", 0, u), ("I_c_uA", "I_off_uA", n, i)]:
        j = keys.index(key)
        effect = loading[shift:shift + n] - instrument[j]
        centered[col] = original - values[key]
        centered["s" + col] = np.sqrt(np.sum(effect**2, axis=1) + fit[j]**2)
    return values | errors, details, centered.sort_values("U_c_V"), dict(zip(keys, instrument[:, -2:]))


def generate(params, model):
    """Вызывается после записи основных результатов; все выходы имеют префикс ai_."""
    discharge(params, {"up": "Рост", "down": "Спад"}, model.rows, model.measurement, model.number)
    source = read_strings("setup_params.csv")
    ignition = source.loc[source.param == "U_ign", "value"]
    if not ignition.empty and ignition.iloc[0]:
        value, digit = readings(ignition)
        factor = params["k_div"] / 1000
        nominal, sigma = value[0] * factor, point_sigma(value[0], digit[0]) * factor
        write_processed(pd.DataFrame([dict(U_kV=nominal, sU_kV=sigma)]), "ai_ignition.csv")
        (PROC_DIR / "ai_ignition.tex").write_text(model.measurement(nominal, sigma) + "\n", encoding="utf-8")

    raw = read_strings("probe_iv.csv")
    summaries, all_details, gain_rows, family = [], [], [], []
    table_probe, table_base, table_debye = [], [], []
    for index, (ip, group) in enumerate(sorted(raw.groupby(raw.Ip_mA.astype(float)), reverse=True), 1):
        # Ток разряда фиксировали один раз на всю серию, многократного усреднения нет.
        literals = group.Ip_mA.unique()
        if len(literals) != 1:
            raise ValueError(f"Разные разряды записи одного режима {ip}: {literals}")
        setpoint, digit = readings(literals)
        sip = point_sigma(setpoint[0], digit[0])
        row, details, centered, gain = probe_budget(group, params, model)
        summaries.append(dict(Ip_mA=ip, sIp_mA=sip, **row))
        all_details.append(details.assign(Ip_mA=ip))
        for key in ("Te_eV", "n_1e10_cm3"):
            gain_rows.append((ip, key, row["s" + key], *gain[key], 0.0))
        gain_rows.append((ip, "Ip_mA", sip, 0.0, 0.0, ip * SCALE / ROOT3))
        filename = f"ai_probe_centered_{index}.csv"
        write_processed(centered, filename)
        mark = ("*", "square*", "triangle*")[(index - 1) % 3]
        family.append(fr"\addplot+[only marks,mark={mark},mark size=1.4pt,error bars/.cd,"
                      r"x dir=both,x explicit,y dir=both,y explicit] table[col sep=comma,"
                      r"x=U_c_V,y=I_c_uA,x error=sU_c_V,y error=sI_c_uA] "
                      fr"{{../data/processed/{filename}}};")
        family.append(r"\addlegendentry{$" + model.number(ip, 2) + r"$ мА}")
        def cells(keys):
            return [model.measurement(row[k], row["s" + k]) for k in keys]
        ip_cell = model.measurement(ip, sip)
        table_probe.append([ip_cell] + cells(["Iin_uA", "dIdU_uA_V", "Te_eV"]) +
                           ["$" + model.number(100 * row["sTe_eV"] / row["Te_eV"], 1) + "$"])
        table_base.append([ip_cell, model.measurement(row["Te_K"] / 1000, row["sTe_K"] / 1000)]
                          + cells(["n_1e10_cm3", "wp_1e9_rad_s", "fp_MHz"]))
        table_debye.append([ip_cell] + cells(["rDe_um", "rD_um", "ND", "alpha_1e7"]))
    # Разные режимы имеют общие V2, A2 и A1. Для каждого параметра матрица
    # включает их межрежимные корреляции и независимые вклады на диагонали.
    cross = []
    for ip, key, sigma, gv, ga, gi in gain_rows:
        for jp, key2, _, hv, ha, hi in gain_rows:
            if key == key2:
                cross.append(dict(quantity=key, I1_mA=ip, I2_mA=jp,
                                  covariance=sigma**2 if ip == jp else gv * hv + ga * ha + gi * hi))
    write_processed(pd.DataFrame(cross), "ai_between_regimes.csv")
    write_processed(pd.DataFrame(summaries), "ai_plasma_params.csv")
    write_processed(pd.concat(all_details, ignore_index=True), "ai_probe_budget.csv")
    (PROC_DIR / "ai_probe_family.tex").write_text("\n".join(family) + "\n", encoding="utf-8")
    for filename, table in [("ai_probe_rows", table_probe), ("ai_plasma_base_rows", table_base),
                            ("ai_plasma_debye_rows", table_debye)]:
        model.rows(filename + ".tex", table)
