"""Обработка сырых данных лабораторной работы 3.5.1.

    «Изучение плазмы газового разряда в неоне»

Читает   data/raw/{setup_params,discharge_iv,probe_iv}.csv
Пишет    data/processed/*.csv  (в том числе fit_results.csv со всеми итогами)
         data/processed/probe_family.tex  (кривые семейства для pgfplots)
Печатает сводку с промежуточными и окончательными результатами.

Запуск (из каталога работы):
    python scripts/process_data.py

Сырые данные никогда не изменяются. Пока таблицы в data/raw/ пусты, скрипт
ничего не пишет, а отчёт собирается с пустыми таблицами-заготовками.

Формулы (методичка, раздел V):
    k T_e = e I_iн / (2 dI/dU|_0)                          (5.26)
    I_iн  = 0.4 n e S sqrt(2 k T_e / m_i),  S = pi d l     (5.22)
    w_p   = sqrt(4 pi n e^2 / m_e)          [СГС]          (5.1)
    r_De  = sqrt(k T_e / (4 pi n e^2)),  r_D ~ r_Di (T_i << T_e)   (5.9), (5.10)
    N_D   = 4/3 pi n r_D^3                                  (5.13)
    alpha = n / n_0,  n_0 = P / (k T)
Расчёт ведётся в СИ, результаты переводятся в удобные единицы.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab_utils import (
    PROC_DIR,
    Results,
    get_param,
    lsq,
    read_params,
    read_raw,
    write_processed,
)

# Физические константы (СИ)
E = 1.602176634e-19      # Кл
K_B = 1.380649e-23       # Дж/К
EPS0 = 8.8541878128e-12  # Ф/м
M_E = 9.1093837015e-31   # кг
AMU = 1.66053906660e-27  # кг
TORR = 133.322           # Па

RES = Results()


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def sig(x: float, n: int = 3) -> float:
    """Округление до n значащих цифр (для таблиц отчёта)."""
    if x is None or not np.isfinite(x) or x == 0:
        return x
    return float(f"{x:.{n}g}")


def warn_default(params: dict, name: str, default: float, what: str) -> float:
    value = get_param(params, name, default)
    if name not in params or not np.isfinite(params.get(name, np.nan)):
        print(f"  [!] {name} ({what}) не задан в setup_params.csv -> {default}")
    return value


# --------------------------------------------------------------------------
# 1. ВАХ тлеющего разряда
# --------------------------------------------------------------------------
def process_discharge(params: dict) -> None:
    """п. 8: I_р(U_р), дифференциальное сопротивление R_диф = dU/dI."""
    header("1. Вольт-амперная характеристика разряда")
    df = read_raw("discharge_iv.csv")
    if df.empty:
        print("  нет данных")
        return

    k_div = warn_default(params, "k_div", 1.0, "коэффициент делителя")
    df["U_r_V"] = k_div * df["U_V1_V"].astype(float)
    df["branch"] = df["branch"].astype(str).str.strip()

    u_ign = get_param(params, "U_ign", np.nan)
    u_ext = get_param(params, "U_ext", np.nan)
    if np.isfinite(u_ign):
        RES.add("U_ign", k_div * u_ign, unit="V", comment="напряжение зажигания")
    if np.isfinite(u_ext):
        RES.add("U_ext", k_div * u_ext, unit="V", comment="напряжение гашения")

    out = []
    for branch, part in df.groupby("branch", sort=False):
        part = part.sort_values("I_mA").copy()
        i = part["I_mA"].to_numpy(float)
        u = part["U_r_V"].to_numpy(float)
        # локальное дифференциальное сопротивление, В/мА = кОм
        part["R_diff_kOhm"] = np.gradient(u, i) if len(i) >= 2 else np.nan
        out.append(part)

        k, sk, b, sb = lsq(i, u)
        RES.add(f"R_diff_lsq_{branch}", k, sk, "kOhm",
                f"наклон МНК U_р(I_р) по всей ветви '{branch}'")
        if len(i) >= 2:
            j = int(np.nanargmax(np.abs(part["R_diff_kOhm"].to_numpy())))
            RES.add(f"R_diff_max_{branch}", part["R_diff_kOhm"].iloc[j],
                    unit="kOhm", comment=f"max |dU/dI| на ветви '{branch}' "
                                         f"при I_р = {i[j]:g} мА")

    res = pd.concat(out)
    res["U_r_V"] = res["U_r_V"].map(lambda x: sig(x, 4))
    res["R_diff_kOhm"] = res["R_diff_kOhm"].map(lambda x: sig(x, 3))
    cols = ["branch", "U_V1_V", "U_r_V", "I_mA", "R_diff_kOhm"]
    write_processed(res, "discharge_iv_processed.csv", cols)
    # отдельные файлы ветвей -- для pgfplots
    for branch in ("up", "down"):
        write_processed(res[res["branch"] == branch],
                        f"discharge_iv_{branch}.csv", cols)


# --------------------------------------------------------------------------
# 2. Зондовые характеристики
# --------------------------------------------------------------------------
def tanh_fit(u: np.ndarray, i: np.ndarray, iin0: float, te0: float):
    """Контрольная аппроксимация I = I_iн th(U / 2T_e) + A U  (T_e в вольтах)."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return np.nan, np.nan

    def model(x, iin, te, a):
        return iin * np.tanh(x / (2 * te)) + a * x

    try:
        p, cov = curve_fit(model, u, i, p0=[iin0, te0, 0.0], maxfev=20000)
        return p[1], np.sqrt(cov[1, 1])
    except (RuntimeError, ValueError):
        return np.nan, np.nan


def process_one_probe(ip: float, part: pd.DataFrame, params: dict):
    """Обработка одной зондовой характеристики при токе разряда ip (мА)."""
    u = part["U_V"].to_numpy(float)
    i = part["I_uA"].to_numpy(float)
    order = np.argsort(u)
    u, i = u[order], i[order]
    u_max = np.nanmax(np.abs(u))

    u_asym = get_param(params, "U_asym", 0.5 * u_max)
    u_lin = get_param(params, "U_lin", 0.1 * u_max)

    # а) асимптоты при U -> +U_max и U -> -U_max; их пересечения с осью U = 0
    pos, neg = u > u_asym, u < -u_asym
    k_p, sk_p, b_p, sb_p = lsq(u[pos], i[pos])
    k_n, sk_n, b_n, sb_n = lsq(u[neg], i[neg])
    i_off = 0.5 * (b_p + b_n)                    # смещение нуля тока
    iin = 0.5 * (b_p - b_n)
    s_iin = 0.5 * np.hypot(sb_p, sb_n)

    # центрирование: сдвиг по току на i_off и по напряжению так, чтобы I(0) = 0
    i_c = i - i_off
    u_off = np.interp(0.0, i_c, u) if np.all(np.diff(i_c) > 0) else 0.0
    u_c = u - u_off

    # б) наклон в начале координат
    lin = np.abs(u_c) < u_lin
    k0, sk0, b0, sb0 = lsq(u_c[lin], i_c[lin])   # мкА/В

    # T_e: k T_e = e I_iн / (2 dI/dU)  ->  в эВ: I_iн[мкА] / (2 k0[мкА/В])
    te = iin / (2 * k0)
    s_te = te * np.hypot(s_iin / iin, sk0 / k0)
    te_fit, s_te_fit = tanh_fit(u_c, i_c, iin, te if np.isfinite(te) else 3.0)

    print(f"  I_р = {ip:g} мА: U_max = {u_max:g} В, асимптоты |U| > {u_asym:g} В, "
          f"наклон по |U| < {u_lin:g} В")
    print(f"    смещение: dI = {i_off:.3g} мкА, dU = {u_off:.3g} В; "
          f"наклоны асимптот {k_p:.3g}, {k_n:.3g} мкА/В")

    centered = pd.DataFrame({"Ip_mA": ip, "U_c_V": u_c, "I_c_uA": i_c})
    row = {
        "Ip_mA": ip,
        "Iin_uA": iin, "sIin_uA": s_iin,
        "dIdU_uA_V": k0, "sdIdU_uA_V": sk0,
        "Te_eV": te, "sTe_eV": s_te,
        "Te_fit_eV": te_fit, "sTe_fit_eV": s_te_fit,
        "asym_slope_pos": k_p, "asym_slope_neg": k_n,
        "U_lin_V": u_lin, "U_asym_V": u_asym,
        # для графиков plot_data.py (в центрированных координатах)
        "asym_b_pos": b_p - i_off + k_p * u_off, "asym_b_neg": b_n - i_off + k_n * u_off,
        "U_off_V": u_off, "I_off_uA": i_off,
    }
    return centered, row


def plasma_parameters(row: dict, params: dict) -> dict:
    """п. 11-14: n_e, w_p, r_De, r_D, N_D, alpha по I_iн и T_e."""
    d = warn_default(params, "d_probe", np.nan, "диаметр зонда") * 1e-3
    l = warn_default(params, "l_probe", np.nan, "длина зонда") * 1e-3
    sd = params.get("d_probe__sigma", np.nan)
    a_ion = get_param(params, "A_ion", 22.0)
    p_gas = get_param(params, "P_gas", 2.0) * TORR
    t_gas = get_param(params, "T_gas", 300.0)

    s = np.pi * d * l                         # м^2
    m_i = a_ion * AMU
    iin = row["Iin_uA"] * 1e-6                # А
    kte = row["Te_eV"] * E                    # Дж

    n = iin / (0.4 * E * s * np.sqrt(2 * kte / m_i))       # м^-3
    rel_n = np.hypot(row["sIin_uA"] / row["Iin_uA"],
                     0.5 * row["sTe_eV"] / row["Te_eV"])
    if np.isfinite(sd):
        rel_n = np.hypot(rel_n, sd / (d * 1e3))
    wp = np.sqrt(n * E**2 / (EPS0 * M_E))                  # рад/с
    r_de = np.sqrt(EPS0 * kte / (n * E**2))                # м
    r_di = np.sqrt(EPS0 * K_B * t_gas / (n * E**2))        # м
    r_d = (r_de**-2 + r_di**-2) ** -0.5                    # (5.10)
    n_d = 4 / 3 * np.pi * n * r_d**3
    n0 = p_gas / (K_B * t_gas)                             # м^-3
    alpha = n / n0

    return {
        "Ip_mA": row["Ip_mA"],
        "Te_eV": sig(row["Te_eV"]), "sTe_eV": sig(row["sTe_eV"], 2),
        "Te_K": sig(row["Te_eV"] * E / K_B),
        "n_1e10_cm3": sig(n * 1e-6 / 1e10), "sn_1e10_cm3": sig(n * 1e-6 / 1e10 * rel_n, 2),
        "wp_1e9_rad_s": sig(wp / 1e9),
        "rDe_um": sig(r_de * 1e6),
        "rD_um": sig(r_d * 1e6),
        "ND": sig(n_d),
        "alpha_1e7": sig(alpha * 1e7),
    }


def write_family_tex(groups: list[tuple[float, str]]) -> None:
    """Генерирует \\addplot-команды семейства зондовых ВАХ для pgfplots."""
    marks = ["*", "square*", "triangle*", "diamond*", "pentagon*", "o", "square"]
    lines = ["% Сгенерировано scripts/process_data.py -- вручную не править"]
    for k, (ip, fname) in enumerate(groups):
        ip_tex = f"{ip:g}".replace(".", "{,}")
        lines += [
            f"\\addplot+[mark={marks[k % len(marks)]}, mark size=1.5pt, thin]",
            f"    table[col sep=comma, x=U_c_V, y=I_c_uA] {{../data/processed/{fname}}};",
            f"\\addlegendentry{{$I_{{\\text{{р}}}} = {ip_tex}$~мА}}",
        ]
    (PROC_DIR / "probe_family.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  -> data/processed/probe_family.tex")


def process_probes(params: dict) -> None:
    header("2. Зондовые характеристики двойного зонда")
    df = read_raw("probe_iv.csv")
    if df.empty:
        print("  нет данных")
        return

    rows, plasma, groups = [], [], []
    for k, (ip, part) in enumerate(sorted(df.groupby("Ip_mA"), key=lambda g: -g[0])):
        centered, row = process_one_probe(float(ip), part, params)
        rows.append(row)
        fname = f"probe_centered_{k + 1}.csv"
        write_processed(centered.round(5), fname)
        groups.append((float(ip), fname))

        RES.add(f"Iin(Ip={ip:g})", row["Iin_uA"], row["sIin_uA"], "uA", "ионный ток насыщения")
        RES.add(f"dI/dU(Ip={ip:g})", row["dIdU_uA_V"], row["sdIdU_uA_V"], "uA/V", "наклон в нуле")
        RES.add(f"Te(Ip={ip:g})", row["Te_eV"], row["sTe_eV"], "eV", "по (5.26)")
        RES.add(f"Te_fit(Ip={ip:g})", row["Te_fit_eV"], row["sTe_fit_eV"], "eV",
                "контроль: МНК по I = I_iн th(U/2T_e) + AU")
        plasma.append(plasma_parameters(row, params))

    write_family_tex(groups)

    res = pd.DataFrame(rows)
    table = pd.DataFrame({
        "Ip_mA": res["Ip_mA"],
        "Iin_uA": res["Iin_uA"].map(sig), "sIin_uA": res["sIin_uA"].map(lambda x: sig(x, 2)),
        "dIdU_uA_V": res["dIdU_uA_V"].map(sig), "sdIdU_uA_V": res["sdIdU_uA_V"].map(lambda x: sig(x, 2)),
        "Te_eV": res["Te_eV"].map(sig), "sTe_eV": res["sTe_eV"].map(lambda x: sig(x, 2)),
        "Te_fit_eV": res["Te_fit_eV"].map(sig),
    })
    write_processed(table, "probe_results.csv")
    pl = pd.DataFrame(plasma)
    write_processed(pl, "plasma_params.csv")

    print("\n  Параметры плазмы:")
    print(pl.to_string(index=False))


def main() -> None:
    params = read_params()
    # погрешности параметров установки (столбец sigma) -- в params как <name>__sigma
    raw = read_raw("setup_params.csv")
    if not raw.empty and "sigma" in raw.columns:
        for _, r in raw.iterrows():
            try:
                params[f"{str(r['param']).strip()}__sigma"] = float(r["sigma"])
            except (TypeError, ValueError):
                pass

    process_discharge(params)
    process_probes(params)
    RES.save()
    print("\nГотово.")


if __name__ == "__main__":
    main()
