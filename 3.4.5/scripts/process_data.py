"""Обработка сырых данных лабораторной работы 3.4.5.

    «Петля гистерезиса (динамический метод)»

Читает   data/raw/*.csv
Пишет    data/processed/*.csv        -- таблицы для отчёта (csvsimple / pgfplots)
         data/processed/results.tex  -- макросы \\res{...} с итоговыми числами
         data/processed/fit_results.csv
Печатает сводку с промежуточными и окончательными результатами.

Запуск (из каталога работы):
    python scripts/process_data.py

Сырые данные никогда не изменяются. Скрипт работает и на незаполненных
таблицах: тогда в data/processed/ появляются файлы с одними заголовками,
а в отчёте вместо чисел стоят метки ???.

Основные формулы
----------------
Канал X (ток):  M_x = 2*sqrt(2)*I_eff / [2X]            (мА/дел, калибровка п.9)
                теория: M_x = K_x / R0
Напряжённость:  H = N0 * I / L,  L = 2*pi*R             (4.16)
                => h_x = N0 * M_x / L                   (А/м на деление)
Канал Y:        B = tau * U_C / (S * N_и)                (3)
                => b_y = tau * K_y / (S * N_и)          (Тл на деление)
RC-цепочка:     U_out / U_in = 1 / (omega0 * tau)        (5)
Потери:         w = площадь петли = A[дел^2] * h_x * b_y  (Дж/м^3 за цикл)
                P/V = nu0 * w                           (Вт/м^3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab_utils import (
    Results,
    TexResults,
    fmt,
    get_param,
    num_str,
    read_params,
    read_raw,
    write_processed,
)

MU0 = 4e-7 * np.pi
RES = Results()
TEX = TexResults()
N_SAMPLES = 3           # число образцов, под которое свёрстан отчёт


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def num(series_or_value):
    """Безопасное приведение к float (пустые ячейки -> nan)."""
    return pd.to_numeric(series_or_value, errors="coerce")


# --------------------------------------------------------------------------
# Калибровка канала X (п.9)
# --------------------------------------------------------------------------
def process_calib_x(params: dict) -> dict:
    header("Калибровка канала X (п.9)")
    cols = ["Kx", "I_eff_mA", "X2", "Mx_exp", "Mx_th", "dev_percent"]
    df = read_raw("calib_x.csv")
    R0 = get_param(params, "R0", np.nan)
    div_err = get_param(params, "div_err", 0.1)
    eps_I = get_param(params, "eps_ammeter", 0.02)

    calib = {"table": {}, "R0_eff": np.nan, "eps": np.nan, "R0": R0}
    if df.empty:
        write_processed(pd.DataFrame(), "calib_x_processed.csv", cols)
        print("  нет данных; будет использована теория M_x = K_x/R0")
        return calib

    Kx = num(df["Kx"]).to_numpy(float)
    I = num(df["I_eff_mA"]).to_numpy(float)
    X2 = num(df["X2"]).to_numpy(float)
    Mx = 2 * np.sqrt(2) * I / X2                       # мА/дел
    Mx_th = Kx / R0 if np.isfinite(R0) else np.full_like(Kx, np.nan)
    dev = (Mx - Mx_th) / Mx_th * 100

    out = pd.DataFrame({
        "Kx": [num_str(v, sig=3) for v in Kx],
        "I_eff_mA": [num_str(v, sig=3) for v in I],
        "X2": [num_str(v, sig=3) for v in X2],
        "Mx_exp": [num_str(v, sig=3) for v in Mx],
        "Mx_th": [num_str(v, sig=3) for v in Mx_th],
        "dev_percent": [num_str(v, sig=2) for v in dev],
    })
    write_processed(out, "calib_x_processed.csv", cols)

    # для каждого Kx -- среднее M_x
    for k in np.unique(Kx[np.isfinite(Kx)]):
        m = Kx == k
        calib["table"][float(k)] = float(np.nanmean(Mx[m]))
    # «эффективное» R0 по калибровке: K_x / M_x (нужно, если какой-то K_x
    # не калибровался)
    calib["R0_eff"] = float(np.nanmean(Kx / Mx))
    calib["eps"] = float(np.sqrt(eps_I**2 + (div_err / np.nanmin(X2))**2))
    RES.add("R0_eff", calib["R0_eff"], np.nanstd(Kx / Mx), "Ohm",
            "K_x/M_x по калибровке канала X")
    TEX.add("R0eff", calib["R0_eff"], calib["eps"] * calib["R0_eff"])
    TEX.add("epsMx", calib["eps"] * 100, sig=2)
    if np.isfinite(R0):
        TEX.add("R0", R0)
    return calib


def mx_for(Kx: float, calib: dict) -> float:
    """M_x (мА/дел) для данного K_x: калибровка -> пересчёт -> теория."""
    if not np.isfinite(Kx):
        return np.nan
    for k, m in calib["table"].items():
        if np.isclose(k, Kx):
            return m
    if np.isfinite(calib["R0_eff"]):
        return Kx / calib["R0_eff"]
    return Kx / calib["R0"]


# --------------------------------------------------------------------------
# Проверка канала Y (п.10)
# --------------------------------------------------------------------------
def process_calib_y(params: dict) -> None:
    header("Проверка калибровки канала Y (п.10)")
    cols = ["Ky", "U_eff_mV", "Y2", "My_exp", "dev_percent"]
    df = read_raw("calib_y.csv")
    if df.empty:
        write_processed(pd.DataFrame(), "calib_y_processed.csv", cols)
        print("  нет данных")
        return
    Ky = num(df["Ky"]).to_numpy(float)
    U = num(df["U_eff_mV"]).to_numpy(float)
    Y2 = num(df["Y2"]).to_numpy(float)
    My = 2 * np.sqrt(2) * U / Y2
    dev = (My - Ky) / Ky * 100
    out = pd.DataFrame({
        "Ky": [num_str(v) for v in Ky],
        "U_eff_mV": [num_str(v) for v in U],
        "Y2": [num_str(v) for v in Y2],
        "My_exp": [num_str(v) for v in My],
        "dev_percent": [num_str(v, sig=2) for v in dev],
    })
    write_processed(out, "calib_y_processed.csv", cols)
    RES.add("max_dev_Y", np.nanmax(np.abs(dev)), unit="%",
            comment="макс. отклонение M_y от K_y")
    TEX.add("devY", np.nanmax(np.abs(dev)), sig=2)


# --------------------------------------------------------------------------
# Постоянная времени RC-цепочки (пп.11-13)
# --------------------------------------------------------------------------
def process_rc(params: dict) -> tuple[float, float]:
    header("Постоянная времени интегрирующей цепочки (пп.11-13)")
    cols = ["U_in_mV", "U_out_mV", "nu_Hz", "tau_ms"]
    nu0 = get_param(params, "nu0", 50.0)
    R = get_param(params, "R_int", np.nan) * 1e3          # Ом
    C = get_param(params, "C_int", np.nan) * 1e-6         # Ф
    eps_K = get_param(params, "eps_K", 0.03)

    tau_rc = R * C
    if np.isfinite(tau_rc):
        RES.add("tau_RC", tau_rc * 1e3, unit="ms", comment="R_и*C_и")
        TEX.add("tauRC", tau_rc * 1e3)
        TEX.add("Rint", R / 1e3)
        TEX.add("Cint", C * 1e6)

    df = read_raw("rc.csv")
    tau_exp, s_tau = np.nan, np.nan
    if df.empty:
        write_processed(pd.DataFrame(), "rc_processed.csv", cols)
    else:
        Uin = num(df["U_in_mV"]).to_numpy(float)
        Uout = num(df["U_out_mV"]).to_numpy(float)
        nu = num(df["nu_Hz"]).to_numpy(float) if "nu_Hz" in df else np.full_like(Uin, np.nan)
        nu = np.where(np.isfinite(nu), nu, nu0)
        tau = Uin / (2 * np.pi * nu * Uout)                 # с
        write_processed(pd.DataFrame({
            "U_in_mV": [num_str(v) for v in Uin],
            "U_out_mV": [num_str(v) for v in Uout],
            "nu_Hz": [num_str(v) for v in nu],
            "tau_ms": [num_str(v * 1e3) for v in tau],
        }), "rc_processed.csv", cols)
        tau_exp = float(np.nanmean(tau))
        n = np.sum(np.isfinite(tau))
        stat = np.nanstd(tau, ddof=1) / np.sqrt(n) / tau_exp if n > 1 else 0.0
        eps = np.sqrt(stat**2 + 2 * eps_K**2)
        s_tau = eps * tau_exp
        RES.add("tau_exp", tau_exp * 1e3, s_tau * 1e3, "ms", "по формуле (5)")
        TEX.add("tauExp", tau_exp * 1e3, s_tau * 1e3)
        TEX.add("omegaTau", 2 * np.pi * nu0 * tau_exp, sig=2)
        if np.isfinite(tau_rc):
            TEX.add("tauDev", abs(tau_exp - tau_rc) / tau_rc * 100, sig=2)

    # что используем дальше
    if np.isfinite(tau_exp):
        return tau_exp, s_tau / tau_exp
    if np.isfinite(tau_rc):
        print("  [!] tau по эксперименту нет -- используется R_и*C_и")
        TEX.add("omegaTau", 2 * np.pi * nu0 * tau_rc, sig=2)
        return tau_rc, 0.02
    print("  [!] tau неизвестна")
    return np.nan, np.nan


# --------------------------------------------------------------------------
# Предельные петли (пп.3-5, 14-16)
# --------------------------------------------------------------------------
def read_samples() -> pd.DataFrame:
    cols = ["sample", "material", "N0", "N_i", "S_cm2", "L_cm"]
    df = read_raw("samples.csv")
    if df.empty:
        write_processed(pd.DataFrame(), "samples_processed.csv", cols)
        return df
    # в отчёт -- только заполненные строки
    filled = df[pd.to_numeric(df["N0"], errors="coerce").notna()].copy()
    filled["material"] = filled["material"].fillna("")
    write_processed(filled, "samples_processed.csv", cols)
    df["sample"] = num(df["sample"]).astype("Int64")
    return df.set_index("sample")


def coefs(sample_row, Kx, Ky, calib, tau):
    """h_x (А/м/дел) и b_y (Тл/дел) для образца при данных K_x, K_y."""
    N0 = float(num(sample_row["N0"]))
    Ni = float(num(sample_row["N_i"]))
    S = float(num(sample_row["S_cm2"])) * 1e-4
    L = float(num(sample_row["L_cm"])) * 1e-2
    hx = N0 * mx_for(Kx, calib) * 1e-3 / L
    by = tau * Ky * 1e-3 / (S * Ni)
    return hx, by


def process_loops(params, calib, tau, eps_tau, samples) -> None:
    header("Предельные петли гистерезиса (пп.14-16)")
    cols = ["sample", "material", "hx", "by", "Hs", "dHs", "Bs", "dBs",
            "Hc", "dHc", "Br", "dBr", "w", "P"]
    df = read_raw("loops.csv")
    div_err = get_param(params, "div_err", 0.1)
    eps_K = get_param(params, "eps_K", 0.03)
    nu0 = get_param(params, "nu0", 50.0)
    eps_hx = calib["eps"] if np.isfinite(calib["eps"]) else eps_K
    eps_by = np.sqrt(eps_K**2 + (eps_tau if np.isfinite(eps_tau) else 0) ** 2)

    rows = []
    for _, r in df.iterrows():
        s = int(num(r["sample"])) if np.isfinite(num(r["sample"])) else None
        if s is None or samples.empty or s not in samples.index:
            continue
        X2s, Y2s = num(r["X2s"]), num(r["Y2s"])
        if not (np.isfinite(X2s) and np.isfinite(Y2s)):
            continue
        sr = samples.loc[s]
        hx_s, by_s = coefs(sr, num(r["Kx_s"]), num(r["Ky_s"]), calib, tau)
        Kx_c = num(r["Kx_c"]) if np.isfinite(num(r["Kx_c"])) else num(r["Kx_s"])
        Ky_r = num(r["Ky_r"]) if np.isfinite(num(r["Ky_r"])) else num(r["Ky_s"])
        hx_c, _ = coefs(sr, Kx_c, 1.0, calib, tau)
        _, by_r = coefs(sr, 1.0, Ky_r, calib, tau)

        Hs, dHs = hx_s * X2s / 2, hx_s * X2s / 2 * np.hypot(eps_hx, div_err / X2s)
        Bs, dBs = by_s * Y2s / 2, by_s * Y2s / 2 * np.hypot(eps_by, div_err / Y2s)
        X2c, Y2r = num(r["X2c"]), num(r["Y2r"])
        Hc, dHc = hx_c * X2c / 2, hx_c * X2c / 2 * np.hypot(eps_hx, div_err / X2c)
        Br, dBr = by_r * Y2r / 2, by_r * Y2r / 2 * np.hypot(eps_by, div_err / Y2r)
        A = num(r["area_div2"]) if "area_div2" in r else np.nan
        w = A * hx_s * by_s
        P = nu0 * w

        material = str(sr.get("material", "") or "").strip()
        if material.lower() == "nan":
            material = ""
        rows.append({
            "sample": s, "material": material,
            "hx": num_str(hx_s), "by": num_str(by_s, sig=3),
            "Hs": num_str(Hs, dHs), "dHs": num_str(dHs, dHs),
            "Bs": num_str(Bs, dBs), "dBs": num_str(dBs, dBs),
            "Hc": num_str(Hc, dHc), "dHc": num_str(dHc, dHc),
            "Br": num_str(Br, dBr), "dBr": num_str(dBr, dBr),
            "w": num_str(w), "P": num_str(P),
        })
        print(f"\n  образец {s} ({material}):  h_x = {hx_s:.4g} А/м/дел, "
              f"b_y = {by_s:.4g} Тл/дел")
        RES.add(f"Hs_{s}", Hs, dHs, "A/m")
        RES.add(f"Bs_{s}", Bs, dBs, "T")
        RES.add(f"Hc_{s}", Hc, dHc, "A/m")
        RES.add(f"Br_{s}", Br, dBr, "T")
        TEX.add(f"Hs{s}", Hs, dHs)
        TEX.add(f"Bs{s}", Bs, dBs)
        TEX.add(f"Hc{s}", Hc, dHc)
        TEX.add(f"Br{s}", Br, dBr)
        TEX.add(f"Ieff{s}", float(num(r["I_eff_mA"])))
        TEX.add_text(f"mat{s}", material)
        if np.isfinite(w):
            RES.add(f"w_{s}", w, unit="J/m^3", comment="потери за цикл")
            TEX.add(f"w{s}", w)
            TEX.add(f"P{s}", P / 1e3)

    write_processed(pd.DataFrame(rows), "loops_processed.csv", cols)


# --------------------------------------------------------------------------
# Начальные кривые намагничивания (п.6, п.17)
# --------------------------------------------------------------------------
def process_initial(params, calib, tau, samples) -> None:
    header("Начальные кривые намагничивания (п.17)")
    df = read_raw("initial_curve.csv")
    cols_c = ["H", "B", "mu"]
    cols_d = ["H", "mu_diff"]
    summary = []

    for s in range(1, N_SAMPLES + 1):
        sub = df[num(df["sample"]) == s] if not df.empty else pd.DataFrame()
        if sub.empty or samples.empty or s not in samples.index:
            write_processed(pd.DataFrame(), f"initial_{s}.csv", cols_c)
            write_processed(pd.DataFrame(), f"mudiff_{s}.csv", cols_d)
            continue
        sr = samples.loc[s]
        H, B = [], []
        for _, r in sub.iterrows():
            hx, by = coefs(sr, num(r["Kx"]), num(r["Ky"]), calib, tau)
            H.append(hx * num(r["X2"]) / 2)
            B.append(by * num(r["Y2"]) / 2)
        H, B = np.array(H, float), np.array(B, float)
        m = np.isfinite(H) & np.isfinite(B) & (H > 0)
        H, B = H[m], B[m]
        order = np.argsort(H)
        H, B = np.r_[0.0, H[order]], np.r_[0.0, B[order]]   # кривая из начала
        mu = np.where(H > 0, B / (MU0 * np.where(H > 0, H, 1)), np.nan)
        write_processed(pd.DataFrame({
            "H": [num_str(v, sig=4) for v in H],
            "B": [num_str(v, sig=4) for v in B],
            "mu": [num_str(v, sig=3) for v in mu],
        }), f"initial_{s}.csv", cols_c)

        Hm = (H[1:] + H[:-1]) / 2
        mud = np.diff(B) / np.diff(H) / MU0
        write_processed(pd.DataFrame({
            "H": [num_str(v, sig=4) for v in Hm],
            "mu_diff": [num_str(v, sig=3) for v in mud],
        }), f"mudiff_{s}.csv", cols_d)

        mu_init = mu[1] if len(mu) > 1 else np.nan        # по первой точке
        k = int(np.nanargmax(mud)) if np.any(np.isfinite(mud)) else None
        mu_max = mud[k] if k is not None else np.nan
        H_at = Hm[k] if k is not None else np.nan
        RES.add(f"mu_init_{s}", mu_init, comment="B/(mu0 H) в первой точке")
        RES.add(f"mu_diff_max_{s}", mu_max, comment=f"при H ~ {H_at:.3g} А/м")
        TEX.add(f"muInit{s}", mu_init, sig=2)
        TEX.add(f"muMax{s}", mu_max, sig=2)
        TEX.add(f"HmuMax{s}", H_at, sig=2)
        TEX.add(f"Hmax{s}", np.nanmax(H), sig=3)
        TEX.add(f"Bmax{s}", np.nanmax(B), sig=3)
        summary.append(s)

    if not summary:
        print("  нет данных")


def main() -> None:
    params = read_params()
    calib = process_calib_x(params)
    process_calib_y(params)
    tau, eps_tau = process_rc(params)
    samples = read_samples()
    process_loops(params, calib, tau, eps_tau, samples)
    process_initial(params, calib, tau, samples)
    header("Сохранение итогов")
    RES.save()
    TEX.save()


if __name__ == "__main__":
    main()
