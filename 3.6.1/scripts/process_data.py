"""Обработка сырых данных лабораторной работы 3.6.1.

    «Спектральный анализ электрических сигналов»

Читает   data/raw/*.csv
Пишет    data/processed/*.csv  (в том числе fit_results.csv со всеми итогами)
Печатает сводку с промежуточными и окончательными результатами.

Запуск (из каталога работы):
    python scripts/process_data.py

Сырые данные никогда не изменяются.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab_utils import (
    Results,
    fmt,
    get_param,
    lsq,
    lsq_origin,
    read_params,
    read_raw,
    write_processed,
)

RES = Results()


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------
# А. Спектр периодической последовательности прямоугольных импульсов
# --------------------------------------------------------------------------
def process_harmonics(params: dict) -> None:
    """п.А.3: частоты и амплитуды гармоник, сравнение с теорией.

    nu_n = n/T,  |a_n/a_1| = |sin(pi*n*tau/T)| / (n*|sin(pi*tau/T)|).
    """
    header("А. Гармоники прямоугольного сигнала")
    cols = ["n", "nu_exp_kHz", "nu_theor_kHz", "a_exp_mV",
            "ratio_exp", "ratio_theor", "dev_percent"]

    df = read_raw("harmonics.csv")
    T_ms = get_param(params, "T_rect", 1.0)
    tau_us = get_param(params, "tau_rect", 50.0)
    a1 = get_param(params, "a1_rect", np.nan)
    print(f"  T = {T_ms} мс, tau = {tau_us} мкс, a_1 = {a1} мВ")

    n = df["n"].to_numpy(dtype=float)
    a = df["a_exp_mV"].to_numpy(dtype=float)
    q = np.pi * tau_us / (1000 * T_ms)

    out = pd.DataFrame({
        "n": n.astype(int),
        "nu_exp_kHz": df["nu_exp_kHz"],
        "nu_theor_kHz": n / T_ms,
        "a_exp_mV": a,
        "ratio_exp": a / a1,
        "ratio_theor": np.abs(np.sin(q * n)) / (n * np.abs(np.sin(q))),
    })
    out["dev_percent"] = (out["ratio_exp"] - out["ratio_theor"]) / out["ratio_theor"] * 100
    out = out.round({"nu_theor_kHz": 0, "ratio_exp": 3, "ratio_theor": 3, "dev_percent": 1})
    out["nu_theor_kHz"] = out["nu_theor_kHz"].astype(int)

    print(out.to_string(index=False))
    write_processed(out, "harmonics_processed.csv", cols)

    nu_dev = np.abs(out["nu_exp_kHz"] - out["nu_theor_kHz"]) / out["nu_theor_kHz"] * 100
    RES.add("макс. отклонение nu_n от n/T", float(nu_dev.max()), np.nan, "%")
    RES.add("макс. |отклонение| |a_n/a_1| (n <= 15)",
            float(np.abs(out.loc[out["n"] <= 15, "dev_percent"]).max()), np.nan, "%")


def process_width(params: dict) -> None:
    """п.А.4: ширина спектра Dnu в зависимости от 1/tau при T = const."""
    header("А. Ширина спектра Dnu(1/tau)")
    cols = ["tau_us", "inv_tau_kHz", "Dnu_kHz", "Dnu_tau"]

    df = read_raw("spectrum_width.csv")
    tau = df["tau_us"].to_numpy(dtype=float)                  # мкс
    dnu = df["Dnu_kHz"].to_numpy(dtype=float)
    out = pd.DataFrame({
        "tau_us": tau.astype(int),
        "inv_tau_kHz": np.round(1000.0 / tau, 2),             # 1/мкс -> кГц
        "Dnu_kHz": dnu,
        "Dnu_tau": np.round(dnu * tau / 1000.0, 3),
    })
    print(out.to_string(index=False))
    write_processed(out, "spectrum_width_processed.csv", cols)

    x = 1000.0 / tau
    k0, sk0 = lsq_origin(x, dnu)
    k, sk, b, sb = lsq(x, dnu)
    RES.add("k_Dnu (Dnu = k/tau, через 0)", k0, sk0, "", "теория: k = 1")
    RES.add("k_Dnu (Dnu = k/tau + b)", k, sk, "")
    RES.add("b_Dnu", b, sb, "кГц")


def process_spacing(params: dict) -> None:
    """п.А.5: расстояние между гармониками dnu в зависимости от 1/T при tau = const."""
    header("А. Расстояние между гармониками dnu(1/T)")
    cols = ["T_ms", "inv_T_kHz", "dnu_kHz", "dnu_T"]

    df = read_raw("harmonics_spacing.csv")
    nu_rep = df["nu_rep_kHz"].to_numpy(dtype=float)           # 1/T, кГц
    dnu = df["dnu_kHz"].to_numpy(dtype=float)
    out = pd.DataFrame({
        "T_ms": np.round(1.0 / nu_rep, 3),
        "inv_T_kHz": nu_rep,
        "dnu_kHz": dnu,
        "dnu_T": np.round(dnu / nu_rep, 3),
    })
    print(out.to_string(index=False))
    write_processed(out, "harmonics_spacing_processed.csv", cols)

    k0, sk0 = lsq_origin(nu_rep, dnu)
    k, sk, b, sb = lsq(nu_rep, dnu)
    RES.add("k_dnu (dnu = k/T, через 0)", k0, sk0, "", "теория: k = 1")
    RES.add("k_dnu (dnu = k/T + b)", k, sk, "")
    RES.add("b_dnu", b, sb, "кГц")


# --------------------------------------------------------------------------
# Б. Цуги
# --------------------------------------------------------------------------
def process_bursts(params: dict) -> None:
    """Часть Б: ширина спектра цугов Dnu = 1/tau, tau = N/nu0; dnu = 1/T."""
    header("Б. Спектр последовательности цугов")
    cols = ["nu0_kHz", "T_ms", "N", "tau_us", "dnu_kHz", "dnu_theor_kHz",
            "Dnu_kHz", "Dnu_theor_kHz", "Dnu_tau"]

    df = read_raw("bursts.csv")
    nu0 = df["nu0_kHz"].to_numpy(dtype=float)
    N = df["N"].to_numpy(dtype=float)
    T = df["T_ms"].to_numpy(dtype=float)
    tau_ms = N / nu0
    Dnu = df["Dnu_kHz"].to_numpy(dtype=float)

    out = pd.DataFrame({
        "series": df["series"],
        "nu0_kHz": nu0.astype(int),
        "T_ms": T,
        "N": N.astype(int),
        "tau_us": np.round(tau_ms * 1000, 1),
        "dnu_kHz": df["dnu_kHz"],
        "dnu_theor_kHz": 1.0 / T,
        "Dnu_kHz": Dnu,
        "Dnu_theor_kHz": np.round(1.0 / tau_ms, 2),
        "Dnu_tau": np.round(Dnu * tau_ms, 2),
    })
    print(out.to_string(index=False))

    write_processed(out[out["series"] == "N"], "bursts_N_processed.csv", cols)
    write_processed(out[out["series"] == "nu0"], "bursts_nu0_processed.csv", cols)

    prod = out["Dnu_tau"].to_numpy(dtype=float)
    RES.add("Dnu*tau (цуги, среднее)", float(prod.mean()),
            float(prod.std(ddof=1) / np.sqrt(len(prod))), "", "теория: 1")


# --------------------------------------------------------------------------
# Г. Амплитудная модуляция
# --------------------------------------------------------------------------
def process_am(params: dict) -> None:
    """Часть Г: m = (Amax-Amin)/(Amax+Amin) и зависимость a_бок/a_осн от m."""
    header("Г. Амплитудно-модулированный сигнал")
    cols = ["m_set_percent", "m", "a_side_mV", "ratio", "ratio_theor"]

    amax = get_param(params, "A_max_am", np.nan)
    amin = get_param(params, "A_min_am", np.nan)
    m_check = get_param(params, "m_check", 50.0) / 100
    m_exp = (amax - amin) / (amax + amin)
    RES.add("m по A_max, A_min", m_exp, np.nan, "", f"на генераторе m = {m_check:g}")

    a_main = get_param(params, "a_main_am", np.nan)
    df = read_raw("am_modulation.csv")
    m = df["m_set_percent"].to_numpy(dtype=float) / 100
    a_side = df["a_side_mV"].to_numpy(dtype=float)
    ratio = a_side / a_main

    out = pd.DataFrame({
        "m_set_percent": m * 100,
        "m": m,
        "a_side_mV": a_side,
        "ratio": np.round(ratio, 4),
        "ratio_theor": m / 2,
    })
    out["m_set_percent"] = out["m_set_percent"].astype(int)
    print(out.to_string(index=False))
    write_processed(out, "am_processed.csv", cols)

    k0, sk0 = lsq_origin(m, ratio)
    k, sk, b, sb = lsq(m, ratio)
    RES.add("k_AM (через 0)", k0, sk0, "", "теория: k = 0.5")
    RES.add("k_AM (y=kx+b)", k, sk, "")
    RES.add("b_AM", b, sb, "")


# --------------------------------------------------------------------------
def main() -> None:
    params = read_params()
    print("Параметры установки из data/raw/setup_params.csv:")
    print("  " + ", ".join(f"{k}={v}" for k, v in params.items()))

    process_harmonics(params)
    process_width(params)
    process_spacing(params)
    process_bursts(params)
    process_am(params)

    header("Итоговые результаты -> data/processed/fit_results.csv")
    RES.save()


if __name__ == "__main__":
    main()
