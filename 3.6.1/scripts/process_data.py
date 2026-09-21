"""Обработка сырых данных лабораторной работы 3.6.1.

    «Спектральный анализ электрических сигналов»

Читает   data/raw/*.csv
Пишет    data/processed/*.csv  (в том числе fit_results.csv со всеми итогами)
Печатает сводку с промежуточными и окончательными результатами.

Запуск (из каталога работы):
    python scripts/process_data.py

Сырые данные никогда не изменяются. Скрипт корректно отрабатывает и на пустых
таблицах: в этом случае обработанные файлы создаются с одним заголовком,
а в сводке выводится «нет данных».
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

# Теоретическое значение произведения ширин для гауссова импульса,
# если обе ширины измеряются по полувысоте АМПЛИТУДЫ: 4*ln2/pi.
GAUSS_PRODUCT = 4 * np.log(2) / np.pi          # ~ 0.8825
# То же для прямоугольного импульса (ширина спектра по полувысоте): 2*1.8955/pi.
RECT_PRODUCT_FWHM = 2 * 1.89549 / np.pi        # ~ 1.2067

RES = Results()


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------
# А. Спектр периодической последовательности прямоугольных импульсов
# --------------------------------------------------------------------------
def process_harmonics(params: dict) -> None:
    """п.7: частоты и амплитуды гармоник, сравнение с теорией.

    nu_n = n/T,   |a_n| = (tau/T) * |sin(pi*nu_n*tau)| / (pi*nu_n*tau),
    сравниваются относительные величины |a_n/a_1|.
    """
    header("А. Гармоники прямоугольного сигнала (п.7)")
    cols = ["n", "nu_exp_kHz", "nu_theor_kHz", "a_exp_mV",
            "ratio_exp", "sigma_ratio_exp", "ratio_theor", "dev_percent"]

    df = read_raw("harmonics.csv")
    T_ms = get_param(params, "T_rect", 1.0)          # мс
    tau_us = get_param(params, "tau_rect", 50.0)     # мкс
    print(f"  T = {T_ms} мс, tau = {tau_us} мкс, tau/T = {tau_us / (1000 * T_ms):.3f}")

    if df.empty:
        print("  нет данных")
        write_processed(pd.DataFrame(), "harmonics_processed.csv", cols)
        return

    n = df["n"].to_numpy(dtype=float)
    out = pd.DataFrame({"n": n.astype(int)})
    out["nu_exp_kHz"] = df["nu_exp_kHz"]
    out["nu_theor_kHz"] = n / T_ms                                   # 1/мс = кГц
    out["a_exp_mV"] = df["a_exp_mV"]

    a1 = float(df.loc[df["n"] == 1, "a_exp_mV"].iloc[0]) if (df["n"] == 1).any() \
        else float(df["a_exp_mV"].iloc[0])
    sigma_a = df.get("sigma_a_mV", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    a = df["a_exp_mV"].to_numpy(dtype=float)

    out["ratio_exp"] = a / a1
    # погрешность отношения: относительные погрешности складываются квадратично
    sigma_a1 = sigma_a[df["n"] == 1][0] if (df["n"] == 1).any() else np.nan
    out["sigma_ratio_exp"] = np.abs(a / a1) * np.sqrt(
        (sigma_a / a) ** 2 + (sigma_a1 / a1) ** 2)

    # теория: |a_n/a_1| = |sin(pi*n*tau/T)| / (n * |sin(pi*tau/T)|)
    q = np.pi * tau_us / (1000 * T_ms)
    out["ratio_theor"] = np.abs(np.sin(q * n)) / (n * np.abs(np.sin(q)))
    out["dev_percent"] = (out["ratio_exp"] - out["ratio_theor"]) / out["ratio_theor"] * 100

    out = out.round(4)
    print(out.to_string(index=False))
    write_processed(out, "harmonics_processed.csv", cols)


def process_width(params: dict) -> None:
    """п.8: ширина спектра Dnu в зависимости от 1/tau при T = const."""
    header("А. Ширина спектра Dnu(1/tau) (п.8)")
    cols = ["tau_us", "inv_tau_kHz", "sigma_inv_tau_kHz", "Dnu_kHz", "sigma_Dnu_kHz"]

    df = read_raw("spectrum_width.csv")
    if df.empty:
        print("  нет данных")
        write_processed(pd.DataFrame(), "spectrum_width_processed.csv", cols)
        RES.add("k_Dnu (наклон Dnu(1/tau))", np.nan, np.nan, "")
        return

    tau = df["tau_us"].to_numpy(dtype=float)                 # мкс
    sigma_tau = df.get("sigma_tau_us", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    inv_tau = 1000.0 / tau                                   # 1/мкс -> кГц
    out = pd.DataFrame({
        "tau_us": tau,
        "inv_tau_kHz": inv_tau,
        "sigma_inv_tau_kHz": inv_tau * sigma_tau / tau,
        "Dnu_kHz": df["Dnu_kHz"],
        "sigma_Dnu_kHz": df.get("sigma_Dnu_kHz", np.nan),
    }).round(4)

    print(out.to_string(index=False))
    write_processed(out, "spectrum_width_processed.csv", cols)

    k0, sk0 = lsq_origin(out["inv_tau_kHz"], out["Dnu_kHz"])
    k, sk, b, sb = lsq(out["inv_tau_kHz"], out["Dnu_kHz"])
    print("  МНК (через 0):", fmt(k0, sk0))
    print("  МНК (y=kx+b): ", fmt(k, sk), " b =", fmt(b, sb, "кГц"))
    RES.add("k_Dnu (Dnu = k/tau, через 0)", k0, sk0, "", "теория: k = 1")
    RES.add("k_Dnu (Dnu = k/tau + b)", k, sk, "")
    RES.add("b_Dnu", b, sb, "кГц")


def process_spacing(params: dict) -> None:
    """п.9: расстояние между гармониками dnu в зависимости от 1/T при tau = const."""
    header("А. Расстояние между гармониками dnu(1/T) (п.9)")
    cols = ["T_ms", "inv_T_kHz", "sigma_inv_T_kHz", "dnu_kHz", "sigma_dnu_kHz"]

    df = read_raw("harmonics_spacing.csv")
    if df.empty:
        print("  нет данных")
        write_processed(pd.DataFrame(), "harmonics_spacing_processed.csv", cols)
        RES.add("k_dnu (наклон dnu(1/T))", np.nan, np.nan, "")
        return

    T = df["T_ms"].to_numpy(dtype=float)                      # мс
    sigma_T = df.get("sigma_T_ms", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    m = df.get("m", pd.Series(1.0, index=df.index)).fillna(1.0).to_numpy(dtype=float)
    nu_n = df["nu_n_kHz"].to_numpy(dtype=float)
    nu_nm = df["nu_nm_kHz"].to_numpy(dtype=float)
    sigma_nu = df.get("sigma_nu_kHz", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)

    inv_T = 1.0 / T                                           # 1/мс = кГц
    out = pd.DataFrame({
        "T_ms": T,
        "inv_T_kHz": inv_T,
        "sigma_inv_T_kHz": inv_T * sigma_T / T,
        "dnu_kHz": (nu_nm - nu_n) / m,
        "sigma_dnu_kHz": np.sqrt(2) * sigma_nu / m,
    }).round(4)

    print(out.to_string(index=False))
    write_processed(out, "harmonics_spacing_processed.csv", cols)

    k0, sk0 = lsq_origin(out["inv_T_kHz"], out["dnu_kHz"])
    k, sk, b, sb = lsq(out["inv_T_kHz"], out["dnu_kHz"])
    print("  МНК (через 0):", fmt(k0, sk0))
    print("  МНК (y=kx+b): ", fmt(k, sk), " b =", fmt(b, sb, "кГц"))
    RES.add("k_dnu (dnu = k/T, через 0)", k0, sk0, "", "теория: k = 1")
    RES.add("k_dnu (dnu = k/T + b)", k, sk, "")
    RES.add("b_dnu", b, sb, "кГц")


# --------------------------------------------------------------------------
# Б. Цуги
# --------------------------------------------------------------------------
def process_bursts(params: dict) -> None:
    """пп.13-14: теорема смещения и соотношения неопределённостей для цугов."""
    header("Б. Спектр последовательности цугов (пп.13-14)")
    cols = ["nu0_kHz", "T_ms", "N", "tau_ms", "nu_center_kHz", "Dnu_kHz",
            "Dnu_theor_kHz", "dnu_kHz", "dnu_theor_kHz", "Dnu_tau"]

    df = read_raw("bursts.csv")
    if df.empty:
        print("  нет данных")
        write_processed(pd.DataFrame(), "bursts_processed.csv", cols)
        return

    nu0 = df["nu0_kHz"].to_numpy(dtype=float)
    T = df["T_ms"].to_numpy(dtype=float)
    N = df["N"].to_numpy(dtype=float)
    tau = N / nu0                                     # мс (N периодов несущей)

    out = pd.DataFrame({
        "nu0_kHz": nu0,
        "T_ms": T,
        "N": N,
        "tau_ms": tau,
        "nu_center_kHz": df["nu_center_kHz"],
        "Dnu_kHz": df["Dnu_kHz"],
        "Dnu_theor_kHz": 2.0 / tau,                   # полная ширина 2/tau
        "dnu_kHz": df["dnu_kHz"],
        "dnu_theor_kHz": 1.0 / T,
        "Dnu_tau": df["Dnu_kHz"].to_numpy(dtype=float) * tau,
    }).round(4)

    print(out.to_string(index=False))
    write_processed(out, "bursts_processed.csv", cols)
    RES.add("Dnu*tau (цуги, среднее)", float(np.nanmean(out["Dnu_tau"])),
            float(np.nanstd(out["Dnu_tau"], ddof=1)) if len(out) > 1 else np.nan,
            "", "теория: 2")


# --------------------------------------------------------------------------
# В. Гауссианы
# --------------------------------------------------------------------------
def process_gauss(params: dict) -> None:
    """п.17: соотношение неопределённостей для гауссова импульса."""
    header("В. Спектр последовательности гауссианов (п.17)")
    cols = ["nu0_kHz", "T_ms", "tau_ms", "Dnu_kHz", "product", "sigma_product"]

    df = read_raw("gauss.csv")
    print(f"  теория (по полувысоте амплитуды): Dnu*tau = 4ln2/pi = {GAUSS_PRODUCT:.4f}")
    print(f"  для прямоугольного импульса (та же методика): {RECT_PRODUCT_FWHM:.4f}")
    if df.empty:
        print("  нет данных")
        write_processed(pd.DataFrame(), "gauss_processed.csv", cols)
        RES.add("Dnu*tau (гауссианы)", np.nan, np.nan, "")
        return

    tau = df["tau_ms"].to_numpy(dtype=float)
    dnu = df["Dnu_kHz"].to_numpy(dtype=float)
    s_tau = df.get("sigma_tau_ms", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    s_dnu = df.get("sigma_Dnu_kHz", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)

    product = dnu * tau
    out = pd.DataFrame({
        "nu0_kHz": df["nu0_kHz"],
        "T_ms": df["T_ms"],
        "tau_ms": tau,
        "Dnu_kHz": dnu,
        "product": product,
        "sigma_product": product * np.sqrt((s_tau / tau) ** 2 + (s_dnu / dnu) ** 2),
    }).round(4)

    print(out.to_string(index=False))
    write_processed(out, "gauss_processed.csv", cols)
    RES.add("Dnu*tau (гауссианы, среднее)", float(np.nanmean(product)),
            float(np.nanstd(product, ddof=1)) if len(product) > 1 else np.nan,
            "", f"теория: {GAUSS_PRODUCT:.3f}")


# --------------------------------------------------------------------------
# Г. Амплитудная модуляция
# --------------------------------------------------------------------------
def process_am(params: dict) -> None:
    """пп.20-23: m = (Amax-Amin)/(Amax+Amin) и зависимость a_бок/a_осн от m."""
    header("Г. Амплитудно-модулированный сигнал (пп.20-23)")
    cols = ["m_set", "m_exp", "sigma_m_exp", "m_used", "ratio", "sigma_ratio", "ratio_theor"]

    df = read_raw("am_modulation.csv")
    if df.empty:
        print("  нет данных")
        write_processed(pd.DataFrame(), "am_processed.csv", cols)
        RES.add("k_AM (a_бок/a_осн = k*m)", np.nan, np.nan, "")
        return

    m_set = df["m_set_percent"].to_numpy(dtype=float) / 100.0
    amax = df["A_max_V"].to_numpy(dtype=float)
    amin = df["A_min_V"].to_numpy(dtype=float)
    s_A = df.get("sigma_A_V", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    a_main = df["a_main_mV"].to_numpy(dtype=float)
    a_side = df["a_side_mV"].to_numpy(dtype=float)
    s_a = df.get("sigma_a_mV", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)

    m_exp = (amax - amin) / (amax + amin)
    # sigma_m для m = (A1-A2)/(A1+A2) при одинаковой sigma_A
    sigma_m = 2 * s_A * np.sqrt(amax**2 + amin**2) / (amax + amin) ** 2
    ratio = a_side / a_main
    sigma_ratio = ratio * np.sqrt((s_a / a_side) ** 2 + (s_a / a_main) ** 2)

    # для графика используем измеренное m, а если A_max/A_min не измерялись -- заданное
    m_used = np.where(np.isfinite(m_exp), m_exp, m_set)

    out = pd.DataFrame({
        "m_set": m_set,
        "m_exp": m_exp,
        "sigma_m_exp": sigma_m,
        "m_used": m_used,
        "ratio": ratio,
        "sigma_ratio": sigma_ratio,
        "ratio_theor": m_set / 2,
    }).round(4)

    print(out.to_string(index=False))
    write_processed(out, "am_processed.csv", cols)

    k0, sk0 = lsq_origin(m_used, ratio)
    k, sk, b, sb = lsq(m_used, ratio)
    print("  МНК (через 0):", fmt(k0, sk0))
    print("  МНК (y=kx+b): ", fmt(k, sk), " b =", fmt(b, sb))
    RES.add("k_AM (через 0)", k0, sk0, "", "теория: k = 0.5")
    RES.add("k_AM (y=kx+b)", k, sk, "")


def process_pm(params: dict) -> None:
    """пп.24-25 (*): фазовая модуляция, сравнение с функциями Бесселя."""
    header("Д. Фазовая модуляция (пп.24-25, *)")
    cols = ["phi_m_deg", "phi_m_rad", "ratio1", "ratio2", "ratio1_theor", "ratio2_theor"]

    df = read_raw("pm_modulation.csv")
    if df.empty:
        print("  нет данных (пункт необязательный)")
        write_processed(pd.DataFrame(), "pm_processed.csv", cols)
        return

    from scipy.special import jv

    phi_deg = df["phi_m_deg"].to_numpy(dtype=float)
    phi = np.deg2rad(phi_deg)
    a_main = df["a_main_mV"].to_numpy(dtype=float)
    a1 = df["a_side1_mV"].to_numpy(dtype=float)
    a2 = df.get("a_side2_mV", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)

    out = pd.DataFrame({
        "phi_m_deg": phi_deg,
        "phi_m_rad": phi,
        "ratio1": a1 / a_main,
        "ratio2": a2 / a_main,
        "ratio1_theor": np.abs(jv(1, phi) / jv(0, phi)),
        "ratio2_theor": np.abs(jv(2, phi) / jv(0, phi)),
    }).round(4)

    print(out.to_string(index=False))
    write_processed(out, "pm_processed.csv", cols)


# --------------------------------------------------------------------------
# Е. RC-фильтр
# --------------------------------------------------------------------------
def process_filter(params: dict) -> None:
    """пп.28-29: коэффициент фильтрации K(nu) и постоянная времени tau_RC.

    K(nu) = 1 / sqrt(1 + (2*pi*nu*tau_RC)^2)  =>  1/K^2 = 1 + (2*pi*tau_RC)^2 * nu^2,
    то есть 1/K^2 линейно зависит от nu^2 с наклоном (2*pi*tau_RC)^2.
    """
    header("Е. Фильтрация сигнала RC-цепочкой (пп.28-29)")
    cols = ["nu_kHz", "K", "sigma_K", "K_theor", "nu2_kHz2", "invK2", "sigma_invK2"]

    R = params.get("R_filter", np.nan)          # Ом
    C_nF = params.get("C_filter", np.nan)       # нФ
    tau_rc_direct = R * C_nF * 1e-9 * 1e3 if np.isfinite(R) and np.isfinite(C_nF) else np.nan  # мс
    if np.isfinite(tau_rc_direct):
        print(f"  прямой расчёт: tau_RC = R*C = {tau_rc_direct * 1e3:.3f} мкс, "
              f"nu_RC = 1/tau_RC = {1 / tau_rc_direct:.3f} кГц")
        RES.add("tau_RC (расчёт R*C)", tau_rc_direct * 1e3, np.nan, "мкс")
    else:
        print("  R и/или C не заданы в setup_params.csv -- прямой расчёт пропущен")

    df = read_raw("filter.csv")
    if df.empty:
        print("  нет данных")
        write_processed(pd.DataFrame(), "filter_processed.csv", cols)
        RES.add("tau_RC (по графику 1/K^2(nu^2))", np.nan, np.nan, "мкс")
        return

    nu = df["nu_kHz"].to_numpy(dtype=float)
    af = df["a_filt_mV"].to_numpy(dtype=float)
    a0 = df["a_orig_mV"].to_numpy(dtype=float)
    s_a = df.get("sigma_a_mV", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)

    K = af / a0
    sigma_K = K * np.sqrt((s_a / af) ** 2 + (s_a / a0) ** 2)
    invK2 = 1.0 / K**2
    sigma_invK2 = 2 * invK2 * sigma_K / K

    # МНК: (1/K^2 - 1) = slope * nu^2, наклон = (2*pi*tau_RC)^2 [мс^2]
    slope, sigma_slope = lsq_origin(nu**2, invK2 - 1.0)
    tau_rc = np.sqrt(slope) / (2 * np.pi) if np.isfinite(slope) and slope > 0 else np.nan
    sigma_tau_rc = tau_rc * 0.5 * sigma_slope / slope if np.isfinite(tau_rc) else np.nan

    out = pd.DataFrame({
        "nu_kHz": nu,
        "K": K,
        "sigma_K": sigma_K,
        "K_theor": 1 / np.sqrt(1 + (2 * np.pi * nu * tau_rc) ** 2) if np.isfinite(tau_rc) else np.nan,
        "nu2_kHz2": nu**2,
        "invK2": invK2,
        "sigma_invK2": sigma_invK2,
    }).round(4)

    print(out.to_string(index=False))
    write_processed(out, "filter_processed.csv", cols)

    print("  наклон (2*pi*tau_RC)^2 =", fmt(slope, sigma_slope, "мс^2"))
    RES.add("tau_RC (по графику 1/K^2(nu^2))",
            tau_rc * 1e3 if np.isfinite(tau_rc) else np.nan,
            sigma_tau_rc * 1e3 if np.isfinite(sigma_tau_rc) else np.nan, "мкс")
    if np.isfinite(tau_rc):
        RES.add("nu_1/2 = 1/(2*pi*tau_RC)", 1 / (2 * np.pi * tau_rc), np.nan, "кГц",
                "частота, на которой K = 1/sqrt(2)")
    if np.isfinite(tau_rc) and np.isfinite(tau_rc_direct):
        disc = abs(tau_rc - tau_rc_direct) / tau_rc_direct * 100
        print(f"  расхождение с прямым расчётом R*C: {disc:.1f} %")


# --------------------------------------------------------------------------
def main() -> None:
    params = read_params()
    print("Параметры установки из data/raw/setup_params.csv:")
    print("  " + (", ".join(f"{k}={v}" for k, v in params.items()) or "не заданы"))

    process_harmonics(params)
    process_width(params)
    process_spacing(params)
    process_bursts(params)
    process_gauss(params)
    process_am(params)
    process_pm(params)
    process_filter(params)

    header("Итоговые результаты -> data/processed/fit_results.csv")
    RES.save()
    print("\nГотово. Для построения графиков запустите: python scripts/plot_data.py")


if __name__ == "__main__":
    main()
