"""Обработка сырых данных лабораторной работы 3.1.1 «Магнитометр».

Читает   data/raw/*.csv
Пишет    data/processed/*.csv, data/processed/fit_results.csv,
         data/processed/results.tex (числа для отчёта, подставляются сами)

Запуск (из каталога работы):
    python scripts/process_data.py

Пока данные не заполнены, скрипт создаёт таблицы-заготовки (только с
номерами измерений), а все результаты в отчёте выводятся как ???.
Внутри расчёта всё переводится в СИ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab_utils import (
    Results,
    col,
    get_param,
    header,
    mean_with_error,
    read_params,
    read_raw,
    write_processed,
)

MU0 = 4e-7 * np.pi          # Гн/м
RES = Results()


def rel(value: float, sigma: float) -> float:
    """Относительная погрешность (0, если величина точная или не задана)."""
    if not np.isfinite(value) or value == 0 or not np.isfinite(sigma):
        return 0.0
    return abs(sigma / value)


# --------------------------------------------------------------------------
# А. Горизонтальная составляющая магнитного поля Земли
# --------------------------------------------------------------------------
def process_rod(p: dict) -> dict:
    """Момент инерции стержня, формула (6)."""
    header("А.1. Момент инерции стержня")
    df = read_raw("rod_size.csv")
    s_cal = get_param(p, "sigma_caliper")[0]
    s_cal = s_cal if np.isfinite(s_cal) else 0.0
    l_mm, s_l_mm = mean_with_error(col(df, "l_mm"), s_cal)
    d_mm, s_d_mm = mean_with_error(col(df, "d_mm"), s_cal)
    m_g, s_m_g = get_param(p, "m_rod")

    l, s_l = l_mm * 1e-3, s_l_mm * 1e-3
    r, s_r = d_mm * 0.5e-3, s_d_mm * 0.5e-3
    m, s_m = m_g * 1e-3, s_m_g * 1e-3

    J = m * l**2 / 12 * (1 + 3 * (r / l) ** 2)
    # вклад поправки 3(r/l)^2 мал, её погрешностью пренебрегаем
    s_J = J * np.sqrt(rel(m, s_m) ** 2 + (2 * rel(l, s_l)) ** 2)

    RES.add("lrod", "длина стержня l", l_mm, s_l_mm, "мм")
    RES.add("drod", "диаметр стержня d", d_mm, s_d_mm, "мм")
    RES.add("corr", "поправка 3(r/l)^2", 3 * (r / l) ** 2 * 100, np.nan, "%")
    RES.add("J", "момент инерции J", J * 1e7, s_J * 1e7, "10^-7 кг*м^2")

    write_processed(df, "rod_size_processed.csv", ["trial", "l_mm", "d_mm"])
    return {"J": (J, s_J)}


def process_deflection(p: dict) -> dict:
    """Смещение зайчика x1 в поле стержня, формула (4)."""
    header("А.2. Отклонение стрелки в поле стержня")
    df = read_raw("deflection_rod.csv")
    s_x = get_param(p, "sigma_x")[0]
    s_x = s_x if np.isfinite(s_x) else 0.0

    xp, xm = col(df, "x1_plus_mm"), col(df, "x1_minus_mm")
    df["x1_mm"] = (xp + xm) / 2
    df["dev_percent"] = np.abs(xp - xm) / df["x1_mm"] * 100
    for i, dev in enumerate(df["dev_percent"]):
        if np.isfinite(dev) and dev > 5:
            print(f"  [!] измерение {i + 1}: расхождение {dev:.1f} % > 5 % -- проверить установку")

    x1_mm, s_x1_mm = mean_with_error(np.concatenate([xp, xm]), s_x)
    RES.add("xone", "смещение x1", x1_mm, s_x1_mm, "мм")
    write_processed(df, "deflection_rod_processed.csv",
                    ["trial", "x1_plus_mm", "x1_minus_mm", "x1_mm", "dev_percent"])
    return {"x1": (x1_mm * 1e-3, s_x1_mm * 1e-3)}


def process_period(p: dict) -> dict:
    """Период крутильных колебаний стержня, формула (5)."""
    header("А.3. Период крутильных колебаний стержня")
    df = read_raw("rod_period.csv")
    s_t = get_param(p, "sigma_t")[0]
    s_t = s_t if np.isfinite(s_t) else 0.0

    N, t = col(df, "N_osc"), col(df, "t_s")
    df["T_s"] = t / N
    df["sigma_T_s"] = s_t / N
    # приборная погрешность периода -- по наибольшему N (самое точное измерение)
    s_instr = np.nanmin(df["sigma_T_s"]) if np.isfinite(df["sigma_T_s"]).any() else 0.0
    T, s_T = mean_with_error(df["T_s"], s_instr)
    RES.add("T", "период T", T, s_T, "с")
    if np.isfinite(T) and s_t > 0:
        print(f"    для eps_T <= 1 % нужно N >= {s_t / (0.01 * T):.0f} колебаний")
    write_processed(df, "rod_period_processed.csv",
                    ["trial", "N_osc", "t_s", "T_s", "sigma_T_s"])
    return {"T": (T, s_T)}


def calc_B0(p: dict, rod: dict, defl: dict, per: dict) -> dict:
    """Горизонтальная составляющая поля Земли, формула (7)."""
    header("А.4. Горизонтальная составляющая поля Земли B0")
    R_cm, s_R_cm = get_param(p, "R_ring")
    L_cm, s_L_cm = get_param(p, "L")
    R, s_R = R_cm * 1e-2, s_R_cm * 1e-2
    L, s_L = L_cm * 1e-2, s_L_cm * 1e-2
    J, s_J = rod["J"]
    x1, s_x1 = defl["x1"]
    T, s_T = per["T"]

    B0 = 1 / (T * R) * np.sqrt(2 * np.pi * MU0 * J * L / (R * x1))
    eps = np.sqrt(rel(T, s_T) ** 2 + (1.5 * rel(R, s_R)) ** 2
                  + (0.5 * rel(J, s_J)) ** 2 + (0.5 * rel(L, s_L)) ** 2
                  + (0.5 * rel(x1, s_x1)) ** 2)
    s_B0 = B0 * eps

    # магнитный момент стержня из (5): m = 4 pi^2 J / (T^2 B0)
    m_dip = 4 * np.pi**2 * J / (T**2 * B0)

    RES.add("tgone", "tg phi1 = x1/2L", x1 / (2 * L), np.nan)
    RES.add("Bzero", "B0", B0 * 1e6, s_B0 * 1e6, "мкТл")
    RES.add("Bzerogs", "B0 (в гауссах)", B0 * 1e4, s_B0 * 1e4, "Гс")
    RES.add("mrod", "магнитный момент стержня", m_dip * 1e3, m_dip * 1e3 * eps, "мА*м^2")
    return {"B0": (B0, s_B0), "R": (R, s_R), "L": (L, s_L),
            "eps_B0_parts": {"T": rel(T, s_T), "R": rel(R, s_R), "J": rel(J, s_J),
                             "L": rel(L, s_L), "x1": rel(x1, s_x1)}}


# --------------------------------------------------------------------------
# Б. Электродинамическая постоянная
# --------------------------------------------------------------------------
def process_current(p: dict, a: dict) -> None:
    """Токи в СИ (8) и СГС (9), постоянная c по (10)."""
    header("Б. Электродинамическая постоянная c")
    df = read_raw("current.csv")
    N_turns = get_param(p, "N_turns")[0]
    C_uF, s_C_uF = get_param(p, "C")
    nu, s_nu = get_param(p, "nu")
    s_x = get_param(p, "sigma_x")[0]
    s_x = s_x if np.isfinite(s_x) else 0.0
    s_U = get_param(p, "sigma_U")[0]
    s_U = s_U if np.isfinite(s_U) else 0.0

    B0, s_B0 = a["B0"]
    R, s_R = a["R"]
    L, s_L = a["L"]
    A = 2 * B0 * R / (MU0 * N_turns)          # постоянная прибора, А
    C = C_uF * 1e-6

    U = col(df, "U_V")
    x2 = (col(df, "x2_plus_mm") + col(df, "x2_minus_mm")) / 2 * 1e-3
    df["x2_mm"] = x2 * 1e3
    df["tg_phi2"] = x2 / (2 * L)
    df["I_SI_mA"] = A * df["tg_phi2"] * 1e3
    C_cm = C * 9e11                             # 1 Ф = 9*10^11 см
    U_cgs = U / 300                             # 1 В = 1/300 ед. СГС
    df["I_CGS_1e6"] = C_cm * U_cgs * nu / 1e6   # ед. СГС (статампер), x10^6
    df["c_1e8"] = (df["I_CGS_1e6"] * 1e6) / (df["I_SI_mA"] * 1e-3) / 10 / 1e8

    # Итог -- по средним U и x2
    x2_all = np.concatenate([col(df, "x2_plus_mm"), col(df, "x2_minus_mm")])
    x2m, s_x2m = mean_with_error(x2_all, s_x)
    Um, s_Um = mean_with_error(U, s_U)
    x2m, s_x2m = x2m * 1e-3, s_x2m * 1e-3

    I_si = A * x2m / (2 * L)
    I_cgs = C_cm * Um / 300 * nu
    c = I_cgs / I_si / 10

    e = a["eps_B0_parts"]
    # I_SI ~ T^-1 R^-1/2 J^1/2 L^-1/2 x1^-1/2 x2  (B0 подставлено из (7))
    eps_Isi = np.sqrt(e["T"] ** 2 + (0.5 * e["R"]) ** 2 + (0.5 * e["J"]) ** 2
                      + (0.5 * e["L"]) ** 2 + (0.5 * e["x1"]) ** 2 + rel(x2m, s_x2m) ** 2)
    eps_Icgs = np.sqrt(rel(C_uF, s_C_uF) ** 2 + rel(Um, s_Um) ** 2 + rel(nu, s_nu) ** 2)
    eps_c = np.hypot(eps_Isi, eps_Icgs)

    RES.add("A", "постоянная прибора A", A * 1e3, A * 1e3 * np.hypot(rel(B0, s_B0), rel(R, s_R)), "мА")
    RES.add("U", "напряжение U", Um, s_Um, "В")
    RES.add("xtwo", "отклонение x2", x2m * 1e3, s_x2m * 1e3, "мм")
    RES.add("ISI", "I (СИ)", I_si * 1e3, I_si * 1e3 * eps_Isi, "мА")
    RES.add("ICGS", "I (СГС)", I_cgs / 1e6, I_cgs / 1e6 * eps_Icgs, "10^6 ед. СГС")
    RES.add("c", "электродинамическая постоянная c", c / 1e8, c / 1e8 * eps_c, "10^8 м/с")
    RES.add("cdev", "отклонение от c табл.", (c / 2.99792458e8 - 1) * 100, np.nan, "%")

    write_processed(df, "current_processed.csv",
                    ["trial", "U_V", "x2_plus_mm", "x2_minus_mm", "x2_mm",
                     "tg_phi2", "I_SI_mA", "I_CGS_1e6", "c_1e8"])


def main() -> None:
    params = read_params()
    rod = process_rod(params)
    defl = process_deflection(params)
    per = process_period(params)
    a = calc_B0(params, rod, defl, per)
    process_current(params, a)
    header("Итог")
    RES.save()


if __name__ == "__main__":
    with np.errstate(all="ignore"):
        pd.set_option("mode.chained_assignment", None)
        main()
