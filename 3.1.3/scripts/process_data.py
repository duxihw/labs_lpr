"""Обработка сырых данных лабораторной работы 3.1.3
«Измерение магнитного поля Земли».

Читает   data/raw/*.csv
Пишет    data/processed/*.csv, data/processed/fit_results.csv,
         data/processed/results.tex (числа для отчёта, подставляются сами),
         data/processed/fit_T.csv, fit_M.csv (аппроксимирующие прямые для графиков)

Запуск (из каталога работы):
    python scripts/process_data.py

Пока данные не заполнены, скрипт создаёт таблицы-заготовки, а все
результаты в отчёте выводятся как ???.
Расчёт ведётся в гауссовой системе (СГС), как в методичке:
см, г, с, дин, эрг, Гс. Перевод: 1 Гс = 100 мкТл = 10^-4 Тл,
1 эрг/Гс = 10^-3 А*м^2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab_utils import (
    Results,
    col,
    fit_line_csv,
    get_param,
    header,
    lsq,
    lsq_origin,
    mean_with_error,
    read_params,
    read_raw,
    write_processed,
)

G = 981.0                      # см/с^2
R_EARTH = 6.371e6              # м
# Сумма ряда (12): 1 + 1/2^4 + 1/3^4 + 1/4^4
CHAIN_SUM = 1 + 1 / 2**4 + 1 / 3**4 + 1 / 4**4
# Каким методом определённый магнитный момент шарика использовать в частях II, III:
# "A" -- по r_max, "B" -- по силе сцепления (обычно точнее), "mean" -- среднее.
M_DIP_METHOD = "B"

RES = Results()


def rel(value: float, sigma: float) -> float:
    if not np.isfinite(value) or value == 0 or not np.isfinite(sigma):
        return 0.0
    return abs(sigma / value)


def instr(p: dict, name: str) -> float:
    v = get_param(p, name)[0]
    return v if np.isfinite(v) else 0.0


# --------------------------------------------------------------------------
# I. Магнитный момент, намагниченность, остаточная индукция
# --------------------------------------------------------------------------
def process_balls(p: dict) -> dict:
    header("I.1. Размеры и масса шариков, поле на полюсе")
    s_cal, s_sc = instr(p, "sigma_caliper"), instr(p, "sigma_scale")

    d_mm, s_d_mm = mean_with_error(col(read_raw("ball_size.csv"), "d_mm"), s_cal)
    R, s_R = d_mm / 20, s_d_mm / 20                          # см

    dm = read_raw("ball_mass.csv")
    n_b, m_tot = col(dm, "n_balls"), col(dm, "m_total_g")
    m1 = m_tot / n_b
    m, s_m = mean_with_error(m1, s_sc / np.nanmax(n_b) if np.isfinite(n_b).any() else 0.0)

    Bp_meas, s_Bp_meas = mean_with_error(col(read_raw("pole_field.csv"), "Bp_mT"),
                                         instr(p, "sigma_Bp"))

    RES.add("d", "диаметр шарика d", d_mm, s_d_mm, "мм")
    RES.add("m", "масса шарика m", m, s_m, "г")
    RES.add("Bpmeas", "B_p (измерено)", Bp_meas, s_Bp_meas, "мТл")
    return {"R": (R, s_R), "d": (d_mm / 10, s_d_mm / 10), "m": (m, s_m),
            "Bp": (Bp_meas, s_Bp_meas)}


def method_A(p: dict, b: dict) -> tuple[float, float]:
    """Метод А, формула (10): m = sqrt(m g r_max^4 / 6)."""
    header("I.2. Магнитный момент: метод А (r_max)")
    df = read_raw("rmax.csv")
    h_cm, s_h_cm = mean_with_error(col(df, "h_mm"), instr(p, "sigma_h"))
    h_cm, s_h_cm = h_cm / 10, s_h_cm / 10
    d, s_d = b["d"]
    m, s_m = b["m"]
    r = h_cm + d
    s_r = np.hypot(s_h_cm, s_d)
    df["r_max_mm"] = col(df, "h_mm") + d * 10
    mu = np.sqrt(m * G * r**4 / 6)
    s_mu = mu * np.hypot(0.5 * rel(m, s_m), 2 * rel(r, s_r))
    RES.add("rmax", "r_max", r * 10, s_r * 10, "мм")
    RES.add("mA", "магнитный момент (метод А)", mu, s_mu, "эрг/Гс")
    write_processed(df, "rmax_processed.csv", ["trial", "h_mm", "r_max_mm"])
    return mu, s_mu


def method_B(p: dict, b: dict) -> tuple[float, float]:
    """Метод Б, формулы (11), (12): F = 1.08 F0, F0 = 3 m^2 / (8 R^4)."""
    header("I.3. Магнитный момент: метод Б (сила сцепления)")
    df = read_raw("chain_break.csv")
    M, s_M = mean_with_error(col(df, "M_g"), instr(p, "sigma_scale"))
    R, s_R = b["R"]
    F = M * G                                     # дин
    F0 = F / CHAIN_SUM
    mu = np.sqrt(8 * R**4 * F0 / 3)
    s_mu = mu * np.hypot(0.5 * rel(M, s_M), 2 * rel(R, s_R))
    df["F_N"] = col(df, "M_g") * 1e-3 * G / 100   # Н
    RES.add("Mchain", "масса оторвавшейся цепочки", M, s_M, "г")
    RES.add("Fzero", "сила сцепления F0", F0 * 1e-5, F0 * 1e-5 * rel(M, s_M), "Н")
    RES.add("mB", "магнитный момент (метод Б)", mu, s_mu, "эрг/Гс")
    write_processed(df, "chain_break_processed.csv", ["trial", "M_g", "F_N"])
    return mu, s_mu


def magnetization(b: dict, mu: float, s_mu: float) -> None:
    header("I.4. Намагниченность и остаточная индукция")
    R, s_R = b["R"]
    V = 4 / 3 * np.pi * R**3
    Mz = mu / V                                   # Гс (эрг/(Гс*см^3))
    s_Mz = Mz * np.hypot(rel(mu, s_mu), 3 * rel(R, s_R))
    Br = 4 * np.pi * Mz                           # Гс
    Bp = 2 / 3 * Br
    RES.add("Mag", "намагниченность M", Mz, s_Mz, "Гс (= 10^3 А/м)")
    RES.add("Br", "остаточная индукция B_r", Br * 1e-4, s_Mz * 4 * np.pi * 1e-4, "Тл")
    RES.add("Bpcalc", "B_p = 2/3 B_r (расчёт)", Bp * 0.1, s_Mz * 4 * np.pi * 2 / 3 * 0.1, "мТл")
    Bp_meas = b["Bp"][0]
    RES.add("Bpdev", "расхождение B_p расч./изм.", (Bp * 0.1 / Bp_meas - 1) * 100, np.nan, "%")


# --------------------------------------------------------------------------
# II. Горизонтальная составляющая
# --------------------------------------------------------------------------
def horizontal(p: dict, b: dict, mu: float, s_mu: float) -> tuple[float, float]:
    """T_n = k n, k = 2 pi sqrt(m R^2 / (3 m_dip B_par)), формула (15)."""
    header("II. Горизонтальная составляющая B_par")
    s_t = instr(p, "sigma_t")
    df = read_raw("torsion_period.csv")
    n, N, t = col(df, "n"), col(df, "N_osc"), col(df, "t_s")
    df["T_s"] = t / N
    df["sigma_T_s"] = s_t / N

    ring = read_raw("torsion_ring.csv")
    T_ring = (col(ring, "t_s") / col(ring, "N_osc"))
    T_ring = T_ring[0] if len(T_ring) else np.nan
    RES.add("Tring", "период «кольца» (m = 0)", T_ring, s_t / col(ring, "N_osc")[0] if len(ring) else np.nan, "с")

    k, s_k = lsq_origin(n, df["T_s"])
    k2, s_k2, b2, s_b2 = lsq(n, df["T_s"])
    RES.add("kT", "наклон T(n), через 0", k, s_k, "с")
    RES.add("kTb", "наклон T(n) = k n + b", k2, s_k2, "с")
    RES.add("bT", "свободный член b", b2, s_b2, "с")

    m, s_m = b["m"]
    R, s_R = b["R"]
    Bpar = 4 * np.pi**2 * m * R**2 / (3 * mu * k**2)    # Гс
    eps = np.sqrt(rel(m, s_m) ** 2 + (2 * rel(R, s_R)) ** 2
                  + rel(mu, s_mu) ** 2 + (2 * rel(k, s_k)) ** 2)
    RES.add("Bpar", "B_par", Bpar * 100, Bpar * 100 * eps, "мкТл")

    fit_line_csv("fit_T.csv", k, 0.0, 0, 13)
    write_processed(df, "torsion_period_processed.csv",
                    ["n", "N_osc", "t_s", "T_s", "sigma_T_s"])
    return Bpar, Bpar * eps


# --------------------------------------------------------------------------
# III. Вертикальная составляющая
# --------------------------------------------------------------------------
def vertical(p: dict, mu: float, s_mu: float) -> tuple[float, float]:
    """M_n = m_gr g r_gr = n m_dip B_perp, формула (16)."""
    header("III. Вертикальная составляющая B_perp")
    df = read_raw("moment.csv")
    n = col(df, "n")
    m_gr = col(df, "m_gr_mg") * 1e-3               # г
    r_gr = col(df, "r_gr_mm") / 10                 # см
    s_mgr = instr(p, "sigma_scale")
    s_rgr = instr(p, "sigma_ruler") / 10
    Mn = m_gr * G * r_gr                           # дин*см
    df["M_dyncm"] = Mn
    df["sigma_M_dyncm"] = Mn * np.sqrt((s_mgr / m_gr) ** 2 + (s_rgr / r_gr) ** 2)

    k, s_k = lsq_origin(n, Mn)
    k2, s_k2, b2, s_b2 = lsq(n, Mn)
    RES.add("kM", "наклон M(n), через 0", k, s_k, "дин*см")
    RES.add("kMb", "наклон M(n) = k n + b", k2, s_k2, "дин*см")
    RES.add("bM", "свободный член b", b2, s_b2, "дин*см")

    Bperp = k / mu
    s_Bperp = Bperp * np.hypot(rel(k, s_k), rel(mu, s_mu))
    RES.add("Bperp", "B_perp", Bperp * 100, s_Bperp * 100, "мкТл")

    fit_line_csv("fit_M.csv", k, 0.0, 0, 13)
    write_processed(df, "moment_processed.csv",
                    ["n", "m_gr_mg", "r_gr_mm", "M_dyncm", "sigma_M_dyncm"])
    return Bperp, s_Bperp


def inclination(p: dict, Bpar, s_Bpar, Bperp, s_Bperp) -> None:
    header("IV. Наклонение, полное поле, момент Земли")
    beta = np.arctan2(Bperp, Bpar)
    # d(beta) = (Bpar dBperp - Bperp dBpar) / B^2
    B = np.hypot(Bpar, Bperp)
    s_beta = np.hypot(Bpar * s_Bperp, Bperp * s_Bpar) / B**2
    s_B = np.hypot(Bpar * s_Bpar, Bperp * s_Bperp) / B
    RES.add("beta", "наклонение beta", np.degrees(beta), np.degrees(s_beta), "град")
    RES.add("Btot", "полная индукция B", B * 100, s_B * 100, "мкТл")

    lat_deg = get_param(p, "latitude")[0]
    RES.add("lat", "широта места опыта", lat_deg, np.nan, "град")
    lat = np.radians(lat_deg)
    beta_th = np.arctan(2 * np.tan(lat))           # дипольная модель: tg beta = 2 tg phi
    RES.add("betath", "наклонение (дипольная модель)", np.degrees(beta_th), np.nan, "град")
    # B_par = (mu0/4pi) m_E cos(phi) / R_E^3 (СИ)
    mE = Bpar * 1e-4 * R_EARTH**3 / (1e-7 * np.cos(lat))
    RES.add("mE", "магнитный момент Земли", mE / 1e22, s_Bpar / Bpar * mE / 1e22, "10^22 А*м^2")


def main() -> None:
    p = read_params()
    b = process_balls(p)
    mA = method_A(p, b)
    mB = method_B(p, b)
    if M_DIP_METHOD == "A":
        mu, s_mu = mA
    elif M_DIP_METHOD == "B":
        mu, s_mu = mB
    else:
        mu = (mA[0] + mB[0]) / 2
        s_mu = max(np.hypot(mA[1], mB[1]) / 2, abs(mA[0] - mB[0]) / 2)
    RES.add("mu", f"магнитный момент для частей II, III (метод {M_DIP_METHOD})", mu, s_mu, "эрг/Гс")
    RES.add("muSI", "то же в СИ", mu * 1e-3, s_mu * 1e-3, "А*м^2")
    magnetization(b, mu, s_mu)
    Bpar, s_Bpar = horizontal(p, b, mu, s_mu)
    Bperp, s_Bperp = vertical(p, mu, s_mu)
    inclination(p, Bpar, s_Bpar, Bperp, s_Bperp)
    header("Итог")
    RES.save()


if __name__ == "__main__":
    with np.errstate(all="ignore"):
        main()
