"""Обработка сырых данных лабораторной работы 3.4.4.

    «Петля гистерезиса (статический метод)»

Читает   data/raw/*.csv
Пишет    data/processed/*.csv        -- таблицы и кривые для отчёта
         data/processed/results.tex  -- макросы \\res{...} с итоговыми числами
         data/processed/fit_results.csv
Печатает сводку с промежуточными и окончательными результатами.

Запуск (из каталога работы):
    python scripts/process_data.py

Сырые данные никогда не изменяются. На незаполненных таблицах скрипт создаёт
файлы с одними заголовками, а в отчёте вместо чисел остаются метки ???.

Основные формулы (нумерация -- как в методичке)
-----------------------------------------------
(1)  H = N I / (pi D)
(4)  dB = mu0 * (N_c N'_c / N') * (R / R_c) * (d_c / d)^2 * (dI_c / l_c) * (dx / dx_c)
         = k_B * dx
Петля строится суммированием скачков dB. Каждая ветвь симметрична
относительно центра петли, поэтому B в её начальной точке равен половине
суммы всех скачков по ветви: B(C) = +S_down/2, B(C') = -S_up/2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab_utils import (
    Results,
    TexResults,
    get_param,
    lsq,
    num_str,
    read_params,
    read_raw,
    write_processed,
)

MU0 = 4e-7 * np.pi
RES = Results()
TEX = TexResults()


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def num(x):
    return pd.to_numeric(x, errors="coerce")


# --------------------------------------------------------------------------
# Коэффициенты пересчёта
# --------------------------------------------------------------------------
def h_coef(params) -> float:
    """H = h_I * I,  h_I = N / (pi D)  [А/м на А]."""
    N = get_param(params, "N", np.nan)
    D = get_param(params, "D_cm", np.nan) * 1e-2
    return N / (np.pi * D)


def process_calib(params) -> tuple[float, float]:
    """Калибровка (пп.6-8): k_B -- Тл на деление шкалы."""
    header("Калибровка гальванометра (пп.6-8)")
    cols = ["dI_c_mA", "dx_c", "ratio"]
    df = read_raw("calib.csv")
    div_err = get_param(params, "div_err", 0.5)
    eps_I = get_param(params, "eps_ammeter", 0.01)
    eps_g = get_param(params, "eps_geom", 0.02)

    if df.empty:
        write_processed(pd.DataFrame(), "calib_processed.csv", cols)
        print("  нет данных")
        return np.nan, np.nan

    dI = num(df["dI_c_mA"]).to_numpy(float)
    dx = num(df["dx_c"]).to_numpy(float)
    ratio = dI / dx                                          # мА/дел
    write_processed(pd.DataFrame({
        "dI_c_mA": [num_str(v) for v in dI],
        "dx_c": [num_str(v) for v in dx],
        "ratio": [num_str(v, sig=4) for v in ratio],
    }), "calib_processed.csv", cols)

    r = float(np.nanmean(ratio))
    n = int(np.sum(np.isfinite(ratio)))
    stat = (np.nanstd(ratio, ddof=1) / np.sqrt(n) / r) if n > 1 else 0.0
    eps_r = np.sqrt(stat**2 + eps_I**2 + (div_err / np.nanmean(dx)) ** 2)

    Nc = get_param(params, "N_c", np.nan)
    Ncm = get_param(params, "N_c_meas", np.nan)
    Nm = get_param(params, "N_meas", np.nan)
    lc = get_param(params, "l_c_cm", np.nan) * 1e-2
    dc = get_param(params, "d_c_cm", np.nan) * 1e-2
    d = get_param(params, "d_cm", np.nan) * 1e-2
    Rr = get_param(params, "R_ratio", 1.0)

    kB = MU0 * Nc * Ncm / Nm * Rr * (dc / d) ** 2 * (r * 1e-3) / lc   # Тл/дел
    eps_k = np.sqrt(eps_r**2 + eps_g**2)
    RES.add("dIc_over_dxc", r, r * eps_r, "mA/div")
    RES.add("k_B", kB, kB * eps_k, "T/div", "цена деления по индукции")
    TEX.add("calRatio", r, r * eps_r)
    TEX.add("kB", kB * 1e3, kB * eps_k * 1e3)          # мТл/дел
    TEX.add("epsK", eps_k * 100, sig=2)
    return kB, eps_k


# --------------------------------------------------------------------------
# Предельная петля (пп.5, 12-13, 15)
# --------------------------------------------------------------------------
def zero_cross(H, B):
    """H, при котором B меняет знак (линейная интерполяция)."""
    for i in range(len(B) - 1):
        if B[i] == 0:
            return H[i]
        if B[i] * B[i + 1] < 0:
            return H[i] - B[i] * (H[i + 1] - H[i]) / (B[i + 1] - B[i])
    return np.nan


def b_at_zero_h(H, B):
    """B при H = 0: точка с I = 0, если есть, иначе интерполяция."""
    z = np.where(H == 0)[0]
    if len(z):
        return B[z[0]]
    return zero_cross(B, H)


def process_loop(params, kB, eps_k, hI):
    header("Предельная петля гистерезиса (пп.12-13, 15)")
    cols_t = ["branch", "I_mA", "H", "dx", "dB", "B"]
    cols_c = ["H", "B"]
    df = read_raw("loop.csv")
    div_err = get_param(params, "div_err", 0.5)
    eps_I = get_param(params, "eps_ammeter", 0.01)

    if df.empty:
        write_processed(pd.DataFrame(), "loop_table.csv", cols_t)
        write_processed(pd.DataFrame(), "loop_down.csv", cols_c)
        write_processed(pd.DataFrame(), "loop_up.csv", cols_c)
        print("  нет данных")
        return {}

    table, out = [], {}
    sums = {}
    for br, sign in (("down", -1), ("up", +1)):
        sub = df[df["branch"].astype(str).str.strip() == br]
        if sub.empty:
            write_processed(pd.DataFrame(), f"loop_{br}.csv", cols_c)
            continue
        I = num(sub["I_mA"]).to_numpy(float)
        dx = np.nan_to_num(np.abs(num(sub["dx"]).to_numpy(float)))
        H = hI * I * 1e-3
        S = kB * dx.sum()
        sums[br] = dx.sum()
        B = -sign * S / 2 + sign * kB * np.cumsum(dx)
        write_processed(pd.DataFrame({
            "H": [num_str(v, sig=4) for v in H],
            "B": [num_str(v, sig=4) for v in B]}), f"loop_{br}.csv", cols_c)
        for i in range(len(I)):
            table.append({"branch": r"$C\rightarrow C'$" if br == "down" else r"$C'\rightarrow C$",
                          "I_mA": num_str(I[i], sig=4), "H": num_str(H[i], sig=4),
                          "dx": num_str(dx[i]) if dx[i] else "",
                          "dB": num_str(kB * dx[i] * sign * 1e3, sig=3) if dx[i] else "",
                          "B": num_str(B[i], sig=3)})
        n_steps = int(np.sum(dx > 0))
        out[br] = dict(H=H, B=B, I=I, n=n_steps)

    write_processed(pd.DataFrame(table), "loop_table.csv", cols_t)

    if len(sums) == 2:
        mism = abs(sums["down"] - sums["up"]) / np.mean(list(sums.values())) * 100
        RES.add("sum_mismatch", mism, unit="%", comment="Σdx по ветвям")
        TEX.add("sumDown", sums["down"])
        TEX.add("sumUp", sums["up"])
        TEX.add("sumMism", mism, sig=2)

    # характеристики
    Hc_list, Br_list, Bs_list, Hm_list = [], [], [], []
    for br, d in out.items():
        Hc_list.append(abs(zero_cross(d["H"], d["B"])))
        Br_list.append(abs(b_at_zero_h(d["H"], d["B"])))
        Bs_list.append(abs(d["B"][0]))
        Hm_list.append(abs(d["H"][0]))
    Hc, Br = np.nanmean(Hc_list), np.nanmean(Br_list)
    Bmax, Hmax = np.nanmean(Bs_list), np.nanmean(Hm_list)
    n = max(d["n"] for d in out.values())
    # погрешности: масштаб k_B + накопленная ошибка отсчётов (~sqrt(n/2) шагов)
    dBr = np.hypot(Br * eps_k, kB * div_err * np.sqrt(n / 2))
    dBmax = np.hypot(Bmax * eps_k, kB * div_err * np.sqrt(n) / 2)
    # H_c: шаг по H в интервале, где B меняет знак (точка F) + разброс
    # между ветвями + амперметр
    steps = []
    for d in out.values():
        B_, H_ = d["B"], d["H"]
        for i in range(len(B_) - 1):
            if B_[i] * B_[i + 1] <= 0:
                steps.append(abs(H_[i + 1] - H_[i]))
                break
    stepH = np.nanmean(steps) if steps else 0.0
    dHc = np.sqrt((Hc * eps_I) ** 2 + (stepH / 4) ** 2
                  + (np.nanstd(Hc_list) if len(Hc_list) > 1 else 0) ** 2)

    RES.add("Hc", Hc, dHc, "A/m")
    RES.add("Br", Br, dBr, "T")
    RES.add("B_max", Bmax, dBmax, "T", "B в точке C (H = H_max)")
    RES.add("H_max", Hmax, Hmax * eps_I, "A/m")
    TEX.add("Hc", Hc, dHc)
    TEX.add("Br", Br, dBr)
    TEX.add("Bmax", Bmax, dBmax)
    TEX.add("Hmax", Hmax, Hmax * eps_I)
    TEX.add("Imax", np.nanmax(np.abs(np.concatenate([d["I"] for d in out.values()]))))

    # участок насыщения CD: B = mu0*Ms + k*H по n_sat точкам с наибольшим |H|
    n_sat = int(get_param(params, "n_sat", 4))
    Hs_all = np.concatenate([np.abs(d["H"][:n_sat]) for d in out.values()])
    Bs_all = np.concatenate([np.abs(d["B"][:n_sat]) for d in out.values()])
    if len(Hs_all) >= 3 and np.ptp(Hs_all) > 0:
        k, sk, b, sb = lsq(Hs_all, Bs_all)
        sb = np.hypot(sb, b * eps_k)
        RES.add("mu0Ms", b, sb, "T", "экстраполяция CD к H = 0")
        RES.add("slope_CD_over_mu0", k / MU0, sk / MU0, "", "наклон CD в ед. mu0")
        TEX.add("muMs", b, sb)
        TEX.add("Ms", b / MU0 / 1e6, sb / MU0 / 1e6)        # МА/м
        TEX.add("slopeCD", k / MU0, sk / MU0, sig=2)
        # для линии на графике (pgfplots: числа с точкой)
        TEX.add_text("satK", f"{k:.6g}")
        TEX.add_text("satB", f"{b:.6g}")
        TEX.add_text("satHmin", f"{float(np.min(Hs_all)) * 0.5:.4g}")
        TEX.add_text("satHmax", f"{float(np.max(Hs_all)) * 1.05:.4g}")
    TEX.add_text("nSat", str(n_sat))
    return out


# --------------------------------------------------------------------------
# Начальная кривая (пп.10, 14, 16)
# --------------------------------------------------------------------------
def process_initial(params, kB, eps_k, hI):
    header("Начальная кривая намагничивания (пп.14, 16)")
    cols = ["I_mA", "H", "dx", "B", "mu"]
    df = read_raw("initial.csv")
    if df.empty:
        write_processed(pd.DataFrame(), "initial_processed.csv", cols)
        write_processed(pd.DataFrame(), "mudiff.csv", ["H", "mu_diff"])
        print("  нет данных")
        return
    I = np.r_[0.0, num(df["I_mA"]).to_numpy(float)]
    dx = np.r_[0.0, np.nan_to_num(np.abs(num(df["dx"]).to_numpy(float)))]
    H = hI * I * 1e-3
    B = kB * np.cumsum(dx)
    mu = np.where(H > 0, B / (MU0 * np.where(H > 0, H, 1)), np.nan)
    write_processed(pd.DataFrame({
        "I_mA": [num_str(v, sig=4) for v in I],
        "H": [num_str(v, sig=4) for v in H],
        "dx": [num_str(v) if v else "" for v in dx],
        "B": [num_str(v, sig=4) for v in B],
        "mu": [num_str(v, sig=3) for v in mu],
    }), "initial_processed.csv", cols)

    Hm = (H[1:] + H[:-1]) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        mud = np.diff(B) / np.diff(H) / MU0
    ok = np.isfinite(mud)
    write_processed(pd.DataFrame({
        "H": [num_str(v, sig=4) for v in Hm[ok]],
        "mu_diff": [num_str(v, sig=3) for v in mud[ok]]}), "mudiff.csv", ["H", "mu_diff"])

    if ok.any():
        k = int(np.nanargmax(np.where(ok, mud, -np.inf)))
        mmax = mud[k]
        RES.add("mu_diff_max", mmax, mmax * eps_k, "", f"при H ~ {Hm[k]:.3g} А/м")
        TEX.add("muMax", mmax, mmax * np.hypot(eps_k, 0.1))
        TEX.add("HmuMax", Hm[k], sig=2)
        # максимальная проницаемость mu = B/(mu0 H) вдоль кривой
        TEX.add("muBHmax", np.nanmax(mu), sig=2)
        TEX.add("muInit", mu[1] if len(mu) > 1 else np.nan, sig=2)


def main() -> None:
    params = read_params()
    hI = h_coef(params)
    if np.isfinite(hI):
        TEX.add("hI", hI, sig=4)
    kB, eps_k = process_calib(params)
    process_loop(params, kB, eps_k, hI)
    process_initial(params, kB, eps_k, hI)
    for key, p in [("N", "N"), ("Nm", "N_meas"), ("D", "D_cm"), ("d", "d_cm"),
                   ("Nc", "N_c"), ("Ncm", "N_c_meas"), ("lc", "l_c_cm"),
                   ("dc", "d_c_cm"), ("RM", "R_M"), ("RMp", "R_M_prime"),
                   ("Rc", "R_c"), ("Rg", "R_g")]:
        v = params.get(p)
        if v is not None and np.isfinite(v):
            TEX.add_text(key, f"{v:g}".replace(".", "{,}"))
    header("Сохранение итогов")
    RES.save()
    TEX.save()


if __name__ == "__main__":
    main()
