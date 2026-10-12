"""Обработка данных лабораторной работы 3.6.1.

    «Спектральный анализ электрических сигналов»

Исходные данные:
  * части А (гармоники, Dnu(1/tau), dnu(1/T)) и Б -- таблицы измерений
    data/raw/{harmonics,spectrum_width,harmonics_spacing,bursts}.csv;
  * спектр меандра, части Г и Е -- отсчёты со скриншотов осциллографа
    (data/raw/screens/*.csv, их готовит scripts/digitize_screens.py).

Пишет
    data/processed/*.csv       таблицы для отчёта
    data/processed/fit_results.csv   все итоговые числа
    data/processed/fit_macros.tex    те же числа в виде макросов LaTeX
                                     (подключаются в report/main.tex)

Запуск (из каталога работы):
    python scripts/process_data.py

Сырые данные никогда не изменяются.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares

from lab_utils import PROC_DIR, RAW_DIR, Results, fmt, lsq, lsq_origin, read_raw, write_processed

RES = Results()
MACROS: list[str] = []


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def read_screen(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "screens" / name, comment="#")


# --------------------------------------------------------------------------
# Макросы LaTeX с результатами
# --------------------------------------------------------------------------
def _tex_num(x: float, digits: int) -> str:
    s = f"{x:.{digits}f}"
    return s.replace("-", "-").replace(".", "{,}")


def macro(name: str, value: float, sigma: float | None = None,
          digits: int | None = None) -> None:
    """\\name -> «значение \\pm погрешность» (с запятой), \\nameNum -> число."""
    if digits is None:
        if sigma is not None and np.isfinite(sigma) and sigma > 0:
            digits = max(0, int(-np.floor(np.log10(sigma))) + (1 if f"{sigma:.1e}"[0] == "1" else 0))
        else:
            digits = 3
    text = _tex_num(value, digits)
    if sigma is not None and np.isfinite(sigma):
        text += r" \pm " + _tex_num(sigma, digits)
    MACROS.append(rf"\newcommand{{\{name}}}{{{text}}}")
    MACROS.append(rf"\newcommand{{\{name}Val}}{{{_tex_num(value, digits)}}}")
    MACROS.append(rf"\newcommand{{\{name}Num}}{{{value:.6g}}}")


def save_macros() -> None:
    path = PROC_DIR / "fit_macros.tex"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("% Сгенерировано scripts/process_data.py -- вручную не править\n")
        fh.write("\n".join(MACROS) + "\n")
    print(f"  -> data/processed/fit_macros.tex ({len(MACROS) // 3} величин)")


def wlsq_origin(x, y, sy):
    """Взвешенный МНК y = k x: k, sigma_k (по погрешностям точек), chi2/dof."""
    x, y, sy = (np.asarray(v, float) for v in (x, y, sy))
    w = 1 / sy ** 2
    k = np.sum(w * x * y) / np.sum(w * x * x)
    sk = 1 / np.sqrt(np.sum(w * x * x))
    chi2 = float(np.sum(w * (y - k * x) ** 2)) / max(1, len(x) - 1)
    return k, sk * max(1.0, np.sqrt(chi2)), chi2


# ==========================================================================
# А. Прямоугольные импульсы
# ==========================================================================
def rect_model(n, q):
    """|sin(pi n q)| / n, q = tau/T."""
    return np.abs(np.sin(np.pi * n * q)) / n


def process_meander() -> dict:
    header("А. Меандр в диапазоне 0-780 кГц; наложение частот")
    df = read_screen("meander_lines.csv")
    df = df[df["screen"] != "spectrum_am_wide_2.png"]      # дубликат wide_1
    res = {}
    rows = []
    for scr, g in df.groupby("screen", sort=False):
        nu1_set = g["nu1_set_kHz"].iloc[0]
        odd = g[g["n"] % 2 == 1]
        even = g[g["n"] % 2 == 0]
        k, sk, b, sb = lsq(odd["n"], odd["nu_kHz"])
        a1 = odd["a_mV"].iloc[0]
        res[scr] = {"nu1": k, "snu1": sk, "b": b, "a1": a1,
                    "a_even_max": float(even["a_mV"].max()), "nu1_set": nu1_set,
                    "T_us": 1000.0 / k}
        RES.add(f"nu1 ({scr})", k, sk, "кГц")
        RES.add(f"смещение шкалы частот ({scr})", b, sb, "кГц")
        for _, r in odd.iterrows():
            rows.append({
                "nu1_kHz": int(nu1_set), "n": int(r["n"]),
                "nu_kHz": round(r["nu_kHz"] - b, 1),
                "a_mV": round(r["a_mV"], 2),
                "ratio": round(r["a_mV"] / a1, 4),
                "inv_n": round(1 / r["n"], 4),
                "n_ratio": round(r["n"] * r["a_mV"] / a1, 3),
            })
        RES.add(f"макс. a чётных гармоник / a1 ({scr})",
                float(even["a_mV"].max() / a1), np.nan, "")
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    write_processed(out, "meander_processed.csv")

    w1 = res["spectrum_am_wide_1.png"]
    w3 = res["spectrum_am_wide_3.png"]
    macro("nuMeandA", w1["nu1"], w1["snu1"] if w1["snu1"] > 0.05 else 0.05, 1)
    macro("nuMeandB", w3["nu1"], w3["snu1"] if w3["snu1"] > 0.05 else 0.05, 1)
    macro("evenMax", w1["a_even_max"], None, 2)
    macro("evenRel", w1["a_even_max"] / w1["a1"], None, 3)
    m1 = out[out["nu1_kHz"] == 50]
    macro("nRatioMinA", float(m1["n_ratio"].min()), None, 2)
    macro("nRatioThreeA", float(m1.loc[m1["n"] == 3, "n_ratio"].iloc[0]), None, 3)
    macro("nRatioFiveA", float(m1.loc[m1["n"] == 5, "n_ratio"].iloc[0]), None, 3)

    # --- наложение частот -----------------------------------------------
    oth = read_screen("meander_other_lines.csv")
    oth = oth[oth["screen"] != "spectrum_am_wide_2.png"].copy()
    oth["nu_corr"] = [r["nu_kHz"] - res[r["screen"]]["b"] for _, r in oth.iterrows()]

    def assign(fs):
        """Для каждой линии -- ближайшее изображение нечётной гармоники |n nu1 - k fs|."""
        out_rows, d2 = [], 0.0
        for _, r in oth.iterrows():
            nu1 = res[r["screen"]]["nu1"]
            best = None
            for n in range(1, 80, 2):
                for kk in (1, 2, 3):
                    pred = abs(n * nu1 - kk * fs)
                    if pred > fs / 2 or n * nu1 < fs / 2:
                        continue
                    d = pred - r["nu_corr"]
                    if best is None or abs(d) < abs(best[0]):
                        best = (d, n, kk, pred)
            out_rows.append(best)
            d2 += best[0] ** 2
        return d2, out_rows

    # Экран спектроанализатора PicoScope показывает полосу 0 ... nu_дискр/2.
    # Правый край области графика -- столбец 1092 пикселя (сетка: 45 -> 0,
    # 985 -> 700 кГц), т.е. nu_дискр/2 ~ 780 кГц. Картина наложения для
    # гребёнки нечётных гармоник симметрична относительно nu_дискр -> 2 nu1*k,
    # поэтому искать nu_дискр нужно только вблизи 2 * 780 кГц.
    edge = (1092 - 45) * 700.0 / (985 - 45)
    macro("screenEdge", edge, None, 0)
    grid = np.arange(2 * edge - 30, 2 * edge + 30, 0.05)
    fs0 = grid[np.argmin([assign(f)[0] for f in grid])]
    _, best = assign(fs0)
    # уточнение fs по МНК: nu_obs = s*(k fs - n nu1), s = ±1
    fs_each = []
    for (d, n, kk, pred), (_, r) in zip(best, oth.iterrows()):
        nu1 = res[r["screen"]]["nu1"]
        sign = 1.0 if kk * fs0 > n * nu1 else -1.0
        fs_each.append((sign * r["nu_corr"] + n * nu1) / kk)
    fs_each = np.array(fs_each)
    fs = fs_each.mean()
    # случайная часть + погрешность частоты nu1 (входит в n*nu1 с множителем n)
    n_mean = np.mean([b_[1] / b_[2] for b_ in best])
    s_nu1 = max(r_["snu1"] for r_ in res.values())
    sfs = float(np.hypot(fs_each.std(ddof=1) / np.sqrt(len(fs_each)), n_mean * s_nu1))
    RES.add("частота дискретизации nu_дискр", fs, sfs, "кГц", "по линиям наложения")
    macro("fsamp", fs, sfs, 1)
    macro("fnyq", fs / 2, sfs / 2, 1)

    arows = []
    for (d, n, kk, pred), (_, r), fe in zip(best, oth.iterrows(), fs_each):
        nu1 = res[r["screen"]]["nu1"]
        a1 = res[r["screen"]]["a1"]
        arows.append({
            "nu1_kHz": int(r["nu1_set_kHz"]),
            "nu_obs_kHz": round(r["nu_corr"], 1),
            "n": n, "nu_n_kHz": round(n * nu1, 1), "k": kk,
            "nu_pred_kHz": round(abs(n * nu1 - kk * fs), 1),
            "a_mV": round(r["a_mV"], 2),
            "n_ratio": round(n * r["a_mV"] / a1, 3),
        })
    aout = pd.DataFrame(arows).sort_values(["nu1_kHz", "n"])
    print(aout.to_string(index=False))
    write_processed(aout, "alias_processed.csv")
    return res


def dec(x: float, k: int) -> str:
    return f"{x:.{k}f}".replace(".", "{,}")


def process_harmonics() -> None:
    """А, п.7: частоты и относительные амплитуды гармоник (T = 1 мс, tau = 50 мкс)."""
    header("А. Гармоники прямоугольного сигнала (T = 1 мс, tau = 50 мкс)")
    df = read_raw("harmonics.csv")
    T_ms, tau_us = 1.0, 50.0
    q = tau_us / (1000 * T_ms)
    n = df["n"].to_numpy(float)
    a = df["a_exp_mV"].to_numpy(float)
    ratio = a / a[0]
    th = rect_model(n, q) / rect_model(1, q)
    dev = (ratio - th) / th * 100
    out = pd.DataFrame({
        "n": n.astype(int),
        "nu_exp_kHz": df["nu_exp_kHz"],
        "nu_theor_kHz": (n / T_ms).astype(int),
        "a_exp_mV": a,
        "ratio_exp": np.round(ratio, 3),
        "ratio_theor": np.round(th, 3),
        "dev_percent": ["0{,}0" if abs(d) < 0.05 else f"{d:+.1f}".replace(".", "{,}") for d in dev],
        "dev": np.round(dev, 2),
    })
    print(out.to_string(index=False))
    write_processed(out, "harmonics_processed.csv")
    d = dev[1:]
    RES.add("отклонение |a_n/a_1| от теории: среднее", float(d.mean()), float(d.std(ddof=1)), "%")
    macro("hDevMean", float(d.mean()), None, 1)
    macro("hDevMin", float(d.min()), None, 1)
    macro("hDevMax", float(d.max()), None, 1)
    macro("aOne", a[0], None, 1)
    # оценка «эффективной» длительности по огибающей: подгонка q
    f = lambda nn, qq: rect_model(nn, qq) / rect_model(1, qq)
    p, c = curve_fit(f, n, ratio, p0=[q])
    tau_eff = p[0] * T_ms * 1000
    s_tau = np.sqrt(c[0, 0]) * T_ms * 1000
    RES.add("tau по огибающей гармоник", tau_eff, s_tau, "мкс", "установлено 50 мкс")
    macro("tauEnv", tau_eff, s_tau, 1)


def process_width() -> None:
    """А, п.8: ширина спектра Dnu от 1/tau при T = 1 мс."""
    header("А. Ширина спектра Dnu(1/tau), T = 1 мс")
    df = read_raw("spectrum_width.csv")
    tau = df["tau_us"].to_numpy(float)
    D = df["Dnu_kHz"].to_numpy(float)
    x = 1000.0 / tau
    sD = np.full_like(D, 0.5)          # половина шага гармоник 1/T = 1 кГц
    out = pd.DataFrame({
        "tau_us": tau.astype(int),
        "inv_tau_kHz": [dec(v, 2) for v in x],
        "Dnu_kHz": [f"{dec(v, 1)} \\pm {dec(s, 1)}" for v, s in zip(D, sD)],
        "Dnu_tau": [dec(v, 3) for v in D * tau / 1000],
        "x": np.round(x, 3), "y": D, "ey": sD,
    })
    print(out.to_string(index=False))
    write_processed(out, "spectrum_width_processed.csv")
    k0, sk0, chi = wlsq_origin(x, D, sD)
    k, sk, b, sb = lsq(x, D)
    RES.add("k_Dnu (взвеш. МНК через 0)", k0, sk0, "", f"теория 1, chi2/dof = {chi:.2f}")
    RES.add("k_Dnu (y = kx + b)", k, sk, "")
    RES.add("b_Dnu", b, sb, "кГц")
    macro("kDnu", k0, sk0)
    macro("kDnuFree", k, sk)
    macro("bDnu", b, sb)
    prod = D * tau / 1000
    macro("prodMin", prod.min(), None, 2)
    macro("prodMax", prod.max(), None, 2)
    macro("prodMean", prod.mean(), prod.std(ddof=1) / np.sqrt(len(prod)), 3)


def process_spacing() -> None:
    """А, п.9: расстояние между гармониками dnu от 1/T при tau = 100 мкс."""
    header("А. Расстояние между гармониками dnu(1/T), tau = 100 мкс")
    df = read_raw("harmonics_spacing.csv")
    tau_us = 100.0
    nu = df["nu_rep_kHz"].to_numpy(float)
    dnu = df["dnu_kHz"].to_numpy(float)
    T = 1.0 / nu
    meander = np.isclose(T * 1000, 2 * tau_us)
    out = pd.DataFrame({
        "T_ms": [dec(t, 2) for t in T],
        "T_tau": [dec(t * 1000 / tau_us, 0) for t in T],
        "inv_T_kHz": [dec(v, 3) for v in nu],
        "dnu_kHz": [dec(v, 3) for v in dnu],
        "dnu_T": [dec(v, 3) for v in dnu * T],
        "note": ["меандр" if m else "" for m in meander],
        "x": nu, "y": dnu, "use": (~meander).astype(int),
    })
    print(out.to_string(index=False))
    write_processed(out, "harmonics_spacing_processed.csv")
    x, y = nu[~meander], dnu[~meander]
    k0, sk0 = lsq_origin(x, y)
    k, sk, b, sb = lsq(x, y)
    RES.add("k_dnu (МНК через 0, без меандра)", k0, sk0, "", "теория 1")
    RES.add("k_dnu (y = kx + b)", k, sk, "")
    RES.add("b_dnu", b, sb, "кГц")
    macro("kdnu", k0, max(sk0, 1e-4))
    macro("kdnuFree", k, sk)
    macro("bdnu", b, sb)
    RES.add("dnu*T для T = 2 tau (меандр)", float(dnu[meander][0] * T[meander][0]), np.nan, "")


def process_bursts() -> None:
    """Часть Б: Dnu = 1/tau = nu0/N, dnu = 1/T."""
    header("Б. Спектр последовательности цугов")
    df = read_raw("bursts.csv")
    nu0 = df["nu0_kHz"].to_numpy(float)
    T = df["T_ms"].to_numpy(float)
    N = df["N"].to_numpy(float)
    dnu = df["dnu_kHz"].to_numpy(float)
    D = df["Dnu_kHz"].to_numpy(float)
    tau_us = N / nu0 * 1000
    sD = dnu / 2                       # половина шага гармоник
    names = {"N": "$N$", "nu0": "$\\nu_0$", "T": "$T$"}
    out = pd.DataFrame({
        "series": [names[s] for s in df["series"]],
        "nu0_kHz": nu0.astype(int), "T_ms": T.astype(int), "N": N.astype(int),
        "tau_us": [dec(t, 0) if abs(t - round(t)) < 1e-9 else dec(t, 1) for t in tau_us],
        "dnu_kHz": [dec(v, 2) for v in dnu],
        "dnu_T": [dec(v, 2) for v in dnu * T],
        "Dnu_kHz": [f"{dec(v, 1)} \\pm " + f"{s:g}".replace(".", "{,}") for v, s in zip(D, sD)],
        "inv_tau_kHz": [dec(v, 1) for v in 1000 / tau_us],
        "Dnu_tau": [dec(v, 2) for v in D * tau_us / 1000],
    })
    print(out.to_string(index=False))
    write_processed(out, "bursts_processed.csv")

    # точки для графиков (уникальные tau / уникальные T)
    u = pd.DataFrame({"x": 1000 / tau_us, "y": D, "ey": sD}).drop_duplicates("x")
    write_processed(u.round(4), "bursts_width_points.csv")
    k0, sk0, chi = wlsq_origin(u["x"], u["y"], u["ey"])
    RES.add("цуги: k (Dnu = k/tau)", k0, sk0, "", f"теория 1, chi2/dof = {chi:.2f}")
    macro("kBurst", k0, sk0)
    v = pd.DataFrame({"x": 1 / T, "y": dnu}).drop_duplicates("x")
    write_processed(v.round(4), "bursts_spacing_points.csv")
    k1, sk1 = lsq_origin(v["x"], v["y"])
    RES.add("цуги: k (dnu = k/T)", k1, sk1, "", "теория 1")
    macro("kBurstSpacing", k1, max(sk1, 1e-3), 3)
    p = D * tau_us / 1000
    RES.add("цуги: Dnu*tau (среднее)", p.mean(), p.std(ddof=1) / np.sqrt(len(p)), "")
    macro("burstProd", p.mean(), max(p.std(ddof=1) / np.sqrt(len(p)), 0.01), 2)


# ==========================================================================
# Г. АМ
# ==========================================================================
def process_am() -> None:
    header("Г. Амплитудно-модулированный сигнал")
    df = read_screen("am_lines.csv").set_index("line")
    a0 = df.loc["main", "a_mV"]
    s = df["sigma_a_mV"].iloc[0]
    rows = []
    for key, name in [("left", "$\\nu_0-\\nu_{\\text{мод}}$"), ("main", "$\\nu_0$"),
                      ("right", "$\\nu_0+\\nu_{\\text{мод}}$")]:
        r = df.loc[key]
        rows.append({"line": name, "nu_kHz": f"{r['nu_kHz']:.2f}".replace(".", "{,}"),
                     "a_mV": f"{r['a_mV']:.1f}".replace(".", "{,}"),
                     "ratio": f"{r['a_mV'] / a0:.3f}".replace(".", "{,}")})
    write_processed(pd.DataFrame(rows), "am_processed.csv")
    side = 0.5 * (df.loc["left", "a_mV"] + df.loc["right", "a_mV"])
    ratio = side / a0
    sr = ratio * np.hypot(s / side / np.sqrt(2), s / a0)
    m = 2 * ratio
    RES.add("a_бок/a_осн", ratio, sr, "")
    RES.add("m = 2 a_бок/a_осн", m, 2 * sr, "")
    nu_mod = 0.5 * (df.loc["right", "nu_kHz"] - df.loc["left", "nu_kHz"])
    snm = df["sigma_nu_kHz"].iloc[0] / np.sqrt(2)
    RES.add("nu_мод по спектру", nu_mod, snm, "кГц")
    RES.add("nu_0 по спектру", df.loc["main", "nu_kHz"], df["sigma_nu_kHz"].iloc[0], "кГц")
    macro("amRatio", ratio, sr, 3)
    macro("amM", m, 2 * sr, 3)
    macro("amNumod", nu_mod, snm, 2)
    macro("amNuzero", df.loc["main", "nu_kHz"], df["sigma_nu_kHz"].iloc[0], 2)
    macro("amMain", a0, s, 1)
    macro("amSide", side, s, 1)


# ==========================================================================
# Е. RC-цепочка
# ==========================================================================
def process_filter() -> None:
    header("Е. Фильтрация сигнала RC-цепочкой")
    w = read_screen("filter_waveform.csv")
    t, u = w["t_us"].to_numpy(), w["U_mV"].to_numpy()

    # вершины и впадины
    imax = [i for i in range(5, len(u) - 5) if u[i] == u[i - 5:i + 6].max() and u[i] > 80]
    imin = [i for i in range(5, len(u) - 5) if u[i] == u[i - 5:i + 6].min() and u[i] < -60]
    tmax = np.array([t[i] for i in imax])
    tmin = []
    for tm in tmax:                         # ближайшая впадина перед вершиной
        c = [t[i] for i in imin if tm - 0.5 < t[i] < tm]
        if c:
            tmin.append(max(c))
    tmin = np.array(tmin)
    T = float(np.mean(np.diff(tmax)))
    sT = float(np.std(np.diff(tmax), ddof=1) / np.sqrt(len(tmax) - 1))
    RES.add("T (осциллограмма)", T, sT, "мкс")

    # спад: U = U_low + (U_start - U_low) exp(-t/tau_RC)
    f = lambda x, ul, du, rc: ul + du * np.exp(-x / rc)
    rcs, uls = [], []
    for tm in tmax:
        nxt = [x for x in tmin if x > tm]
        if not nxt:
            continue
        m = (t > tm + 0.05) & (t < nxt[0] - 0.05)
        p, _ = curve_fit(f, t[m] - tm, u[m], p0=[-200, 280, 3])
        rcs.append(p[2])
        uls.append(p[0])
    rcs = np.array(rcs)
    tau_rc = rcs.mean()
    s_rc = rcs.std(ddof=1) / np.sqrt(len(rcs))
    u_low = float(np.mean(uls))
    RES.add("tau_RC (спад, среднее по периодам)", tau_rc, s_rc, "мкс")

    # подъём: U = U_high - (U_high - U_bot) exp(-t/tau_RC) при известном tau_RC
    uh, taus = [], []
    for tb in tmin:
        top = [x for x in tmax if x > tb][0]
        m = (t >= tb + 0.01) & (t <= top - 0.01)
        g = lambda x, h, d: h - d * np.exp(-x / tau_rc)
        p, _ = curve_fit(g, t[m] - tb, u[m], p0=[3000, 3100])
        uh.append(p[0])
        ub = u[np.argmin(np.abs(t - tb))]
        ut = u[np.argmin(np.abs(t - top))]
        taus.append(tau_rc * np.log((p[0] - ub) / (p[0] - ut)))
    U0 = float(np.mean(uh) - u_low)
    tau_p = float(np.mean(taus))
    s_taup = float(np.std(taus, ddof=1) / np.sqrt(len(taus)))
    RES.add("высота импульса на входе U0", U0, float(np.std(uh, ddof=1)), "мВ")
    RES.add("длительность импульса tau", tau_p, s_taup, "мкс")
    macro("filtT", T, max(sT, 0.002), 3)
    macro("filtTau", tau_p, max(s_taup, 0.005), 3)
    macro("filtRC", tau_rc, s_rc, 2)
    macro("filtUzero", U0 / 1000, np.std(uh, ddof=1) / 1000, 2)
    macro("filtNuRC", 1000 / (2 * np.pi * tau_rc), 1000 * s_rc / (2 * np.pi * tau_rc ** 2), 1)
    macro("filtNuRCinv", 1000 / tau_rc, 1000 * s_rc / tau_rc ** 2, 0)

    # спектр
    sp = read_screen("filter_lines.csv")
    n = sp["n"].to_numpy(float)
    nu = sp["nu_MHz"].to_numpy(float) * 1000         # кГц
    a = sp["a_mV"].to_numpy(float)
    k, sk = lsq_origin(n, nu)
    RES.add("nu_1 (спектр)", k, sk, "кГц")
    macro("filtNuOne", k, sk, 1)
    macro("filtTspec", 1000 / k, 1000 * sk / k ** 2, 3)
    q = tau_p / T
    ain = rect_model(n, q) / rect_model(1, q)
    Kexp = (a / a[0]) / ain
    Kth = lambda v, rc: np.sqrt(1 + (2 * np.pi * v[0] * rc / 1000) ** 2) / np.sqrt(1 + (2 * np.pi * v * rc / 1000) ** 2)
    kth = Kth(nu, tau_rc)
    out = pd.DataFrame({
        "n": n.astype(int), "nu_kHz": np.round(nu, 1), "a_mV": a,
        "ain_rel": np.round(ain, 3), "K_exp": np.round(Kexp, 3),
        "K_theor": np.round(kth, 3),
        "dev_percent": [f"{x:+.0f}" for x in (Kexp - kth) / kth * 100],
    })
    print(out.to_string(index=False))
    write_processed(out, "filter_processed.csv")
    dev = np.abs((Kexp - kth) / kth)[1:] * 100
    macro("filtDevMedian", float(np.median(dev)), None, 0)
    macro("filtDevMax", float(dev.max()), None, 0)
    macro("filtWnu", 2 * np.pi * k * tau_rc / 1000, None, 1)
    macro("filtQ", q, None, 3)


def main() -> None:
    process_harmonics()
    process_meander()
    process_width()
    process_spacing()
    process_bursts()
    process_am()
    process_filter()
    header("Итоговые результаты")
    RES.save()
    save_macros()


if __name__ == "__main__":
    main()
