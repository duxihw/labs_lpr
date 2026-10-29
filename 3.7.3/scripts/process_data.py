"""Обработка сырых данных лабораторной работы 3.7.3.

    «Длинные линии»

Читает   data/raw/*.csv
Пишет    data/processed/*.csv   -- таблицы и точки для графиков отчёта;
         data/processed/fit_results.csv -- все итоговые числа;
         data/processed/results.tex     -- те же числа в виде LaTeX-макросов,
                                           которые подставляются в отчёт.
Печатает сводку с промежуточными и окончательными результатами.

Запуск (из каталога работы):
    python scripts/process_data.py

Сырые данные никогда не изменяются. Если какая-то таблица ещё пустая,
соответствующие processed-файлы создаются только с заголовком, а в отчёте
вместо чисел остаются метки «???».

Все вычисления ведутся в СИ. Формулы методички (в СГС) переписаны так:
    k^2 - alpha^2 = L'C' * omega^2,           V_ф = 1/sqrt(L'C'),
    R_0 = sqrt(L'/C'),
    C' = 2*pi*eps0*eps / ln(D/d),             L' = mu0*mu*ln(D/d) / (2*pi),
    alpha = R_x / (2 R_0),   R_x = (2/d) * sqrt(mu0*nu / (pi*sigma))   (скин-эффект)
 => метод А:  alpha  = s2 * sqrt(nu),   sigma = mu0 / (pi * (s2*d*R_0)^2)
    метод Б:  alpha*k = s3 * nu^(3/2),  sigma = mu0/pi * (2*pi*C' / (d*s3))^2
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

EPS0 = 8.8541878128e-12   # Ф/м
MU0 = 1.25663706212e-6    # Гн/м
C_LIGHT = 2.99792458e8    # м/с
SIGMA_CU = 5.96e7         # См/м, табличная проводимость меди (20 °C)

RES = Results()
TEX: dict[str, str] = {}   # имя макроса -> LaTeX-строка (без $)
NUM: dict[str, str] = {}   # имя -> «голое» число для формул pgfplots


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------
# Оформление чисел для LaTeX
# --------------------------------------------------------------------------
def _comma(text: str) -> str:
    return text.replace(".", "{,}")


def tex_pm(value: float, sigma: float = np.nan, power: int = 0) -> str:
    """'(1{,}23 \\pm 0{,}05)\\cdot 10^{8}' или '???', если значения нет."""
    if value is None or not np.isfinite(value):
        return r"\qq"
    v = value / 10.0**power
    if sigma is not None and np.isfinite(sigma) and sigma > 0:
        s = sigma / 10.0**power
        digits = max(0, int(-np.floor(np.log10(s))) + 1)
        body = _comma(f"{v:.{digits}f} \\pm {s:.{digits}f}")
        body = f"({body})" if power else body
    else:
        body = _comma(f"{v:.4g}")
    return body + (f"\\cdot 10^{{{power}}}" if power else "")


def put(name: str, value: float, sigma: float = np.nan, power: int = 0,
        unit: str = "", comment: str = "") -> None:
    """Запоминает результат и в fit_results.csv, и в results.tex."""
    RES.add(name, value, sigma, unit, comment)
    TEX[name] = tex_pm(value, sigma, power)
    if value is not None and np.isfinite(value):
        NUM[name] = f"{value:.6g}"


def save_tex() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["% Сгенерировано scripts/process_data.py -- вручную не править.",
             "% Макрос \\res{имя} выдаёт число (в математическом режиме)."]
    for name, text in TEX.items():
        lines.append(f"\\expandafter\\def\\csname res@{name}\\endcsname{{{text}}}")
    lines.append("% Макрос \\resnum{имя} -- число без оформления (для pgfplots).")
    for name, text in NUM.items():
        lines.append(f"\\expandafter\\def\\csname resnum@{name}\\endcsname{{{text}}}")
    (PROC_DIR / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> data/processed/results.tex ({len(TEX)} макросов)")


def round_sig(x, n: int = 4) -> list[str]:
    """Строки с n значащими цифрами (для таблиц отчёта); NaN -> ''."""
    out = []
    for v in np.asarray(x, dtype=float):
        if not np.isfinite(v):
            out.append("")
        elif v == 0:
            out.append("0")
        else:
            digits = max(0, n - 1 - int(np.floor(np.log10(abs(v)))))
            out.append(f"{v:.{digits}f}")
    return out


# --------------------------------------------------------------------------
# Часть I. Фазовая и групповая скорость по резонансам кабеля
# --------------------------------------------------------------------------
# Множитель p в  V = p * l * Δν,  где Δν -- шаг между соседними резонансами.
# Во всех режимах совпадение сигналов в начале и в конце линии наступает,
# когда время пробега l/V кратно периоду: l/V = n/ν  -> p = 1.
# (Опыт это подтверждает: при 50 Ом и 1 МОм резонансы синусоиды совпадают.)
P_FACTOR = {
    ("sine", "matched"): 1,
    ("sine", "open"): 1,
    ("pulse", "matched"): 1,
    ("pulse", "open"): 1,
}
SERIES_NAMES = {
    ("sine", "matched"): "SineMatched",
    ("sine", "open"): "SineOpen",
    ("pulse", "matched"): "PulseMatched",
    ("pulse", "open"): "PulseOpen",
}


def process_cable_resonances(l_m: float) -> None:
    header("I. Резонансы коаксиального кабеля -> V_ф, V_гр")
    raw = read_raw("cable_resonances.csv")
    cols = ["n", "nu_MHz", "nu_fit_MHz", "resid_kHz", "nu_per_n"]
    for (signal, load), tag in SERIES_NAMES.items():
        if raw.empty:
            df = pd.DataFrame(columns=cols)
        else:
            mask = (raw["signal"].astype(str).str.strip() == signal) & \
                   (raw["load"].astype(str).str.strip() == load)
            df = raw.loc[mask, ["n", "nu_MHz"]].astype(float).sort_values("n")
            df["n"] = df["n"].astype(int)
        print(f"  {signal}/{load}: {len(df)} резонансов")

        k, sk, b, sb = lsq(df["n"], df["nu_MHz"]) if len(df) else (np.nan,) * 4
        df = df.copy()
        df["nu_fit_MHz"] = k * df["n"] + b if np.isfinite(k) else np.nan
        df["resid_kHz"] = (df["nu_MHz"] - df["nu_fit_MHz"]) * 1e3
        df["nu_per_n"] = df["nu_MHz"] / df["n"]
        write_processed(df, f"cable_res_{signal}_{load}.csv", cols)

        p = P_FACTOR[(signal, load)]
        if not (np.isfinite(sk) and sk > 0):
            # точки легли точно на прямую (частоты записаны с шагом 1 МГц) --
            # берём погрешность по разрешению записи частоты: 0,05 МГц на
            # весь диапазон резонансов
            nmax = df["n"].max() if len(df) else np.nan
            sk = 0.05 / (nmax - 1) if np.isfinite(nmax) and nmax > 1 else np.nan
        V, sV = p * l_m * k * 1e6, p * l_m * sk * 1e6
        put(f"dnu{tag}", k, sk, 0, "MHz", f"шаг резонансов, {signal}/{load}")
        put(f"V{tag}", V, sV, 8, "m/s", f"скорость (p = {p}), {signal}/{load}")
        put(f"Vratio{tag}", V / C_LIGHT, sV / C_LIGHT, 0, "",
            f"V/c, {signal}/{load}")


# --------------------------------------------------------------------------
# Часть II. АЧХ и ФЧХ кабеля, погонные параметры, проводимость
# --------------------------------------------------------------------------
def process_afc(l_m: float, params: dict) -> None:
    header("II. АЧХ и ФЧХ кабеля")
    R0 = get_param(params, "R0_cable", 50.0)
    d = get_param(params, "d_inner", 1.37) * 1e-3
    D = get_param(params, "D_outer", 4.6) * 1e-3
    lnDd = np.log(D / d)
    print(f"  l = {l_m} м, R0 = {R0} Ом, d = {d*1e3} мм, D = {D*1e3} мм")

    cols = ["nu_MHz", "U0_V", "Un_V", "dx_ns", "m",
            "Kamp", "dphi_rad", "alpha_m", "k_m", "Vph_1e8",
            "x1", "y1", "x2", "y2", "x3", "y3",
            "y1_fit", "y2_fit", "y3_fit"]
    df = read_raw("afc_pfc.csv")
    if df.empty:
        write_processed(pd.DataFrame(), "afc_processed.csv", cols)
        write_processed(pd.DataFrame(), "afc_table.csv", cols, na_rep="")
        for name in ["LCprod", "Vph", "Lx", "Cx", "eps", "mu", "epsmu",
                     "sigmaA", "sigmaB", "sigmaAratio", "sigmaBratio"]:
            TEX[name] = r"\qq"
        return

    df = df.astype(float).sort_values("nu_MHz").reset_index(drop=True)
    nu = df["nu_MHz"].to_numpy() * 1e6
    m = df["m"].fillna(0).to_numpy()
    dphi = 2 * np.pi * (nu * df["dx_ns"].to_numpy() * 1e-9 + m)
    alpha = np.log(df["U0_V"] / df["Un_V"]).to_numpy() / l_m
    k = dphi / l_m
    omega = 2 * np.pi * nu

    df["Kamp"] = df["Un_V"] / df["U0_V"]
    df["dphi_rad"] = dphi
    df["alpha_m"] = alpha
    df["k_m"] = k
    df["Vph_1e8"] = omega / k / 1e8
    # x1 = omega^2 [1e15 рад^2/с^2], y1 = k^2 - alpha^2 [1/м^2]
    df["x1"] = omega**2 / 1e15
    df["y1"] = k**2 - alpha**2
    # x2 = sqrt(nu) [1e3 Гц^0.5], y2 = alpha [1e-3 1/м]
    df["x2"] = np.sqrt(nu) / 1e3
    df["y2"] = alpha * 1e3
    # x3 = nu^(3/2) [1e9 Гц^1.5], y3 = alpha*k [1e-3 1/м^2]
    df["x3"] = nu**1.5 / 1e9
    df["y3"] = alpha * k * 1e3

    # --- Часть I обработки: L'C' по наклону y1(x1) ---
    print("  Наклон y1 = L'C' * x1 (y = kx + b):")
    s1, ss1, b1, sb1 = lsq(df["x1"], df["y1"])
    df["y1_fit"] = s1 * df["x1"] + b1
    LC, sLC = s1 / 1e15, ss1 / 1e15           # с^2/м^2
    put("LCprod", LC, sLC, -17, "s^2/m^2", "L'C' (наклон y1(x1))")
    put("b1", b1, sb1, 0, "1/m^2", "свободный член y1(x1)")
    Vph, sVph = 1 / np.sqrt(LC), 0.5 * sLC / LC / np.sqrt(LC)
    put("Vph", Vph, sVph, 8, "m/s", "V_ф = 1/sqrt(L'C')")
    put("Vphratio", Vph / C_LIGHT, sVph / C_LIGHT, 0, "", "V_ф / c")
    Lx, sLx = R0 * np.sqrt(LC), 0.5 * R0 * sLC / np.sqrt(LC)
    Cx, sCx = np.sqrt(LC) / R0, 0.5 * sLC / np.sqrt(LC) / R0
    put("Lx", Lx, sLx, -7, "H/m", "погонная индуктивность L' = R0*sqrt(L'C')")
    put("Cx", Cx, sCx, -10, "F/m", "погонная ёмкость C' = sqrt(L'C')/R0")
    put("LxNom", MU0 * lnDd / (2 * np.pi), np.nan, -7, "H/m",
        "L' по геометрии при mu = 1")
    eps, seps = Cx * lnDd / (2 * np.pi * EPS0), sCx * lnDd / (2 * np.pi * EPS0)
    mu, smu = 2 * np.pi * Lx / (MU0 * lnDd), 2 * np.pi * sLx / (MU0 * lnDd)
    put("eps", eps, seps, 0, "", "диэлектрическая проницаемость")
    put("mu", mu, smu, 0, "", "магнитная проницаемость")
    put("epsmu", LC * C_LIGHT**2, sLC * C_LIGHT**2, 0, "", "eps*mu = (c/V_ф)^2")

    # --- Часть II, метод А: alpha(sqrt(nu)) ---
    print("  Метод А: alpha = s2*sqrt(nu) + b:")
    s2, ss2, b2, sb2 = lsq(df["x2"], df["y2"])
    df["y2_fit"] = s2 * df["x2"] + b2
    s2SI, ss2SI = s2 * 1e-6, ss2 * 1e-6       # 1/(м*Гц^0.5)
    put("sTwo", s2SI, ss2SI, -9, "1/(m Hz^0.5)", "наклон alpha(sqrt(nu))")
    put("bTwo", b2 * 1e-3, sb2 * 1e-3, -3, "1/m", "свободный член alpha(sqrt(nu))")
    sigA = MU0 / (np.pi * (s2SI * d * R0) ** 2)
    put("sigmaA", sigA, 2 * sigA * ss2SI / abs(s2SI), 7, "S/m", "sigma, метод А")
    put("sigmaAratio", sigA / SIGMA_CU, 2 * sigA * ss2SI / abs(s2SI) / SIGMA_CU,
        0, "", "sigma_A / sigma_Cu")

    # --- метод Б: alpha*k(nu^1.5) ---
    print("  Метод Б: alpha*k = s3*nu^1.5 + b:")
    s3, ss3, b3, sb3 = lsq(df["x3"], df["y3"])
    df["y3_fit"] = s3 * df["x3"] + b3
    s3SI, ss3SI = s3 * 1e-12, ss3 * 1e-12     # 1/(м^2*Гц^1.5)
    put("sThree", s3SI, ss3SI, -12, "1/(m^2 Hz^1.5)", "наклон alpha*k(nu^1.5)")
    sigB = MU0 / np.pi * (2 * np.pi * Cx / (d * s3SI)) ** 2
    ssigB = sigB * np.sqrt((2 * ss3SI / s3SI) ** 2 + (2 * sCx / Cx) ** 2)
    put("sigmaB", sigB, ssigB, 7, "S/m", "sigma, метод Б")
    put("sigmaBratio", sigB / SIGMA_CU, ssigB / SIGMA_CU, 0, "", "sigma_B / sigma_Cu")

    write_processed(df, "afc_processed.csv", cols)
    # Округлённая копия для таблицы отчёта
    tab = df.copy()
    tab["m"] = tab["m"].fillna(0).astype(int)
    for c in cols[5:]:
        tab[c] = round_sig(tab[c], 4)
    write_processed(tab, "afc_table.csv", cols, na_rep="")


# --------------------------------------------------------------------------
# Модель длинной линии (LC-цепочка)
# --------------------------------------------------------------------------
def process_model(params: dict) -> None:
    header("Модель длинной линии")
    L = get_param(params, "L_cell", 1.5) * 1e-3
    C = get_param(params, "C_cell", 47.0) * 1e-9
    N = int(get_param(params, "N_cells", 10))
    nu0 = 1 / (np.pi * np.sqrt(L * C))
    R0 = np.sqrt(L / C)
    put("nuZero", nu0 / 1e3, np.nan, 0, "kHz", "предельная частота nu_0 = 1/(pi sqrt(LC))")
    put("RZeroModel", R0, np.nan, 0, "Ohm", "согласованная нагрузка sqrt(L/C)")
    put("RMatchModel", get_param(params, "R_match_model", np.nan), np.nan, 0, "Ohm",
        "согласованная нагрузка, подобранная на опыте")

    # --- п.3: сдвиг фаз в цепочке ---
    # Сдвиг dt измерялся между клеммами, разделёнными K звеньями
    # (K = links_phase в setup_params.csv; на низких частотах dt -> K*sqrt(LC)).
    # Осциллограф показывает лишь dt < T, поэтому к nu*dt добавляется целое
    # число периодов m (столбец m в model_phase.csv), подобранное так, чтобы
    # Δφ(ν) росла монотонно и непрерывно.
    K = int(get_param(params, "links_phase", 1))
    put("linksPhase", K, np.nan, 0, "", "число звеньев между клеммами при измерении сдвига фаз")
    put("dtLow", K * np.sqrt(L * C) * 1e6, np.nan, 0, "us", "K*sqrt(LC) -- предел dt при nu -> 0")
    cols = ["nu_kHz", "dt_us", "m", "dphi_tot", "dphi_rad", "dphi_chain", "dphi_cont"]
    df = read_raw("model_phase.csv")
    if not df.empty:
        df = df.astype(float).sort_values("nu_kHz")
        nu = df["nu_kHz"] * 1e3
        df["m"] = df["m"].fillna(0).astype(int) if "m" in df else 0
        df["dphi_tot"] = 2 * np.pi * (nu * df["dt_us"] * 1e-6 + df["m"])
        df["dphi_rad"] = df["dphi_tot"] / K
        df["dphi_chain"] = 2 * np.arcsin(np.clip(nu / nu0, 0, 1))
        df["dphi_cont"] = 2 * np.pi * nu * np.sqrt(L * C)
    write_processed(df, "model_phase_processed.csv", cols)

    # теоретические кривые для графика (плотная сетка)
    grid = np.linspace(0, nu0, 200)
    theory = pd.DataFrame({
        "nu_kHz": grid / 1e3,
        "dphi_chain": 2 * np.arcsin(np.clip(grid / nu0, 0, 1)),
        "dphi_cont": 2 * np.pi * grid * np.sqrt(L * C),
    })
    write_processed(theory, "model_phase_theory.csv")
    tab = df.copy()
    for c in cols[3:]:
        if c in tab:
            tab[c] = round_sig(tab[c], 4)
    write_processed(tab, "model_phase_table.csv", cols, na_rep="")

    # --- п.4: резонансные частоты ---
    # Генератор (50 Ом) много «мягче» R_0 модели, поэтому вход считаем
    # закороченным для стоячих волн; фазовый набег на ячейку theta:
    #   nu = nu_0 * sin(theta/2),
    #   выход замкнут (R = 0):    N*theta = m*pi           (m = 1 .. N-1)
    #   выход разомкнут (R = inf): N*theta = (m - 1/2)*pi   (m = 1 .. N)
    # В таблицы отчёта попадают только измеренные резонансы.
    raw = read_raw("model_resonances.csv")
    rows = []
    if not raw.empty:
        for _, r in raw.iterrows():
            load, mm, exp = str(r["load"]).strip(), int(r["n"]), float(r["nu_kHz"])
            theta = (mm if load == "short" else mm - 0.5) * np.pi / N
            th = nu0 * np.sin(theta / 2) / 1e3
            rows.append({"load": load, "n": mm, "nu_exp_kHz": exp,
                         "nu_theor_kHz": round(th, 2),
                         "dev_percent": round((exp - th) / th * 100, 0)})
    res = pd.DataFrame(rows, columns=["load", "n", "nu_exp_kHz", "nu_theor_kHz", "dev_percent"])
    rcols = ["n", "nu_exp_kHz", "nu_theor_kHz", "dev_percent"]
    for load in ["short", "open"]:
        part = res[res["load"] == load].sort_values("n")[rcols]
        part["dev_percent"] = part["dev_percent"].map(lambda v: f"{v:+.0f}")
        write_processed(part, f"model_res_{load}.csv", rcols)

    # --- п.5: распределение амплитуды по ячейкам ---
    # Теория для m-го резонанса (вход -- узел): U_n ∝ |sin(n*theta_m)|,
    # нормировка на максимум измеренных значений серии.
    dist = read_raw("model_distribution.csv")
    cols = ["cell"]
    if not dist.empty:
        dist = dist.astype(float).sort_values("cell")
        dist["cell"] = dist["cell"].astype(int)
        cells = dist["cell"].to_numpy()
        for i, (load, mm) in DIST_SERIES.items():
            col = f"U{i}_mV"
            if col not in dist or dist[col].isna().all():
                continue
            theta = (mm if load == "short" else mm - 0.5) * np.pi / N
            shape = np.abs(np.sin(cells * theta))
            dist[f"U{i}_th"] = (dist[col].max() * shape / shape.max()).round(1)
            cols += [col, f"U{i}_th"]
            name = ["", "One", "Two", "Three", "Four"][i]
            put(f"nuDist{name}", get_param(params, f"nu_dist_{i}", np.nan), np.nan, 0,
                "kHz", f"частота серии {i} распределения U(n) ({load}, m = {mm})")
    write_processed(dist, "model_distribution_processed.csv", cols)


# Режим и номер резонанса для серий распределения U(n) (п.5):
#   серия 1 -- 4,9 кГц: выход замкнут, первый резонанс;
#   серия 2 -- 2,1 кГц: выход разомкнут, первый резонанс.
DIST_SERIES = {1: ("short", 1), 2: ("open", 1)}


def main() -> None:
    params = read_params()
    l_m = get_param(params, "l_cable", 50.0)
    process_cable_resonances(l_m)
    process_afc(l_m, params)
    process_model(params)
    header("Сохранение итогов")
    RES.save()
    save_tex()


if __name__ == "__main__":
    main()
