"""Обработка данных работы 3.3.4 «Эффект Холла в полупроводниках».

Читает data/raw/*.csv (сырые данные никогда не перезаписываются) и пишет:
  data/processed/calibration_processed.csv  -- градуировка B(I_M)
  data/processed/hall_processed.csv         -- U_perp = U34 - U0 и B для всех точек
  data/processed/slopes.csv                 -- k = dU_perp/dB для каждого тока I
  data/processed/reverse_processed.csv      -- U_perp = (U+ - U-)/2 (обращение поля)
  data/processed/resistivity_processed.csv  -- U35(I)
  data/processed/fit_results.csv            -- все итоговые числа
  data/processed/results.tex                -- те же числа в виде LaTeX-макросов,
                                               подключается из report/main.tex

Запуск:  python scripts/process_data.py   (из папки 3.3.4)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab_utils import PROC_DIR, Results, lsq, read_raw, tex_num, tex_pm, write_processed

E = 1.602176634e-19       # Кл
K_B = 1.380649e-23        # Дж/К
M_E = 9.1093837e-31       # кг


# --------------------------------------------------------------------------
def read_setup() -> dict[str, tuple[float, float]]:
    """setup_params.csv -> {param: (value, sigma)}; пустые поля -> nan."""
    df = read_raw("setup_params.csv")
    out: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        out[str(row["param"]).strip()] = (
            pd.to_numeric(row.get("value"), errors="coerce"),
            pd.to_numeric(row.get("sigma"), errors="coerce"),
        )
    return out


def par(p, name, default=np.nan):
    v, s = p.get(name, (np.nan, np.nan))
    v = default if not np.isfinite(v) else v
    s = 0.0 if not np.isfinite(s) else s
    return float(v), float(s)


def rel(v, s):
    return s / abs(v) if np.isfinite(v) and v != 0 else 0.0


# --------------------------------------------------------------------------
def calibration(p, res: Results):
    """Градуировка электромагнита. Возвращает функцию B(I_M) [мТ] или None."""
    print("\n[1] Градуировка электромагнита")
    df = read_raw("calibration.csv")
    cols = ["I_M_A", "B_mT"]
    if df.empty:
        write_processed(df, "calibration_processed.csv", cols)
        return None

    df = df.apply(pd.to_numeric, errors="coerce")
    if "B_mT" not in df or df["B_mT"].isna().all():
        sn, _ = par(p, "SN_cm2")
        if not np.isfinite(sn):
            print("  [!] нет ни B_mT, ни SN_cm2 -- градуировку пересчитать нельзя")
            write_processed(pd.DataFrame(), "calibration_processed.csv", cols)
            return None
        # Phi [мВб] = B [Тл] * SN [м^2]  ->  B [мТ] = Phi [мВб] / SN [м^2]
        df["B_mT"] = df["Phi_mWb"] / (sn * 1e-4)
        print(f"  B = Phi / SN, SN = {sn} см^2")

    df = df.dropna(subset=["I_M_A", "B_mT"]).sort_values("I_M_A")
    out = df.copy()
    out["B_mT"] = out["B_mT"].round(1)
    write_processed(out, "calibration_processed.csv", cols)
    res.add("calib_points", len(df), comment="точек градуировки")
    res.add("B_max", df["B_mT"].abs().max(), unit="мТ", comment="максимальное поле")

    x, y = df["I_M_A"].to_numpy(), df["B_mT"].to_numpy()
    # Кусочно-линейная интерполяция (п. 12). Знак тока магнита: при
    # обращённом поле (K1 в положении 2) ток вводится положительным,
    # знак учитывается отдельно.
    return lambda im: np.interp(np.abs(im), x, y)


# --------------------------------------------------------------------------
def hall(B_of, p, res: Results):
    """ЭДС Холла: U_perp(B) для каждого тока, наклоны k(I), постоянная Холла."""
    print("\n[2] ЭДС Холла")
    cols = ["series", "I_mA", "U0_mV", "I_M_A", "B_mT", "U34_mV", "U_perp_mV"]
    scols = ["series", "I_mA", "U0_mV", "k_mV_T", "sigma_k", "b_mV", "sigma_b", "n_points"]
    ser = read_raw("hall_series.csv")
    pts = read_raw("hall.csv")
    if ser.empty or pts.empty or B_of is None:
        write_processed(pd.DataFrame(), "hall_processed.csv", cols)
        write_processed(pd.DataFrame(), "slopes.csv", scols)
        return {}

    df = pts.merge(ser, on="series", how="left")
    df["B_mT"] = B_of(df["I_M_A"].to_numpy())
    df["U_perp_mV"] = df["U34_mV"] - df["U0_mV"]                   # формула (1)
    df = df.sort_values(["series", "I_M_A"])
    out = df.copy()
    out["B_mT"] = out["B_mT"].round(1)
    out["U_perp_mV"] = out["U_perp_mV"].round(4)
    write_processed(out, "hall_processed.csv", cols)

    rows = []
    for s, g in df.groupby("series"):
        B_T = g["B_mT"].to_numpy() / 1e3
        k, sk, b, sb = lsq(B_T, g["U_perp_mV"].to_numpy())
        rows.append(dict(series=s, I_mA=g["I_mA"].iloc[0], U0_mV=g["U0_mV"].iloc[0],
                         k_mV_T=k, sigma_k=sk,
                         b_mV=b, sigma_b=sb, n_points=len(g)))
        print(f"  серия {s}: I = {g['I_mA'].iloc[0]} мА, k = {k:.4g} +- {sk:.2g} мВ/Тл")
    sl = pd.DataFrame(rows, columns=scols)
    out = sl.copy()
    for c in ["k_mV_T", "sigma_k", "b_mV", "sigma_b"]:
        out[c] = out[c].map(lambda v: float(f"{v:.4g}") if np.isfinite(v) else np.nan)
    write_processed(out, "slopes.csv", scols)

    # k(I): наклон K = R_H / h  [мВ/(Тл*мА) = В/(Тл*А)]
    K, sK, bK, sbK = lsq(sl["I_mA"], sl["k_mV_T"])
    res.add("K", K, sK, "В/(Тл·А)", "наклон k(I)")
    res.add("K_b", bK, sbK, "мВ/Тл", "свободный член k(I)")

    h, sh = par(p, "h_mm")
    epsB, _ = par(p, "eps_B", 0.0)
    if not np.isfinite(h):
        print("  [!] не задана толщина h -- R_H не вычисляется")
        return {"K": (K, sK)}
    RH = K * h * 1e-3                                               # м^3/Кл
    eRH = np.sqrt(rel(K, sK) ** 2 + rel(h, sh) ** 2 + epsB ** 2)
    res.add("R_H", RH, abs(RH) * eRH, "м^3/Кл", "постоянная Холла (знак -- в системе отсчёта установки)")
    n = 1 / (E * abs(RH))
    res.add("n", n, n * eRH, "м^-3", "концентрация носителей")
    return {"K": (K, sK), "RH": (RH, abs(RH) * eRH), "n": (n, n * eRH), "epsB": epsB}


# --------------------------------------------------------------------------
def reverse(B_of, res: Results):
    """Контроль: ЭДС Холла через обращение поля, U_perp = (U+ - U-)/2."""
    print("\n[3] Обращение поля")
    cols = ["I_M_A", "B_mT", "U34_plus_mV", "U34_minus_mV", "U_perp_mV", "U_ohm_mV"]
    df = read_raw("hall_reverse.csv")
    if df.empty or B_of is None:
        write_processed(pd.DataFrame(), "reverse_processed.csv", cols)
        return
    df["B_mT"] = B_of(df["I_M_A"].to_numpy())
    df["U_perp_mV"] = (df["U34_plus_mV"] - df["U34_minus_mV"]) / 2
    df["U_ohm_mV"] = (df["U34_plus_mV"] + df["U34_minus_mV"]) / 2   # омическая добавка ~ U0
    out = df.sort_values("I_M_A").copy()
    out["B_mT"] = out["B_mT"].round(1)
    out[["U_perp_mV", "U_ohm_mV"]] = out[["U_perp_mV", "U_ohm_mV"]].round(4)
    write_processed(out, "reverse_processed.csv", cols)
    k, sk, b, sb = lsq(df["B_mT"] / 1e3, df["U_perp_mV"])
    res.add("k_rev", k, sk, "мВ/Тл", "наклон U_perp(B) методом обращения поля")
    res.add("U_ohm", df["U_ohm_mV"].mean(), df["U_ohm_mV"].std(ddof=1) if len(df) > 1 else np.nan,
            "мВ", "полусумма (U+ + U-)/2 -- омическая добавка")


# --------------------------------------------------------------------------
def resistivity(p, res: Results):
    """rho_0 = U35 a h / (I l)  -- формула (2)."""
    print("\n[4] Удельное сопротивление")
    cols = ["I_mA", "U35_mV", "R35_Ohm"]
    df = read_raw("resistivity.csv")
    if df.empty:
        write_processed(df, "resistivity_processed.csv", cols)
        return {}
    df["R35_Ohm"] = df["U35_mV"] / df["I_mA"]
    out = df.copy()
    out["R35_Ohm"] = out["R35_Ohm"].map(lambda v: float(f"{v:.4g}"))
    write_processed(out, "resistivity_processed.csv", cols)

    if len(df) >= 3:
        R, sR, _, _ = lsq(df["I_mA"], df["U35_mV"])
    else:
        U, I = df["U35_mV"].iloc[-1], df["I_mA"].iloc[-1]
        dU, _ = par(p, "dU_mV", 0.0)
        dI, _ = par(p, "dI_mA", 0.0)
        R = U / I
        sR = abs(R) * np.hypot(rel(U, dU), rel(I, dI))
    res.add("R35", R, sR, "Ом", "сопротивление участка 3-5")

    a, sa = par(p, "a_mm")
    h, sh = par(p, "h_mm")
    l, sl_ = par(p, "l_mm")
    if not all(np.isfinite([a, h, l])):
        print("  [!] не заданы размеры образца -- rho не вычисляется")
        return {}
    rho = abs(R) * a * h / l * 1e-3                                  # Ом*м
    erho = np.sqrt(rel(R, sR) ** 2 + rel(a, sa) ** 2 + rel(h, sh) ** 2 + rel(l, sl_) ** 2)
    res.add("rho", rho, rho * erho, "Ом·м", "удельное сопротивление")
    sigma = 1 / rho
    res.add("sigma0", sigma, sigma * erho, "См/м", "удельная проводимость")
    return {"rho": (rho, rho * erho), "sigma": (sigma, sigma * erho)}


# --------------------------------------------------------------------------
def mobility(hl, rs, p, res: Results):
    """mu = sigma0 * |R_H| (из sigma = e n mu) и оценки к контрольному вопросу 6."""
    print("\n[5] Подвижность и кинетические оценки")
    if "RH" not in hl or "sigma" not in rs:
        return {}
    RH, sRH = hl["RH"]
    sg, ssg = rs["sigma"]
    mu = sg * abs(RH)                                                # м^2/(В*с)
    smu = mu * np.hypot(rel(RH, sRH), rel(sg, ssg))
    res.add("mu", mu * 1e4, smu * 1e4, "см^2/(В·с)", "подвижность")

    T, _ = par(p, "T_K", 295.0)
    meff, _ = par(p, "m_eff", 1.0)
    m = meff * M_E
    nu = E / (m * mu)                                                # 1/с
    vT = np.sqrt(3 * K_B * T / m)                                    # м/с
    lam = vT / nu                                                    # м
    D = mu * K_B * T / E                                             # м^2/с (соотношение Эйнштейна)
    res.add("nu_coll", nu, nu * rel(mu, smu), "1/с", "частота столкновений e/(m mu)")
    res.add("v_T", vT, unit="м/с", comment="тепловая скорость sqrt(3kT/m)")
    res.add("lambda", lam * 1e9, lam * 1e9 * rel(mu, smu), "нм", "длина свободного пробега")
    res.add("D", D * 1e4, D * 1e4 * rel(mu, smu), "см^2/с", "коэффициент диффузии")
    return dict(mu=(mu * 1e4, smu * 1e4), nu=(nu, nu * rel(mu, smu)), vT=(vT, 0),
                lam=(lam * 1e9, lam * 1e9 * rel(mu, smu)), D=(D * 1e4, D * 1e4 * rel(mu, smu)),
                T=(T, 0), meff=(meff, 0))


# --------------------------------------------------------------------------
def write_macros(res: Results, p) -> None:
    """data/processed/results.tex: \\newcommand для каждого итогового числа."""
    vals = {r["quantity"]: (r["value"] if r["value"] is not None else np.nan,
                            r["sigma"] if r["sigma"] is not None else np.nan) for r in res.rows}

    def pm(key, scale=1.0):
        v, s = vals.get(key, (np.nan, np.nan))
        return tex_pm(v * scale, s * scale)

    def num(key, scale=1.0):
        return tex_num(vals.get(key, (np.nan, np.nan))[0] * scale)

    macros = {
        "ResK": pm("K"),
        "ResKb": pm("K_b"),
        "ResRH": pm("R_H"),
        "ResRHabs": tex_pm(abs(vals.get("R_H", (np.nan,))[0]), vals.get("R_H", (0, np.nan))[1]),
        "ResN": pm("n"),
        "ResNcm": pm("n", 1e-6),
        "ResRtf": pm("R35"),
        "ResRho": pm("rho"),
        "ResRhoOhmCm": pm("rho", 1e2),
        "ResSigma": pm("sigma0"),
        "ResMu": pm("mu"),
        "ResKrev": pm("k_rev"),
        "ResUohm": pm("U_ohm"),
        "ResNu": pm("nu_coll"),
        "ResVT": num("v_T"),
        "ResLambda": pm("lambda"),
        "ResD": pm("D"),
        "ResBmax": num("B_max"),
        "ResCalibN": f"{vals['calib_points'][0]:.0f}" if "calib_points" in vals else r"\textbf{???}",
    }
    for name, key in [("ParA", "a_mm"), ("ParH", "h_mm"), ("ParL", "l_mm"), ("ParSN", "SN_cm2"),
                      ("ParT", "T_K"), ("ParMeff", "m_eff")]:
        v, s = p.get(key, (np.nan, np.nan))
        macros[name] = tex_pm(v, s) if np.isfinite(s) and s > 0 else tex_num(v)
    epsB = p.get("eps_B", (np.nan, np.nan))[0]
    macros["ParEpsB"] = tex_num(epsB * 100, 2) if np.isfinite(epsB) else r"\textbf{???}"

    lines = ["% Сгенерировано scripts/process_data.py -- вручную не править."]
    lines += [rf"\newcommand*{{\{k}}}{{{v}}}" for k, v in macros.items()]
    (PROC_DIR / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  -> data/processed/results.tex")


# --------------------------------------------------------------------------
def main() -> None:
    res = Results()
    p = read_setup()
    B_of = calibration(p, res)
    hl = hall(B_of, p, res)
    reverse(B_of, res)
    rs = resistivity(p, res)
    mobility(hl, rs, p, res)
    print("\n[итог]")
    res.save()
    write_macros(res, p)


if __name__ == "__main__":
    main()
