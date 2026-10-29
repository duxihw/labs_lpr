"""Обработка данных лабораторной работы 3.4.5.

    «Петля гистерезиса (динамический метод)»

Читает   data/raw/*.csv
Пишет    data/processed/*.csv        -- таблицы и кривые для отчёта
         data/processed/results.tex  -- макросы \\res{...} с итоговыми числами
         data/processed/fit_results.csv

Запуск (из каталога работы):  python scripts/process_data.py

Единицы отсчёта
---------------
Все длины на экране -- в МАЛЫХ делениях сетки (1 большое = 5 малых).
Коэффициенты отклонения Kx, Ky -- мВ на БОЛЬШОЕ деление (положение переключателя).

Формулы
-------
Канал X:  M_x = 2*sqrt(2)*I_eff/[2X]      (А на малое деление при Kx = 50 мВ/дел)
          H   = (N0/2piR) * M_x * (Kx/50) * [X]
Канал Y:  K'_y = 2*sqrt(2)*U_eff/[2Y]     (В на малое деление при Ky = 200 мВ/дел)
          B   = tau/(S N_и) * K'_y * (Ky/200) * [Y]
Петля:    [2X], [2Y] -- двойные амплитуды  =>  H_s = h*[2X_s]/2 и т.д.
Нач. кривая: [X], [Y] -- координаты вершины от центра (одинарная амплитуда).
RC:       tau = U_вх / (omega0 U_вых)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab_utils import (
    Results,
    TexResults,
    get_param,
    num_str,
    read_params,
    read_raw,
    write_processed,
)

MU0 = 4e-7 * np.pi
RES = Results()
TEX = TexResults()
MIN_STEP_Y = 1.5      # мин. приращение по Y (дел.) при оценке mu_диф


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------
def calib_x(params):
    header("Калибровка канала X (п.9)")
    df = read_raw("calib_x.csv")
    d = get_param(params, "div_err", 0.5)
    R0 = get_param(params, "R0", np.nan)
    Mx = 2 * np.sqrt(2) * df["I_eff_mA"] / df["X2"] * 1e-3          # А/мал.дел
    Mx50 = Mx * 50 / df["Kx"]
    write_processed(pd.DataFrame({
        "Kx": df["Kx"], "I_eff_mA": df["I_eff_mA"], "X2": df["X2"],
        "Mx": [num_str(v * 1e3, sig=4) for v in Mx]}), "calib_x_processed.csv")
    m = Mx50.mean()
    sem = Mx50.std(ddof=1) / np.sqrt(len(Mx50)) / m
    eps = float(np.hypot(sem, d / df["X2"].mean()))
    Mx_nom = (50e-3 / 5) / R0
    RES.add("M_x", m * 1e3, m * eps * 1e3, "mA/div", "Kx = 50 мВ/дел, малое деление")
    TEX.add("Mx", m * 1e3, m * eps * 1e3)
    TEX.add("epsMx", eps * 100, sig=2)
    TEX.add("MxNom", Mx_nom * 1e3)
    TEX.add("MxDev", abs(m - Mx_nom) / Mx_nom * 100, sig=2)
    TEX.add("R0eff", 10e-3 / m, 10e-3 / m * eps)
    TEX.add("R0", R0, sig=2)
    return m, eps


def calib_y(params):
    header("Проверка канала Y (п.10)")
    df = read_raw("calib_y.csv")
    d = get_param(params, "div_err", 0.5)
    r = df.iloc[0]
    Ky1 = 2 * np.sqrt(2) * r["U_eff_mV"] * 1e-3 / r["Y2"]          # В/мал.дел
    nom = r["Ky"] * 1e-3 / 5
    eps = d / r["Y2"]
    RES.add("K'_y", Ky1 * 1e3, Ky1 * eps * 1e3, "mV/div", "при 0.2 В/дел")
    TEX.add("Kyp", Ky1 * 1e3, Ky1 * eps * 1e3)
    TEX.add("KyNom", nom * 1e3)
    TEX.add("KyDev", abs(Ky1 - nom) / nom * 100, sig=2)
    TEX.add("epsKy", eps * 100, sig=2)
    return Ky1 * 200 / r["Ky"], eps        # В/мал.дел при 200 мВ/дел


def rc(params):
    header("Постоянная времени RC-цепочки (пп.11-13)")
    df = read_raw("rc.csv")
    d = get_param(params, "div_err", 0.5)
    nu0 = get_param(params, "nu0", 50)
    R = get_param(params, "R_int", np.nan) * 1e3
    C = get_param(params, "C_int", np.nan) * 1e-6
    r = df.iloc[0]
    Uin = r["Ky_in"] * r["Y2_in"] / 5            # мВ, размах
    Uout = r["Ky_out"] * r["Y2_out"] / 5
    tau = Uin / (2 * np.pi * nu0 * Uout)
    # отсчёт + погрешность коэффициентов двух разных диапазонов (~3 %)
    eps = np.sqrt((d / r["Y2_in"]) ** 2 + (d / r["Y2_out"]) ** 2 + 2 * 0.03**2)
    tau_rc = R * C
    write_processed(pd.DataFrame([{
        "sig": "вход", "Ky": int(r["Ky_in"]), "Y2": r["Y2_in"], "U": num_str(Uin, sig=3)},
        {"sig": "выход", "Ky": int(r["Ky_out"]), "Y2": r["Y2_out"], "U": num_str(Uout, sig=3)}]),
        "rc_processed.csv")
    RES.add("tau_exp", tau, tau * eps, "s")
    RES.add("tau_RC", tau_rc, unit="s")
    TEX.add("UinPP", Uin / 1e3, sig=3)
    TEX.add("UoutPP", Uout, sig=3)
    TEX.add("ratioU", Uin / Uout, sig=3)
    TEX.add("tauExp", tau, tau * eps)
    TEX.add("tauRC", tau_rc, sig=2)
    TEX.add("tauDev", abs(tau - tau_rc) / tau_rc * 100, sig=2)
    TEX.add("omegaTau", 2 * np.pi * nu0 * tau_rc, sig=3)
    TEX.add("omegaTauExp", 2 * np.pi * nu0 * tau, sig=3)
    TEX.add("Rint", R / 1e3, sig=2)
    TEX.add("Cint", C * 1e6, sig=2)
    return tau_rc


# --------------------------------------------------------------------------
def coefs(sr, Kx, Ky, Mx50, Ky200, tau):
    """Цены малого деления: h (А/м), b (Тл) при данных Kx, Ky (мВ/дел)."""
    h = sr["NL"] * Mx50 * Kx / 50
    b = tau / (sr["SN"] * 1e-4) * Ky200 * Ky / 200
    return h, b


def coef_table(samples, Mx50, Ky200, tau):
    rows = []
    for s, sr in samples.iterrows():
        h50, b200 = coefs(sr, 50, 200, Mx50, Ky200, tau)
        rows.append({"sample": s, "material": sr["material"],
                     "NL": f"{sr['NL']:g}", "SN": f"{sr['SN']:g}",
                     "h50": num_str(h50, sig=3), "b200": num_str(b200, sig=3)})
        TEX.add(f"h{s}", h50, sig=3)
        TEX.add(f"b{s}", b200, sig=3)
    write_processed(pd.DataFrame(rows), "coefs.csv")


def loops(params, samples, Mx50, epsx, Ky200, epsy, tau, fname, out, prefix):
    header(f"Петли гистерезиса ({fname})")
    d = get_param(params, "div_err", 0.5)
    df = read_raw(fname)
    write_processed(df.map(lambda v: f"{v:g}" if isinstance(v, float) else v),
                    fname.replace(".csv", "_raw.csv"))   # копия для отчёта
    rows = []
    for _, r in df.iterrows():
        s = int(r["sample"])
        sr = samples.loc[s]
        Kxs = r.get("Kx_s", r.get("Kx"))
        Kys = r.get("Ky_s", r.get("Ky"))
        Kxc = r.get("Kx_c", Kxs)
        Kyr = r.get("Ky_r", Kys)
        hs, bs = coefs(sr, Kxs, Kys, Mx50, Ky200, tau)
        hc, _ = coefs(sr, Kxc, 200, Mx50, Ky200, tau)
        _, br = coefs(sr, 50, Kyr, Mx50, Ky200, tau)

        def amp(k, L2, e):
            v = k * L2 / 2
            return v, v * np.hypot(e, d / L2)

        Hs, dHs = amp(hs, r["X2s"], epsx)
        Bs, dBs = amp(bs, r["Y2s"], epsy)
        Hc, dHc = amp(hc, r["X2c"], epsx)
        Br, dBr = amp(br, r["Y2r"], epsy)
        rows.append({
            "sample": s, "material": sr["material"],
            "Hs": num_str(Hs, dHs), "dHs": num_str(dHs, dHs),
            "Bs": num_str(Bs, dBs), "dBs": num_str(dBs, dBs),
            "Hc": num_str(Hc, dHc), "dHc": num_str(dHc, dHc),
            "Br": num_str(Br, dBr), "dBr": num_str(dBr, dBr),
            "ratio": num_str(Br / Bs, sig=2)})
        for key, v, e in (("Hs", Hs, dHs), ("Bs", Bs, dBs), ("Hc", Hc, dHc), ("Br", Br, dBr)):
            RES.add(f"{prefix}{key}_{s}", v, e)
            TEX.add(f"{prefix}{key}{s}", v, e)
        TEX.add(f"{prefix}ratio{s}", Br / Bs, sig=2)
    write_processed(pd.DataFrame(rows), out)


def initial(params, samples, Mx50, epsx, Ky200, epsy, tau):
    header("Начальные кривые намагничивания (п.17)")
    d = get_param(params, "div_err", 0.5)
    df = read_raw("initial_curve.csv")
    for s, sr in samples.iterrows():
        sub = df[df["sample"] == s].reset_index(drop=True)
        h, b = coefs(sr, sub["Kx"], sub["Ky"], Mx50, Ky200, tau)
        H = (h * sub["X"]).to_numpy(float)
        B = (b * sub["Y"]).to_numpy(float)
        write_processed(pd.DataFrame({
            "Kx": sub["Kx"].astype(int), "Ky": sub["Ky"].astype(int),
            "X": sub["X"], "Y": sub["Y"],
            "H": [num_str(v, sig=3) for v in H],
            "B": [num_str(v, sig=3) for v in B]}), f"initial_{s}.csv")
        # для графика: с началом координат
        Hp, Bp = np.r_[0.0, H[H > 0]], np.r_[0.0, B[H > 0]]
        write_processed(pd.DataFrame({"H": Hp, "B": Bp}), f"initial_plot_{s}.csv")

        # mu_нач: МНК через начало координат по трём первым точкам с H > 0
        m = H > 0
        H3, B3 = H[m][:3], B[m][:3]
        k = np.sum(H3 * B3) / np.sum(H3**2) if np.any(B3 > 0) else np.nan
        bb = (b.to_numpy(float)[m][:3] if hasattr(b, "to_numpy") else b)
        # прямая через начало координат должна описывать точки в пределах
        # 1 дел.; иначе (резкий скачок B, как у пермаллоя) mu_нач не определяется
        if np.isfinite(k) and np.max(np.abs(B3 - k * H3) / bb) <= 1.0:
            TEX.add(f"muInit{s}", k / MU0, sig=2)
            RES.add(f"mu_init_{s}", k / MU0)
        # mu_диф: по соседним точкам, приращение по Y >= MIN_STEP_Y дел.
        best = None
        for i in range(len(H) - 1):
            dY = B[i + 1] - B[i]
            if H[i + 1] <= H[i] or dY <= 0:
                continue
            ny = dY / b.iloc[i + 1] if hasattr(b, "iloc") else dY / b
            if ny < MIN_STEP_Y - 1e-9:
                continue
            mu = dY / (H[i + 1] - H[i]) / MU0
            nx = (H[i + 1] - H[i]) / (h.iloc[i + 1] if hasattr(h, "iloc") else h)
            e = np.sqrt((d * np.sqrt(2) / ny) ** 2 + (d * np.sqrt(2) / nx) ** 2
                        + epsx**2 + epsy**2)
            if best is None or mu > best[0]:
                best = (mu, mu * e, (H[i] + H[i + 1]) / 2)
        if best:
            RES.add(f"mu_diff_max_{s}", best[0], best[1], comment=f"H ~ {best[2]:.3g}")
            TEX.add(f"muMax{s}", best[0], best[1])
            TEX.add(f"HmuMax{s}", best[2], sig=2)
        TEX.add(f"Hend{s}", H.max(), sig=3)
        TEX.add(f"Bend{s}", B.max(), sig=3)


def main() -> None:
    params = read_params()
    Mx50, epsx = calib_x(params)
    Ky200, epsy = calib_y(params)
    tau = rc(params)
    samples = read_raw("samples.csv").set_index("sample")
    coef_table(samples, Mx50, Ky200, tau)
    loops(params, samples, Mx50, epsx, Ky200, epsy, tau,
          "loops.csv", "loops_processed.csv", "")
    loops(params, samples, Mx50, epsx, Ky200, epsy, tau,
          "loops_screen.csv", "loops_screen_processed.csv", "scr")
    initial(params, samples, Mx50, epsx, Ky200, epsy, tau)
    header("Сохранение итогов")
    RES.save()
    TEX.save()


if __name__ == "__main__":
    main()
