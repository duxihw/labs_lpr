"""
Обработка данных лабораторной 3.2.6 "Изучение гальванометра".

1. Декремент затухания (стационарная схема):
   data/raw/data_table.csv (R, x1, x2, Theta)  ->  data/processed/data_table.csv
   Для каждого измерения считается R_cr = (R + R0)/sqrt((2pi/Theta)^2 + 1) - R0.

2. Линеаризация баллистической кривой:
   data/raw/ballistic_mode_data.csv (R, l, sigma_l, R+R0)
       ->  data/processed/ballistic_linearization.csv
       ->  data/processed/ballistic_fit.csv   (параметры МНК-прямой и итоговые величины)

   Из l_0 = l * exp(Theta/4) находим декремент для каждого R:
       Theta = 4 ln(l_0 / l),
   где l_0 -- отброс разомкнутого гальванометра (R = inf).
   Из формулы 4pi^2/Theta^2 + 1 = (R + R0)^2 / (R_cr + R0)^2 следует линейная связь
       Y = 1/Theta^2 = k X + b,   X = (R + R0)^2,
       k = 1 / (4 pi^2 (R_cr + R0)^2),   b = -1/(4 pi^2),
   откуда R_cr = 1 / (2 pi sqrt(k)) - R0.

3. Баллистическая постоянная в критическом режиме
   C_q = 2a (R1/R2) C U0 / x_max^kp,  x_max^kp = x_max^св / e.

Запуск:
    cd 3.2.6/scripts
    pip install -r requirements.txt
    python process_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"

# ---------------- Параметры установки ----------------
R_0 = 0.560            # внутреннее сопротивление гальванометра, кОм
THETA_0 = 0.249        # декремент разомкнутого гальванометра
L_0 = 24.5             # отброс разомкнутого гальванометра (R = inf), см
SIGMA_L_0 = 0.5        # его погрешность, см
A = 0.98               # расстояние зеркальце-шкала, м
U_0 = 67 / 150 * 3     # напряжение, В
DIV_BALL = 1 / 50      # R1/R2 в баллистическом режиме
C_CAP = 2e-6           # ёмкость конденсатора, Ф
T_0 = 4.64             # период свободных колебаний, с


def decrement_table() -> None:
    df = pd.read_csv(RAW / "data_table.csv")
    df["R_cr"] = (df["R"] + R_0) / np.sqrt((2 * np.pi / df["Theta"]) ** 2 + 1) - R_0
    df["R_cr"] = df["R_cr"].round(3)
    print("[декремент] R_cr mean = %.3f кОм, std = %.3f кОм"
          % (df["R_cr"].mean(), df["R_cr"].std()))
    df.to_csv(PROCESSED / "data_table.csv", index=False)


def ballistic_linearization() -> float:
    df = pd.read_csv(RAW / "ballistic_mode_data.csv").sort_values("R", ascending=False)
    R, l, s_l = df["R"].to_numpy(), df["l"].to_numpy(), df["sigma_l"].to_numpy()

    theta = 4 * np.log(L_0 / l)
    s_theta = 4 * np.sqrt((s_l / l) ** 2 + (SIGMA_L_0 / L_0) ** 2)

    X = (R + R_0) ** 2                  # кОм^2
    Y = 1 / theta ** 2
    s_Y = 2 * s_theta / theta ** 3

    # МНК (невзвешенный) с оценкой погрешностей коэффициентов
    (k, b), cov = np.polyfit(X, Y, 1, cov=True)
    s_k, s_b = np.sqrt(np.diag(cov))

    R_cr = 1 / (2 * np.pi * np.sqrt(k)) - R_0
    s_R_cr = s_k / (4 * np.pi * k ** 1.5)

    out = pd.DataFrame({
        "R": R,
        "RR0": np.round(R + R_0, 2),
        "l": l,
        "sigma_l": s_l,
        "Theta": np.round(theta, 3),
        "sigma_Theta": np.round(s_theta, 3),
        "X": np.round(X, 2),
        "Y": np.round(Y, 4),
        "sigma_Y": np.round(s_Y, 4),
    })
    out.to_csv(PROCESSED / "ballistic_linearization.csv", index=False)

    # --- баллистическая постоянная ---
    x_free = L_0 * 10 * (1 + THETA_0 / 4)          # мм
    x_cr = x_free / np.e                           # мм
    q = C_CAP * U_0 * DIV_BALL                     # Кл
    C_q = q / (x_cr / (2 * A))                     # Кл/(мм/м)
    tau = R_0 * 1e3 * C_CAP                        # с
    # отброс, измеренный при найденном R_cr (линейная интерполяция)
    order = np.argsort(R)
    l_at_Rcr = np.interp(R_cr, R[order], l[order])

    pd.DataFrame([{
        "k": k, "sigma_k": s_k, "b": b, "sigma_b": s_b,
        "b_theory": -1 / (4 * np.pi ** 2),
        "R_cr": R_cr, "sigma_R_cr": s_R_cr,
        "x_free_mm": x_free, "x_cr_mm": x_cr, "l_at_Rcr_cm": l_at_Rcr,
        "q": q, "C_q": C_q, "tau": tau,
    }]).to_csv(PROCESSED / "ballistic_fit.csv", index=False)

    print("[линеаризация] Y = k X + b")
    print("    k = (%.4e +- %.1e) кОм^-2" % (k, s_k))
    print("    b = %.4f +- %.4f   (теория: %.4f)" % (b, s_b, -1 / (4 * np.pi ** 2)))
    print("    R_cr = %.2f +- %.2f кОм" % (R_cr, s_R_cr))
    print("    l(R_cr) по кривой = %.1f см, x_св/e = %.1f см" % (l_at_Rcr, x_cr / 10))
    print("[баллист.] x_св = %.1f мм, x_кр = %.1f мм, q = %.3e Кл" % (x_free, x_cr, q))
    print("    C_q = %.3e Кл/(мм/м), tau = %.2e с (T0 = %.2f с)" % (C_q, tau, T_0))
    return R_cr


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    decrement_table()
    ballistic_linearization()
    print("Сохранено в", PROCESSED)


if __name__ == "__main__":
    main()
