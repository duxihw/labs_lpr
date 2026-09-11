"""
Обработка данных лабораторной 3.2.6 "Изучение гальванометра".

Читает сырые данные измерений (data/raw/data_table.csv: R, x1, x2, Theta),
считает критическое сопротивление R_cr по каждому измерению и сохраняет
результат в data/processed/data_table.csv — именно этот файл подключается
таблицей в report/main.tex.

Запуск:
    cd 3.2.6/scripts
    pip install -r requirements.txt
    python process_data.py
"""

import numpy as np
import pandas as pd

RAW_PATH = "../data/raw/data_table.csv"
PROCESSED_PATH = "../data/processed/data_table.csv"

# Внутреннее сопротивление гальванометра, кОм (R_0 = 560 Ом)
R_0 = 0.560


def main() -> None:
    df = pd.read_csv(RAW_PATH)

    # R_cr = (R + R_0) / sqrt((2*pi/Theta)^2 + 1) - R_0
    df["R_cr"] = (df["R"] + R_0) / np.sqrt((2 * np.pi / df["Theta"]) ** 2 + 1) - R_0
    df["R_cr"] = df["R_cr"].round(3)

    print("R_cr mean =", df["R_cr"].mean().round(3), "кОм")
    print("R_cr std  =", df["R_cr"].std().round(3), "кОм")

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Сохранено: {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
