"""
Лабораторная работа 1.1.1: Изучение равноускоренного движения

Скрипт обработки экспериментальных данных:
чтение data/raw/ -> расчёт -> data/processed/ + графики для отчёта (report/images/)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "common" / "scripts"))
from plotting import plot_with_fit  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
IMAGES_DIR = Path(__file__).resolve().parent.parent / "report" / "images"


def main():
    # TODO: загрузить сырые данные, например:
    # df = pd.read_csv(RAW_DIR / "measurements.csv")

    # TODO: обработать данные, посчитать погрешности

    # TODO: сохранить обработанные данные и графики
    # df.to_csv(PROCESSED_DIR / "results.csv", index=False)
    # fig, ax = plot_with_fit(...)
    # fig.savefig(IMAGES_DIR / "plot.png")

    print("Обработка данных лабораторной 1.1.1 завершена")


if __name__ == "__main__":
    main()
