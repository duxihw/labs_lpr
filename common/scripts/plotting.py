"""
Общие вспомогательные функции для построения графиков и обработки данных,
используемые в нескольких лабораторных работах.
"""

import numpy as np
import matplotlib.pyplot as plt


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Простая линейная регрессия методом наименьших квадратов.

    Возвращает (k, b) для y = k*x + b.
    """
    k, b = np.polyfit(x, y, 1)
    return k, b


def plot_with_fit(x, y, yerr=None, xlabel="", ylabel="", title="", savepath=None):
    """Строит точки эксперимента с погрешностями и линию тренда."""
    k, b = linear_fit(x, y)
    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=yerr, fmt="o", label="Экспериментальные точки")
    xs = np.linspace(min(x), max(x), 100)
    ax.plot(xs, k * xs + b, "-", label=f"Аппроксимация: y = {k:.4f}x + {b:.4f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig, ax
