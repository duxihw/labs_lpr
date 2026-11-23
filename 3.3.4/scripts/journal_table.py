"""Две компактные таблицы прямых отсчётов в исходных единицах."""
import csv
from itertools import zip_longest

from lab_utils import PROC_DIR, RAW_DIR


def raw_rows(name):
    with (RAW_DIR / name).open(encoding="utf-8-sig") as source:
        return list(csv.DictReader(line for line in source if not line.startswith("#")))


def number(value):
    return "$" + value.replace(".", "{,}") + "$"


def write_journal_table():
    points = raw_rows("hall.csv")
    series = raw_rows("hall_series.csv")
    lines = []
    for start in range(0, len(series), 4):
        batch = series[start:start + 4]
        groups = [[p for p in points if p["series"] == s["series"]] for s in batch]
        lines.extend([r"\begin{center}\small", r"\begin{tabular}{" + "cc" * len(batch) + r"}\hline"])
        lines.append(" & ".join(r"\multicolumn{2}{c}{" + number(s["I_mA"]) + " мА}"
                                for s in batch) + r"\\")
        units = ["мВ" if s["U0_mV"] else "мкВ" for s in batch]
        lines.append(" & ".join(r"$I_{\mathrm M}$, А & $U_{34}$, " + u for u in units) + r"\\\hline")
        for row in zip_longest(*groups):
            cells = []
            for point in row:
                cells.extend([number(point["I_M_A"]), number(point["U34_mV"] or point["U34_uV"])]
                             if point else ["", ""])
            lines.append(" & ".join(cells) + r"\\")
        lines.extend([r"\hline\end{tabular}", r"\end{center}"])
    (PROC_DIR / "journal_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
