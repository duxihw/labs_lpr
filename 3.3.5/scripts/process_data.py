"""Обработка работы 3.3.5. Первый элемент каждой выборки — измеренное значение."""
import csv
from decimal import Decimal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lab_utils import FIG_DIR, PROC_DIR, RAW_DIR, TexResults
from lab_utils import tex_num

rng = np.random.default_rng(335)
size = 4001
results = TexResults()


def rows(filename):
    with (RAW_DIR / (filename + ".csv")).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


params = {r["param"]: r["value"] for r in rows("setup_params") if r["value"]}
frac = float(params.get("frac", 0.03))
scales = {}


def noise():
    z = rng.normal(size=size)
    z[0] = 0
    return z


def reading(row, key, meter):
    raw = row[key].strip()
    value = float(raw)
    if row.get("d_" + key):
        return value + float(row["d_" + key]) * noise()
    if meter not in scales:
        scales[meter] = noise()
    digit = float(Decimal(1).scaleb(Decimal(raw).as_tuple().exponent))
    return value * (1 + frac * scales[meter]) + digit * noise()


def error(x):
    return np.std(x[1:], ddof=1)


def result(key, x):
    # Нормируем также малые величины перед округлением.
    exponent = int(np.floor(np.log10(abs(x[0])))) if x[0] else 0
    power = exponent if abs(exponent) >= 3 else 0
    value = tex_num(x[0] / 10.**power, error(x) / 10.**power)
    if power:
        value = rf"({value})\cdot10^{{{power}}}"
    results.add_text(key, value)
    if x[0]:
        results.add(key + "Pct", 100 * error(x) / abs(x[0]), sig=2)


def fit(points):
    x, y = map(np.stack, zip(*points))
    dx = x - x.mean(axis=0)
    if len(np.unique(x[:, 0])) < 2:
        raise ValueError("Для наклона нужны разные абсциссы")
    slope = (dx * (y - y.mean(axis=0))).sum(axis=0) / (dx**2).sum(axis=0)
    return slope, y.mean(axis=0) - slope * x.mean(axis=0)


def plot(name, xlabel, ylabel, datasets):
    path = FIG_DIR / (name + ".pdf")
    if not datasets:
        path.unlink(missing_ok=True)
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.7), layout="constrained")
    for label, points, line in datasets:
        x, y = zip(*points)
        centers = [v[0] for v in x]
        marker = ax.errorbar(centers, [v[0] for v in y], xerr=list(map(error, x)),
                             yerr=list(map(error, y)), fmt="o", ms=3, capsize=2,
                             elinewidth=.7, label=label)
        if line is not None:
            k, b = line
            ends = np.array([min(centers), max(centers)])
            ax.plot(ends, k[0] * ends + b[0], lw=.85, color=marker[0].get_color())
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=.2)
    if any(label for label, _, _ in datasets):
        ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


FIG_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
cal = []
sn = reading(params, "SN", "SN") if "SN" in params else None
for r in rows("calibration"):
    if bool(r["B_mT"]) == bool(r["flux_mWb"]):
        raise ValueError("Для каждой точки требуется один способ измерения поля")
    if r["B_mT"]:
        field = reading(r, "B_mT", "magnetometer") / 1000
    else:
        if sn is None or sn[0] <= 0:
            raise ValueError("Для потокосцепления требуется SN > 0")
        field = reading(r, "flux_mWb", "fluxmeter") / (1000 * sn)
    cal.append((reading(r, "Im_A", "A1"), field))
cal.sort(key=lambda pair: pair[0][0])
grid = np.array([x[0] for x, _ in cal])
if np.any(np.diff(grid) <= 0):
    raise ValueError("Повтор тока в градуировке: уточните ветвь или усредните повторы")
plot("calibration", "Ток электромагнита, А", "Индукция, Тл", [("", cal, None)] if cal else [])


def interpolate(x):
    if len(cal) < 2 or not grid[0] <= x[0] <= grid[-1]:
        raise ValueError(f"Ток {x[0]} А вне доступной градуировки")
    j = np.clip(np.searchsorted(grid, x[0]), 1, len(cal) - 1)
    (a, u), (b, v) = cal[j-1:j+1]
    # Производные линейной интерполяции в номинальном интервале.
    t = (x[0] - a[0]) / (b[0] - a[0])
    return (1-t)*u + t*v + (v[0]-u[0])/(b[0]-a[0]) * (x - (1-t)*a - t*b)


series = {}
for r in rows("series"):
    name = r["series"]
    if name in series or r["sample"] not in ("Cu", "Ag", "Zn"):
        raise ValueError(f"Проверьте идентификатор серии и материал: {name}")
    series[name] = (r["sample"], reading(r, "I_A", "A2"), reading(r, "U0_uV", "microV"))

groups = {}
hall_table = []
for r in rows("hall"):
    name, branch = r["series"], int(r["branch"])
    if branch not in (-1, 1):
        raise ValueError("branch должен быть +1 или -1")
    field = branch * interpolate(reading(r, "Im_A", "A1"))
    voltage = reading(r, "U24_uV", "microV") - series[name][2]
    groups.setdefault((name, branch), []).append((field, voltage))
    hall_table.append([name, branch, r["Im_A"], field[0], error(field),
                       voltage[0], error(voltage)])

with (PROC_DIR / "hall.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["series", "branch", "Im_A", "B_T", "dB_T", "u_uV", "du_uV"])
    writer.writerows(hall_table)

with (PROC_DIR / "calibration.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Im_A", "dIm_A", "B_T", "dB_T"])
    writer.writerows([x[0], error(x), y[0], error(y)] for x, y in cal)

rho = {}
for r in rows("resistivity"):
    m = r["sample"]
    if m in rho:
        raise ValueError(f"Для {m} ожидается одно продольное измерение")
    dims = [m + "_" + key for key in ("a", "h", "l")]
    if all(key in params for key in dims):
        a, h, length = [reading(params, key, key) for key in dims]
        rho[m] = (reading(r, "U34_mV", "milliV") / reading(r, "I_A", "A2")
                  * a * h / length * 1e-6, h)

summary = []
for m in ("Cu", "Ag", "Zn"):
    slopes, traces = [], []
    for (name, branch), points in groups.items():
        sample, current, _ = series[name]
        if sample != m:
            continue
        line = fit(points) if len({p[0][0] for p in points}) >= 2 else None
        direction = "прямое" if branch == 1 else "обратное"
        traces.append((f"{current[0]:g} А, {direction}", points, line))
        if line:
            k, b = line
            summary.append([m, current[0], branch, f"{k[0]:.4g}", f"{error(k):.2g}",
                            f"{b[0]:.3g}", f"{error(b):.2g}"])
            if branch == 1:
                slopes.append((current, k))
    plot("hall_" + m, "Индукция со знаком, Тл", "Поперечное напряжение, мкВ", traces)
    q = None
    if m == "Zn" and len(slopes) == 1:
        current, k = slopes[0]
        q = k / current
    elif len(slopes) >= 2:
        line = fit(slopes)
        q = line[0]
        result(m + "Slope", q)
        plot("slopes_" + m, "Ток образца, А", "Наклон, мкВ/Тл", [("", slopes, line)])
    else:
        plot("slopes_" + m, "", "", [])
    if m in rho:
        resist, h = rho[m]
        result(m + "Rho", resist)
        result(m + "Sigma", 1 / resist)
    if q is None or m + "_h" not in params:
        continue
    h = rho[m][1] if m in rho else reading(params, m + "_h", m + "_h")
    apparent = q * h * 1e-9
    orientation = params.get(m + "_orientation")
    if orientation is not None:
        sign = int(orientation)
        if sign not in (1, -1):
            raise ValueError("orientation должен быть +1 или -1")
        result(m + "RH", sign * apparent)
    result(m + "N", 1 / (1.602176634e-19 * abs(apparent)))
    if m in rho:
        result(m + "Mu", 1e4 * abs(apparent) / resist)

with (PROC_DIR / "slopes.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sample", "current", "branch", "k", "dk", "b", "db"])
    writer.writerows(summary)
if summary:
    results.add_text("hasSlopes", "1")
primary = {sample for sample, _, _ in series.values() if sample != "Zn"}
if len(primary) > 1:
    raise ValueError("В качестве основного образца выберите Cu или Ag")
if primary:
    results.add_text("primary", primary.pop())
results.save()
