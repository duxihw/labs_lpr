"""Расчёт по журналу 23.11.2026. Входные единицы сохраняются в data/raw."""
from __future__ import annotations

import numpy as np
import pandas as pd

from journal_table import write_journal_table
from lab_utils import PROC_DIR, Results, read_raw, tex_num, tex_pm, write_processed
from plot_data import main as plot_all

E = 1.602176634e-19


def slope(x, y):
    """МНК через начало координат."""
    return np.dot(x, y) / np.dot(x, x)


def residual_error(x, y, k):
    return np.linalg.norm(y - k*x) / np.sqrt((len(x)-1) * np.dot(x, x))


def voltage(df, prefix):
    return df[prefix + "_uV"].fillna(df[prefix + "_mV"] * 1000).to_numpy()


def load():
    cal = read_raw("calibration.csv")
    series = read_raw("hall_series.csv").sort_values("series")
    hall = read_raw("hall.csv").sort_values(["series", "I_M_A"]).reset_index(drop=True)
    rev = read_raw("hall_reverse.csv").sort_values("I_M_A")
    longitudinal = read_raw("resistivity.csv")
    params = read_raw("setup_params.csv").set_index("param")
    group = series.set_index("series").index.get_indexer(hall.series)
    if (group < 0).any():
        raise ValueError("Точка без серии")
    pairs = hall.index[group == len(series)-1].to_numpy()
    if not np.array_equal(hall.I_M_A.iloc[pairs], rev.I_M_A.iloc[1:]) or rev.I_M_A.iloc[0] != 0:
        raise ValueError("Токи прямой и обратной серий не совпадают")
    if not (longitudinal.I_mA == series.I_mA.iloc[-1]).all():
        raise ValueError("Продольный ток отличается от тока последней серии")
    geom = params.loc[["a_mm", "l_mm", "L35_mm", "SN_cm2"]]
    v = dict(geom=geom.value.to_numpy(), im_cal=cal.I_M_A.to_numpy(), phi=cal.Phi_mWb.to_numpy(),
             current=series.I_mA.to_numpy(), im=hall.I_M_A.to_numpy(), u=voltage(hall, "U34"),
             zero=voltage(series, "U0"), rev=rev.U34_minus_uV.to_numpy(),
             long=longitudinal.U35_mV.to_numpy(), drift=0.,
             gain_u=1., gain_phi=1., gain_im=1., gain_i=1.,
             residual=np.zeros(len(series)), residual_K=0., residual_rev=0.)
    v = {name: np.array(value, dtype=float) for name, value in v.items()}
    # Погрешности отсчёта независимы; у каждого прибора один общий множитель.
    errors = dict(geom=geom.sigma.to_numpy(), im_cal=0.01, phi=0.1, current=0.1,
                  im=0.01, u=1., zero=1., rev=1., long=0.001, drift=np.ptp(v['long'])/2,
                  gain_u=.03, gain_phi=.03, gain_im=.03, gain_i=.03)
    return v, errors, group, pairs, series.series.to_numpy()


def calculate(v, group, pairs):
    a, width, length, sn = v['geom']
    field = abs(v['phi'] * v['gain_phi']) * 10 / sn
    xcal = v['im_cal'] * v['gain_im']
    im = v['im'] * v['gain_im']
    if im.min() < xcal.min() or im.max() > xcal.max():
        raise ValueError("Ток вне градуировки")
    B = np.interp(im, xcal, field)
    current = v['current'] * v['gain_i']
    U = v['u'] * v['gain_u']
    zero = v['zero'] * v['gain_u']
    signal = U - zero[group]
    k = np.array([slope(B[group == i], signal[group == i]) for i in range(len(current))])
    k = k + v['residual']   # погрешность наклона по остаткам в отдельной серии
    K = slope(current, k) * 1e-3 + v['residual_K']  # В/(Тл А)
    rev = v['rev'] * v['gain_u']
    odd = (U[pairs] - rev[1:])/2
    zero_odd = (zero[-1] - rev[0])/2
    odd_corr = odd - zero_odd
    krev = slope(B[pairs], odd_corr) + v['residual_rev']
    rh = abs(K)*a*1e-3
    u35 = np.mean(v['long']) * v['gain_u'] + v['drift']
    resistance = abs(u35) / current[-1]
    rho = resistance * a * width / length * 1e-3
    return dict(field=field, xcal=xcal, im=im, current=current, B=B, U=U, zero=zero,
                signal=signal, k=k, K=K, rev=rev, odd=odd, zero_odd=zero_odd,
                odd_corr=odd_corr, even=(U[pairs]+rev[1:])/2, krev=krev,
                rh=rh, n=1/(E*rh), u35=u35, R=resistance,
                rho=rho, sigma=1/rho, mu=rh/rho)


def propagate(v, errors, group, pairs):
    """J diag(u²) Jᵀ: общий вход возмущается один раз для всех зависимых величин."""
    y = calculate(v, group, pairs)
    errors['residual'] = np.array([
        residual_error(y['B'][group == i], y['signal'][group == i], k)
        for i, k in enumerate(y['k'])])
    errors['residual_K'] = residual_error(y['current'], y['k'] * 1e-3, y['K'])
    errors['residual_rev'] = residual_error(y['B'][pairs], y['odd_corr'], y['krev'])
    var = {name: np.zeros_like(value) for name, value in y.items()}
    for name, error in errors.items():
        for idx in np.ndindex(v[name].shape):
            s = np.broadcast_to(error, v[name].shape)[idx]
            step = 1e-5*max(1, abs(v[name][idx]))
            base = v[name][idx]
            v[name][idx] = base + step
            plus = calculate(v, group, pairs)
            v[name][idx] = base - step
            minus = calculate(v, group, pairs)
            v[name][idx] = base
            for key in var:
                var[key] += ((plus[key] - minus[key]) * (s/(2*step)))**2
    return y, {key: np.sqrt(value) for key, value in var.items()}


def main():
    v, errors, group, pairs, series = load()
    write_journal_table()
    y, s = propagate(v, errors, group, pairs)
    def write(name, **columns):
        write_processed(pd.DataFrame(columns).round(9), name + ".csv")

    write("calibration_processed", I_M_A=y['xcal'], dI_A=s['xcal'], Phi_mWb=v['phi'],
          B_T=y['field'], dB_T=s['field'])
    write("hall_processed", series=series[group], I_mA=y['current'][group], I_M_A=y['im'],
          B_T=y['B'], dB_T=s['B'], U0_uV=y['zero'][group], U34_uV=y['U'],
          U_perp_uV=y['signal'], dU_perp_uV=s['signal'])
    write("slopes", series=series, I_mA=y['current'], dI_mA=s['current'],
          U0_uV=y['zero'], k_mV_T=y['k']/1e3, sigma_k=s['k']/1e3,
          n_points=np.bincount(group))
    write("reverse_processed", I_M_A=np.r_[0, y['im'][pairs]], B_T=np.r_[0, y['B'][pairs]],
          dB_T=np.r_[0, s['B'][pairs]], Uplus_uV=np.r_[y['zero'][-1], y['U'][pairs]],
          Uminus_uV=y['rev'], Uodd_uV=np.r_[y['zero_odd'], y['odd']],
          Ueven_uV=np.r_[(y['zero'][-1]+y['rev'][0])/2, y['even']],
          Ucorr_uV=np.r_[0, y['odd_corr']], dUcorr_uV=np.r_[0, s['odd_corr']],
          Udirect_uV=np.r_[0, y['signal'][pairs]], dUdirect_uV=np.r_[0, s['signal'][pairs]])
    write("resistivity_processed", I_mA=y['current'][-1], U35_mV=v['long'],
          R35_Ohm=abs(v['long'])/y['current'][-1])
    res = Results()
    macros = {}
    for key, macro, unit, factor in [
        ('K','K','В/(А·Тл)',1), ('krev','Krev','мВ/Тл',.001),
        ('rh','RHabs','м³/Кл',1), ('n','N','м⁻³',1), ('n','Ncm','см⁻³',1e-6),
        ('R','Rtf','Ом',1), ('rho','Rho','Ом·м',1), ('sigma','Sigma','См/м',1),
        ('mu','Mu','см²/(В·с)',1e4), ('mu','MuSI','м²/(В·с)',1)]:
        res.add(macro, y[key]*factor, s[key]*factor, unit)
        macros["Res"+macro] = tex_pm(y[key]*factor, s[key]*factor)
        macros["Rel"+macro.replace("RHabs","RH")] = f"{100*s[key]/abs(y[key]):.1f}".replace(".", "{,}")
    macros["SlopeRows"] = "\n".join(
        rf"\({tex_num(i,1)}\) & \({z:.0f}\) & \({tex_pm(k/1000,sk/1000)}\) \\"
        for i,z,k,sk in zip(y['current'],y['zero'],y['k'],s['k']))
    def rows(values, formats):
        return "\n".join(" & ".join(
            r"\(" + format(value, fmt).replace(".", "{,}") + r"\)"
            for value, fmt in zip(row, formats)) + r" \\" for row in zip(*values))

    macros["CalibrationRows"] = rows([y['xcal'], v['phi'], y['field']], [".2f",".1f",".3f"])
    macros["ReverseRows"] = rows([np.r_[0, y['im'][pairs]], np.r_[y['zero'][-1], y['U'][pairs]],
        y['rev'], np.r_[y['zero_odd'], y['odd']], np.r_[0, y['odd_corr']]],
        [".2f", ".0f", ".0f", ".1f", ".1f"])
    res.save()
    lines = ["% Создано scripts/process_data.py."] + [
        rf"\newcommand*{{\{key}}}{{{val}}}" for key,val in macros.items()]
    (PROC_DIR/"results.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    plot_all()


if __name__ == "__main__":
    main()
