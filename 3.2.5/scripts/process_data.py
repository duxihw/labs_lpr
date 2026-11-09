"""Обработка работы 3.2.5. Единицы входных величин указаны в заголовках CSV."""
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from lab_utils import lsq, read_params, read_raw, write_processed, TexResults

pi = np.pi


def crossing(f, y, level):
    """Первое пересечение уровня, линейная интерполяция без экстраполяции."""
    for i in range(len(f)-1):
        if y[i] == level:
            return f[i]
        if (y[i]-level)*(y[i+1]-level) <= 0:
            return f[i]+(f[i+1]-f[i])*(level-y[i])/(y[i+1]-y[i])
    return np.nan


def calculate(data, params):
    results, curves = {}, defaultdict(list)
    def point(name, x, y):
        curves[name].append((x, y))

    coil = data['lcr']
    coil = coil[coil.freq_Hz == params.get('lcr_Hz')]
    L = coil.L_mH.iloc[0]/1000 if len(coil) else np.nan
    RL = coil.RL_ohm.iloc[0] if len(coil) else np.nan
    periods = data['periods']
    zero = periods[periods.C_nF == 0]
    if len(zero):
        t0 = (zero.dt_us/zero.n).mean()*1e-6
        c0 = t0**2/(4*pi**2*L)
        results['Czero'] = c0*1e9
        for row in periods.itertuples():
            theory = np.sqrt(t0**2 + 4*pi**2*L*row.C_nF*1e-9)
            point('period', theory*1e6, row.dt_us/row.n)
    else:
        c0 = np.nan
    C = params.get('Cset_nF', np.nan)*1e-9+c0
    results['criticalTheory'] = 2*np.sqrt(L/C)
    results['criticalObserved'] = params.get('Rtransition', np.nan)+RL
    for (method, R), rows in data['pairs'].groupby(['method','R_ohm']):
        u, v = rows.Uk.abs(), rows.Ukn.abs()
        if method == 'rise':
            u, v = rows.U0-u, rows.U0-v
        theta = (np.log(u/v)/rows.n).mean()
        if method == 'free':
            point('critical', 1/(R+RL)**2, 1/theta**2)
        for i in (1,2):
            if R == params.get(f'R{i}'):
                results[f'{method}{i}'] = pi/theta
    for i in (1,2):
        R = params.get(f'R{i}', np.nan)
        results[f'calc{i}'] = np.sqrt(L/C)/(R+RL)
        scan = data['resonance']
        scan = scan[scan.R_ohm == R].sort_values('f_Hz')
        amplitude_scan = scan.dropna(subset=['f_Hz','Upp_V'])
        if len(amplitude_scan):
            f,u = amplitude_scan[['f_Hz','Upp_V']].to_numpy().T
            k = np.argmax(u)
            amplitude = u/u[k]
            curves[f'amp{i}'] = list(zip(f/f[k],amplitude))
            results[f'peakFreq{i}'] = f[k]
            results[f'peakV{i}'] = u[k]
            if 0 < k < len(f)-1:
                lo = crossing(f[:k+1][::-1],amplitude[:k+1][::-1],2**-.5)
                hi = crossing(f[k:],amplitude[k:],2**-.5)
                results[f'ach{i}'] = f[k]/(hi-lo)
        phase_scan = scan.dropna(subset=['f_Hz','delay_us'])
        if len(phase_scan):
            f,delay = phase_scan[['f_Hz','delay_us']].to_numpy().T
            phase = -2*pi*f*delay*1e-6
            curves[f'ph{i}'] = list(zip(2*pi*f,phase))
            levels = np.array([-1,-2,-3])*pi/4
            if np.median(phase)>0:
                levels += pi  # Положительная ветвь с опорой на синий канал.
            lo,mid,hi = [crossing(f,phase,level) for level in levels]
            results[f'phase{i}'] = mid/(hi-lo)
    return results, curves


def main():
    data = {name:read_raw(name) for name in
            ('periods.csv','lcr.csv','pairs.csv','resonance.csv')}
    data = {name[:-4]:frame for name,frame in data.items()}
    video = read_raw('video_resonance.csv')
    if not video.empty:
        scan = data['resonance']
        # Максимум из журнала оставляем для независимого сравнения.
        data['resonance'] = pd.concat([scan[~scan.R_ohm.isin(video.R_ohm)],
                                      video.reindex(columns=scan.columns)],ignore_index=True)
    # Каждая строка журнала задаёт одну пару крайних пиков через n периодов.
    journal = read_raw('journal_decay.csv')
    for row in journal.itertuples():
        peaks = [a for a in (row.U1,row.U2,row.U3,row.U4) if pd.notna(a)]
        data['pairs'].loc[len(data['pairs'])] = ['free',row.R_ohm,len(peaks)-1,peaks[0],np.nan,peaks[-1],np.nan,np.nan,np.nan]
    params = {k:v for k,v in read_params().items() if np.isfinite(v)}
    for frame in data.values():
        for col in frame.select_dtypes(include="number"):
            frame[col] = frame[col].astype(float)
    values, curves = calculate(data,params)
    samples, points = defaultdict(list), defaultdict(list)
    rng = np.random.default_rng(325)
    for _ in range(500 if any(frame[col].notna().any() for frame in data.values() for col in frame if col.startswith("s")) or any(k in params for k in ("sR","sCset_nF")) else 0):
        trial = {name:frame.copy() for name,frame in data.items()}
        for name,frame in trial.items():
            for col in frame:
                if 's'+col in frame:
                    frame[col] += rng.normal(0,frame['s'+col].fillna(0))
            if name == 'periods':
                frame.loc[data[name].C_nF == 0,'C_nF'] = 0
        p = dict(params)
        for key in ('Cset_nF','R1','R2','Rtransition'):
            if key in p:
                sigma = params.get('sCset_nF' if key=='Cset_nF' else 'sR',0)
                p[key] += rng.normal(0,sigma)
        # Сопротивление является ключом серии; одинаковая настройка получает общий сдвиг.
        for R in set(data['pairs'].R_ohm) | set(data['resonance'].R_ohm):
            shift = next((p[k]-R for k in ('R1','R2') if params.get(k)==R),
                         rng.normal(0,params.get('sR',0)))
            for name in ('pairs','resonance'):
                trial[name].loc[data[name].R_ohm==R,'R_ohm'] = R+shift
        numbers, graphs = calculate(trial,p)
        for key,value in numbers.items():
            samples[key].append(value)
        for key,array in graphs.items():
            if len(array):
                points[key].append(array)
    tex = TexResults()
    for key,value in values.items():
        if np.isfinite(value):
            sigma = np.nanstd(samples[key],ddof=1) if samples[key] else None
            tex.add(key,value,sigma)
    for key in ('period','critical','amp1','amp2','ph1','ph2'):
        frame = pd.DataFrame(curves[key],columns=['x','y']).dropna()
        if len(frame):
            errors = np.nanstd(points[key],axis=0,ddof=1) if points[key] else np.zeros((len(curves[key]),2))
            frame[['sx','sy']] = errors[frame.index]
            tex.add_text('has'+key,'1')
        write_processed(frame,key+'.csv',['x','y','sx','sy'])
    for name in ('period','critical'):
        frame = pd.DataFrame(curves[name], columns=['x','y']).dropna()
        if len(frame) < 3 or frame.x.nunique() < 2:
            continue
        k, sk, b, sb = lsq(frame.x,frame.y)
        tex.add(name+'Slope',k,sk)
        tex.add(name+'Intercept',b,sb)
        limits = np.array([frame.x.min(),frame.x.max()])
        line = pd.DataFrame({'x':limits,'y':k*limits+b})
        write_processed(line,name+'_fit.csv',['x','y'])
        if name == 'critical' and k > 0:
            tex.add('criticalFit',2*pi*np.sqrt(k),pi*sk/np.sqrt(k))
    peaks = read_raw('image_peaks.csv')
    for series in read_raw('photo_series.csv').itertuples():
        trace = peaks[peaks.source == series.source].sort_values('x_peak_px')
        if trace.empty:
            continue
        name = 'env'+series.source[4:8]
        period = np.median(trace.x_next_peak_px-trace.x_peak_px)
        # Середина измеренного цикла; ноль времени локален для каждого снимка.
        middle = (trace.x_peak_px+trace.x_next_peak_px)/2
        frame = pd.DataFrame({'x':(middle-middle.iloc[0])/period,
                              'y':trace.half_range_px/trace.half_range_px.max()})
        write_processed(frame,name+'.csv',['x','y'])
        tex.add_text('has'+name,'1')
    # Участок роста между курсорным меню слева и ограничением трассы краем экрана справа.
    growth = peaks[(peaks.source == 'IMG_2499.HEIC') & peaks.x_peak_px.between(107,217.5)]
    if len(growth) >= 4:
        period = np.median(growth.x_next_peak_px-growth.x_peak_px)
        cycles = (growth.x_peak_px-growth.x_peak_px.min())/period
        def buildup(n, plateau, deficit, theta):
            return plateau-deficit*np.exp(-theta*n)
        amp = growth.half_range_px.to_numpy()
        params_fit,cov = curve_fit(buildup,cycles,amp,
                                   p0=(amp.max()*1.05,amp.max()-amp.min(),0.35),
                                   bounds=(0,np.inf))
        theta = params_fit[2]
        tex.add('rise1',pi/theta,pi*np.sqrt(cov[2,2])/theta**2)
    # Спад после выключения; первый цикл границы цуга и хвост у шума исключены.
    if not peaks.empty:
        decay = peaks[(peaks.source == 'IMG_2500.HEIC') & peaks.x_peak_px.between(130,272)]
        if len(decay) >= 3:
            x = decay.x_peak_px.to_numpy()
            period_px = np.median(decay.x_next_peak_px-x)
            (slope,_),cov = np.polyfit((x-x[0])/period_px,np.log(decay.half_range_px),1,cov=True)
            tex.add('photoTheta',-slope,np.sqrt(cov[0,0]))
            tex.add('fall1',-pi/slope,pi*np.sqrt(cov[0,0])/slope**2)
    tex.save()


if __name__ == '__main__':
    main()
