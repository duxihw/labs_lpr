"""Обработка работы 3.2.5. Единицы входных величин указаны в заголовках CSV."""
from collections import defaultdict
import numpy as np
import pandas as pd
from lab_utils import read_params, read_raw, write_processed, TexResults

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
    if len(curves['critical']) >= 2:
        x,y = np.array(curves['critical']).T
        if np.isfinite(x).all() and np.ptp(x)>0:
            results['criticalFit'] = 2*pi*np.sqrt(np.polyfit(x,y,1)[0])
    for i in (1,2):
        R = params.get(f'R{i}', np.nan)
        results[f'calc{i}'] = np.sqrt(L/C)/(R+RL)
        scan = data['resonance']
        scan = scan[scan.R_ohm == R].sort_values('f_Hz')
        if len(scan) < 3:
            continue
        f, u, delay = scan[['f_Hz','Upp_V','delay_us']].to_numpy().T
        phase = -2*pi*f*delay*1e-6
        k = np.argmax(u)
        amplitude = u/u[k]
        curves[f'amp{i}'] = list(zip(f/f[k],amplitude))
        curves[f'ph{i}'] = list(zip(2*pi*f,phase))
        lo = crossing(f[:k+1][::-1],amplitude[:k+1][::-1],2**-.5)
        hi = crossing(f[k:],amplitude[k:],2**-.5)
        results[f'ach{i}'] = f[k]/(hi-lo)
        lo,mid,hi = [crossing(f,phase,-j*pi/4) for j in (1,2,3)]
        results[f'phase{i}'] = mid/(hi-lo)
    return results, curves


def main():
    data = {name:read_raw(name) for name in
            ('periods.csv','lcr.csv','pairs.csv','resonance.csv')}
    data = {name[:-4]:frame for name,frame in data.items()}
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
    peaks = read_raw('image_peaks.csv')
    # Спад после выключения; первый цикл границы цуга и хвост у шума исключены.
    if not peaks.empty:
        decay = peaks[(peaks.source == 'IMG_2500.HEIC') & peaks.x_peak_px.between(130,272)]
        if len(decay) >= 3:
            x = decay.x_peak_px.to_numpy()
            period_px = np.median(decay.x_next_peak_px-x)
            (slope,_),cov = np.polyfit((x-x[0])/period_px,np.log(decay.half_range_px),1,cov=True)
            tex.add('photoTheta',-slope,np.sqrt(cov[0,0]))
            tex.add('photoQ',-pi/slope,pi*np.sqrt(cov[0,0])/slope**2)
    tex.save()


if __name__ == '__main__':
    main()
