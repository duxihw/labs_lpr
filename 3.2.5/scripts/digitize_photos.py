"""Оцифровка жёлтой трассы. Координаты в системе обзора шириной 400 пикселей."""
from pathlib import Path
import csv
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

LAB = Path(__file__).resolve().parents[1]
IMG = LAB/'report/img'
RAW = LAB/'data/raw'
# Границы чистой области экрана (x1,y1,x2,y2), период в пикселях обзора.
REGIONS = {2496:(103,30,335,230,8.3),2499:(52,24,358,252,18.1),
           2500:(107,17,367,255,19.9),2502:(40,47,355,254,17.0),
           2503:(148,79,332,200,5.8)}
CROPS = {2489:('setup.jpg',(0,0,400,300)),2490:('spiral.png',(31,143,115,216)),
         2492:('phase.png',(21,76,208,214)),2496:('burst.png',(0,0,375,296)),
         2491:('wiring.jpg',(0,0,400,300)),2504:('generator.jpg',(125,71,310,236)),
         2499:('growth.jpg',(11,16,368,286)),2500:('decay.jpg',(0,5,385,287)),
         2502:('burst_fast.jpg',(27,35,367,274)),2503:('burst_detail.jpg',(56,57,347,279))}


def main(source):
    records = []
    for number,(name,box) in CROPS.items():
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'frame.jpg'
            subprocess.run(['heif-convert',str(source/f'IMG_{number}.HEIC'),str(file)],check=True,stdout=subprocess.DEVNULL)
            with Image.open(file) as photo:
                photo.thumbnail((1850,1850))
                scale = photo.width/400
                crop = photo.crop(tuple(round(v*scale) for v in box))
                crop.save(IMG/name)
                if number == 2490:
                    photo.crop(tuple(round(v*scale) for v in (30,97,211,144))).save(IMG/'free.png')
                if number not in REGIONS:
                    continue
                x1,y1,x2,y2,period = REGIONS[number]
                rgb = np.asarray(photo).astype(float)
                r,g,b = rgb.transpose(2,0,1)
                yellow = (r>105)&(g>95)&(np.minimum(r,g)-b>28)
                xs = np.arange(round(x1*scale),round(x2*scale))
                lo,hi = round(y1*scale),round(y2*scale)
                ys = np.array([np.median(z)+lo if len(z:=np.flatnonzero(yellow[lo:hi,x])) else np.nan for x in xs])
                valid = np.isfinite(ys)
                smooth = median_filter(np.interp(xs,xs[valid],ys[valid]),size=3)
                positive = find_peaks(-smooth,distance=.73*period*scale,prominence=2*scale)[0]
                for a,b in zip(positive,positive[1:]):
                    if not .8 < (b-a)/(scale*period) < 1.2:
                        continue
                    m = a+np.argmax(smooth[a:b])
                    if not all(valid[[a,m,b]]) or min(m-a,b-m)<.15*(b-a):
                        continue
                    records.append((f'IMG_{number}.HEIC',round(xs[a]/scale,2),round(smooth[a]/scale,2),
                                    round(xs[m]/scale,2),round(smooth[m]/scale,2),round(xs[b]/scale,2),
                                    round((smooth[m]-(smooth[a]+smooth[b])/2)/(2*scale),2)))
    with (RAW/'image_peaks.csv').open('w',newline='',encoding='utf-8') as f:
        f.write('# Начало координат слева сверху; ширина кадра 400 px. Полуразмах по минимуму между двумя максимумами.\n')
        writer=csv.writer(f)
        writer.writerow(['source','x_peak_px','y_peak_px','x_trough_px','y_trough_px','x_next_peak_px','half_range_px'])
        writer.writerows(sorted(records))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
