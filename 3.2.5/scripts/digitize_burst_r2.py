"""Снять центры вершин жёлтой трассы на кадрированной IMG_2502."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image

lab = Path(__file__).resolve().parents[1]
r,g,b = np.asarray(Image.open(lab/'report/img/burst_fast.jpg'), dtype=float).transpose(2,0,1)
yellow = (r>170) & (g>150) & (np.minimum(r,g)-b>45)
rows = []
for mode, centers in [('rise',[79,147,224]),('plateau',[463,543,622,702,781,860,939,1017,1096,1175]),
                      ('fall',[1278,1360]),('zero',[49,1480])]:
    for center in centers:
        # В узком окне вокруг вершины находим середину светящейся линии.
        coords = []
        for x in range(center-8,center+9):
            y = np.flatnonzero(yellow[255:690,x])+255
            if len(y):
                coords.append((x,np.median(y)))
        pts = np.array(coords)
        level = np.median(pts[:,1]) if mode=='zero' else (max if mode=='fall' else min)(pts[:,1])
        vertex = pts[np.abs(pts[:,1]-level)<=2]
        rows.append((mode,*np.median(vertex,axis=0)))
with (lab/'data/raw/IMG_2502_extrema.csv').open('w',newline='',encoding='utf-8') as f:
    f.write('# Координаты в burst_fast.jpg, начало слева сверху. Рост: верхние пики; спад: нижние пики.\n')
    w=csv.writer(f)
    w.writerow(['part','x_px','y_px'])
    w.writerows(rows)
