from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.optimize import least_squares
from video_readings import select_video_frames
import pandas as pd

VIDEO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(exist_ok=True)
cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS)
ok,ref = cap.read()
gray = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
grid = np.float32([[1045,291],[1647,340],[1590,723],[1016,653]])
screen = np.float32([[34,578],[357,540],[396,739],[81,793]])
def tracker(boxes):
    mask=np.zeros_like(gray)
    for x,y,w,h in boxes:
        mask[y:y+h,x:x+w]=255
    pts=cv2.goodFeaturesToTrack(gray,350,.008,6,mask=mask)
    return pts,pts.copy()
scope=tracker([(978,197,730,86),(1635,320,260,500),(998,721,550,85)])
gen=tracker([(368,440,320,330),(14,471,355,92),(65,780,400,70)])
def track(now,state):
    base,prev=state
    pts,status,_=cv2.calcOpticalFlowPyrLK(gray,now,base,prev.copy(),winSize=(29,29),maxLevel=3,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    ok=status.ravel().astype(bool)
    H,good=cv2.findHomography(base[ok],pts[ok],cv2.RANSAC,2.2)
    prev[:]=cv2.perspectiveTransform(base,H)
    return H

def warp(im,H,box,w,h):
    src=cv2.perspectiveTransform(box[None],H)[0]
    M=cv2.getPerspectiveTransform(src,np.float32([[0,0],[w,0],[w,h],[0,h]]))
    return cv2.warpPerspective(im,M,(w,h))

def fit_wave(im,blue,cycles):
    b,g,r=im.astype(float).transpose(2,0,1)
    mask=((b-r>9)&(b>150)&(g>145)) if blue else ((r-b>12)&(g-b>13)&(g>153))
    yy,xx=np.indices(mask.shape)
    mask &= (xx>14)&(xx<630)&(yy>7)&(yy<390)&~((xx<158)&(yy>260))
    # Exclude horizontal/vertical cursor lines and border lettering.
    mask[np.sum(mask,axis=1)>220,:]=False
    mask[:,np.sum(mask,axis=0)>55]=False
    y,x=np.where(mask)
    # Fit to the bright trace, rejecting mismatched cursor/menu pixels.
    if len(x)<80: raise ValueError('few trace pixels')
    x=x/640
    def coeff(v):
        a=np.column_stack((np.ones(len(x)),np.cos(2*np.pi*v*x),np.sin(2*np.pi*v*x)))
        c=np.linalg.lstsq(a,y,rcond=None)[0]
        return c,np.median(abs(a@c-y))
    guesses=np.arange(max(1.45,cycles-.10),min(2.65,cycles+.10)+.01,.02)
    v=min(guesses,key=lambda v:coeff(v)[1])
    c,_=coeff(v)
    fit=least_squares(lambda p: p[0]+p[1]*np.cos(2*np.pi*p[3]*x)+p[2]*np.sin(2*np.pi*p[3]*x)-y,
                      [*c,v],loss='soft_l1',f_scale=2,bounds=([-100,-300,-300,1.4],[500,300,300,2.7]))
    center,cc,ss,v=fit.x
    return v,np.hypot(cc,ss),np.arctan2(-ss,-cc),np.median(abs(fit.fun))
rows=[]; index=0; cycles=1.76
while ok:
    now=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
    H=track(now,scope); G=track(now,gen)
    graph=warp(ref,H,grid,640,400); dial=warp(ref,G,screen,360,210)
    cv2.imwrite(str(OUT/f'g{index:04d}.png'),dial[26:65,122:230])
    try:
        vb,ab,pb,eb=fit_wave(graph,True,cycles)
        vy,ay,py,ey=fit_wave(graph,False,vb)
        cycles=vb
        rows.append([index,index/fps,vb/0.000320,ay*4/40,((py-pb+np.pi)%(2*np.pi)-np.pi)/(2*np.pi),eb,ey,vy/vb])
    except ValueError:
        pass
    index+=1;ok,ref=cap.read()
samples = pd.DataFrame(rows,columns=['frame','time','scope_Hz','Upp_V','delay_cycles','blue_err','yellow_err','ratio'])
samples.to_csv(OUT/'samples.csv',index=False)
readings = select_video_frames(samples, OUT)
raw = Path(__file__).resolve().parents[1]/'data/raw/video_resonance.csv'
readings.to_csv(raw,index=False)
# Сохранить кадры с номером и настройкой в рабочем каталоге пользователя.
keep = dict(zip(readings.frame, readings.f_Hz))
cap.release()
cap = cv2.VideoCapture(str(VIDEO))
for i in range(index):
    ok, im = cap.read()
    if not ok:
        break
    if i in keep:
        cv2.imwrite(str(OUT/f'{i:04d}_{keep[i]}Hz.jpg'), im)
cap.release()
