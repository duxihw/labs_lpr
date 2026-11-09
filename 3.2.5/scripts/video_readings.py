from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist


def select_video_frames(samples, folder):
    """Use the highlighted 10 Hz digit; use the blue period to resolve decimal position."""
    frames=samples.frame.to_numpy(dtype=int)
    # Checked major readings on the generator; frame indices are zero-based.
    anchor={5:5500,99:5700,162:5900,192:6000,222:6100,252:6200,282:6300,
            315:6400,345:6500,375:6600,405:6700,435:6800,465:6900,
            495:7000,567:7200,600:7300,636:7400,685:7500,800:7500}
    scope=median_filter(samples.scope_Hz.to_numpy(),3)
    ids=np.searchsorted(frames,list(anchor))
    correction=np.array(list(anchor.values()))-scope[ids]*1.05
    expected=scope*1.05+np.interp(frames,list(anchor),correction)
    shapes=[]
    for frame in frames:
        patch=cv2.imread(str(folder/f'g{frame:04d}.png'),0)
        # Locate the selected character by the top edge of its bright cursor.
        up=cv2.resize(patch,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        search=up[20:60,121:170].astype(float)
        _,_,_,(x,y)=cv2.minMaxLoc(cv2.boxFilter(np.diff(search,axis=0),-1,(14,3)))
        cx=x+121; top=y+20
        glyph=up[top+5:top+39,cx-10:cx+11].astype(float)
        glyph=cv2.resize(glyph,(7,11))
        shapes.append(((glyph-glyph.mean())/max(glyph.std(),1)).ravel())
    shapes=np.array(shapes)
    freq=np.arange(5500,7501,10)
    prior=(expected[:,None]-freq)/10
    state=np.argmin(abs(prior),axis=1)
    # Learn the ten glyph appearances from repeated cycles through the dial.
    # Monotone path permits omitted settings when knob motion is faster than the camera.
    for _ in range(3):
        prototypes=np.array([np.median(shapes[state%10==digit],axis=0) for digit in range(10)])
        distance=cdist(shapes,prototypes,'sqeuclidean')/shapes.shape[1]
        cost=.6*prior**2+distance[:,np.arange(len(freq))%10]*2
        for t,value in zip(ids,anchor.values()):
            cost[t,freq!=value]=np.inf
        prev=np.full(len(freq),np.inf);prev[0]=0
        back=np.zeros(cost.shape,dtype=int)
        for t,row in enumerate(cost):
            candidates=np.array([np.r_[np.full(step,np.inf),prev[:len(freq)-step]] for step in range(5)])
            steps=candidates.argmin(axis=0)
            back[t]=np.arange(len(freq))-steps
            prev=candidates.min(axis=0)+row
        state[-1]=prev.argmin()
        for t in range(len(frames)-1,0,-1):
            state[t-1]=back[t,state[t]]
    records=[]
    for k in np.unique(state):
        loc=np.flatnonzero(state==k)
        # Retain one central observation after both instruments have refreshed.
        j=loc[len(loc)//2]
        r=samples.iloc[j]
        if abs(prior[j,k])>2 or abs(r.ratio-1)>.02:
            continue
        records.append((410,int(freq[k]),round(r.Upp_V,3),round(r.delay_cycles*1e6/freq[k],3),
                        int(r.frame),round(r.time,4),len(loc)))
    return pd.DataFrame(records,columns=['R_ohm','f_Hz','Upp_V','delay_us','frame','t_s','frame_count'])
