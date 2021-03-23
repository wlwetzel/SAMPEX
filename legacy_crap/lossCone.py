import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from scipy import signal
sys.path.append('/home/wyatt/pyModules')
import SAMP_Data
import SAMPEXreb
import datetime
import matplotlib.animation as animation
from Wavelet import wavelet,wave_signif,invertWave
from scipy.integrate import simps
from pitchAngle_v3 import *
"""
for looking at what loss cone sampex was looking at for a given month
"""
"""
# TODO: Clean up this nonsense so it's less scripty and more functional cause
        its a bitch to read
"""
radToDeg = 180.0 / m.pi

def eqPitch(eq_B, local_B,angle):
    pitch =  np.arcsin( np.sqrt(np.sin(angle)**2 * eq_B / local_B))
    return pitch

month = '01'
eventList = []
hours = ['0' + str(i) if i<10 else str(i) for i in range(24)]
days = ['0' + str(i) if i<10 else str(i) for i in range(1,31)]
for day in days:
    for hour in hours:
        eventList.append([pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':00:00' , tz = 'utc'),
                          pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':29:59' , tz = 'utc')])
        eventList.append([pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':29:59' , tz = 'utc'),
                          pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':59:59' , tz = 'utc')])

outFile = '/home/wyatt/Documents/SAMPEX/generated_Data/loss_Jan93.csv'
j=1

df = pd.read_csv(outFile,header=None,index_col=0,names=['loss'])
dat = df['loss'].to_numpy()
plt.hist(dat,bins=np.linspace(0,10,200),density=True)
plt.show()
# for event in eventList:
#     print(j/48)
#     j+=1
#     start = event[0]
#     end = event[1]
#
#     workDir = '/home/wyatt/Documents/SAMPEX/'
#     filename = "/home/wyatt/Documents/SAMPEX/data/HILThires/"
#     if len(str(start.dayofyear))==1:
#         day = '00'+str(start.dayofyear)
#     elif len(str(start.dayofyear))==2:
#         day = '0'+str(start.dayofyear)
#     else:
#         day = str(start.dayofyear)
#
#     filename = filename + 'State1/hhrr1993' + day+'.txt'
#     data = SAMPEXreb.quick_read(filename,start,end)
#     data = data.drop(columns=['Rate5','Rate6'])
#     times = data.index
#
#     if not times.empty:
#         data = findPitches(times,interpolate=False)
#         length = len(data.index.values)
#
#         for i in range(length):
#             num = 4
#         #convert to pitchangle
#             eq_B = data['Equator_B_Mag'].iloc[i]
#             local_B = data['B_Mag'].iloc[i]
#             #loss cone
#             loss = data['Loss_Cone_2'].iloc[i] / radToDeg
#             loss = eqPitch(eq_B, local_B, loss)
#
#             df = pd.DataFrame(data={'loss':[loss*radToDeg]})
#             with open(outFile,'a') as f:
#                 df.to_csv(f,header=None)
