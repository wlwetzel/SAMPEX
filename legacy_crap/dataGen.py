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
from pitchAngle_v2 import *
#%matplotlib inline
#%%
"""
change this section to generate different months of data
"""
month = '02'
histFile = 'hist_Feb93.csv'
lookFile = 'look_Feb93.csv'

eventList = []
hours = ['0' + str(i) if i<10 else str(i) for i in range(24)]
days = ['0' + str(i) if i<10 else str(i) for i in range(1,29)]
for day in days:
    for hour in hours:
        eventList.append([pd.Timestamp('1993-'+month+'-'+day+ ' ' +hour+ ':00:00' , tz = 'utc'),
                          pd.Timestamp('1993-'+month+'-'+day+ ' ' +hour+ ':29:59' , tz = 'utc')])
        eventList.append([pd.Timestamp('1993-'+month+'-'+day+ ' ' +hour+ ':29:59' , tz = 'utc'),
                          pd.Timestamp('1993-'+month+'-'+day+ ' ' +hour+ ':59:59' , tz = 'utc')])
#%%

i=1

with open('/home/wyatt/Documents/SAMPEX/'+histFile,'a') as file:
    for event in eventList:
        print(i/48.0)
        i+=1
        start = event[0]
        end = event[1]
        data,baseDF , peakDF,pitchAtPeaks = peakFind(start, end)
        weights,bins = binning(peakDF, pitchAtPeaks)
        histDF = pd.DataFrame(data = {'data':bins,'weights':weights})
        histDF.to_csv(file,header=False)

#%%
columnList = [['det_1_Alpha'  ,'det_1_Beta'],
              [ 'det_2_Alpha' ,'det_2_Beta'] ,
              ['det_3_Alpha' ,'det_3_Beta'],
               ['det_4_Alpha', 'det_4_Beta'] ]
j = 1

with open('/home/wyatt/Documents/SAMPEX/' + lookFile,'a') as file:
    for event in eventList:
        binList = np.array([])
        print(j/48.0)
        j+=1
        times = readInTimes(event[0], event[1])
        pitchInfo = findPitches(times,interpolate=False)
        if not pitchInfo.empty:
            for i in range(len(pitchInfo.index)):
                for col in columnList:
                    #linspace from alpha to beta for each detector
                    alpha=pitchInfo.iloc[i].loc[col[0]]
                    beta=pitchInfo.iloc[i].loc[col[1]]
                    bins,_ = getBins(alpha, beta)
                    binList = np.append(binList,bins)

        binList = binList.flatten()
        histDF = pd.DataFrame(data = {'data':binList})
        histDF.to_csv(file,header=False)
#%%
