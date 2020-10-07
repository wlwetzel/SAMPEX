import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from scipy import signal
sys.path.append('/home/wyatt/pyModules')
sys.path.append('/home/wyatt/Documents/SAMPEX')
import SAMP_Data
import SAMPEXreb
#import Wavelet
import datetime
import matplotlib.animation as animation
from Wavelet import wavelet,wave_signif,invertWave
workDir = '/home/wyatt/Documents/SAMPEX/'
filename = "/home/wyatt/Documents/SAMPEX/data/HILThires/"

start  = pd.Timestamp('1997-10-01 10:25:00' , tz = 'utc')
end  = pd.Timestamp('1997-10-01 10:45:00' , tz = 'utc')

filename = filename + 'State4/hhrr1997' + str(start.dayofyear)+'.txt'
data = SAMPEXreb.quick_read(filename,start,end)
wavedat =  data['Rate1'].to_numpy()
times = data.index.values


#bigFile = '/home/wyatt/Documents/SAMPEX/OrbitData/PSSet_6sec_1997010_1997036.txt'
# start_time = pd.to_datetime('1997 035',utc=True,format = '%Y %j') + pd.to_timedelta(0 , unit='s')
# end_time = pd.to_datetime('1997 035',utc=True,format = '%Y %j') + pd.to_timedelta(2000 , unit='s')
dataObj = SAMP_Data.OrbitData(date=start)
df = dataObj.read_time_range(start,end,parameters=None)
#print(df)
