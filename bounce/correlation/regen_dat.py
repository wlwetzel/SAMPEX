import numpy as np
import plotly.express as px
import pandas as pd
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
import SAMP_Data as sp
import plotly.graph_objects as go
from scipy import signal
from spacepy.time import Ticktock
from ast import literal_eval
from joblib import dump, load
from more_itertools import chunked
import tkinter
import spacepy.coordinates as spc
import spacepy.irbempy as irb

#94 and 96 were altered by my nonsense

peaks_94 = pd.read_csv("/home/wyatt/Documents/SAMPEX/bounce/correlation/data/peaks_1994.csv")
peaks_94['Peaks'] = peaks_94['Peaks'].apply(literal_eval)
peaks_94 = peaks_94.set_index('Burst')
indices_94 = peaks_94.index
peaks_96 = pd.read_csv("/home/wyatt/Documents/SAMPEX/bounce/correlation/data/peaks_1996.csv")
peaks_96['Peaks'] = peaks_96['Peaks'].apply(literal_eval)
peaks_96 = peaks_96.set_index('Burst')
indices_96 = peaks_96.index
time = 0
for ind in indices_94:
    peaks = peaks_94["Peaks"].loc[ind]
    if not peaks:
        time  = time+pd.Timedelta('10s')
        #these were rejected on peak select, so pick a random time to fill in data
    else:
        time = pd.Timestamp(peaks[0])

    obj  = sp.HiltData(date=str(time.year) + str(time.dayofyear))
    data = obj.read(None,None)
    dat_to_write = data.loc[time-pd.Timedelta('2s'):time+pd.Timedelta('2s'),:]
    dat_to_write = dat_to_write.reset_index()
    dat_to_write["bounce"] = ind
    dat_to_write = dat_to_write[ ['bounce'] + [ col for col in dat_to_write.columns if col != 'bounce' ] ]
    dat_to_write.to_csv("/home/wyatt/Documents/SAMPEX/bounce/correlation/data/accepted_1994.csv",
                        index=False,header=None,mode='a')
for ind in indices_96:
    peaks = peaks_96["Peaks"].loc[ind]
    if not peaks:
        time  = time+pd.Timedelta('10s')
        #these were rejected on peak select, so pick a random time to fill in data
    else:
        time = pd.Timestamp(peaks[0])
    obj  = sp.HiltData(date=str(time.year) + str(time.dayofyear))
    data = obj.read(None,None)
    dat_to_write = data.loc[time-pd.Timedelta('2s'):time+pd.Timedelta('2s'),:]
    dat_to_write = dat_to_write.reset_index()
    dat_to_write["bounce"] = ind
    dat_to_write = dat_to_write[ ['bounce'] + [ col for col in dat_to_write.columns if col != 'bounce' ] ]
    dat_to_write.to_csv("/home/wyatt/Documents/SAMPEX/bounce/correlation/data/accepted_1996.csv",
                        index=False,header=None,mode='a')
