import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
from SAMP_Data import *
from scipy import signal
import plotly.express as px
from itertools import groupby
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

"""
o'brien burst parameter: (n_100 - a_500)/sqrt(1+a_500)
n_100 = num in 100ms
a_500 = 500ms running average
"""
#let's load the 96 bounce packet

date = '1992278'
date = '1994138'
# date = '1993079'
obj = HiltData(date=date)
data = obj.read(70000,80000)
# data = data[['Rate1','Rate2','Rate3','Rate4']]
# x = data['Rate1']
# a_500 = x.rolling(window=5,center=True).mean()
# burst_param = (x-a_500)/np.sqrt(1+a_500)
# bursts = x[burst_param>10]
fig = px.line(data)
# fig.update_layout(yaxis_title_text="Counts",title_text = "SAMPEX Microbursts",showlegend=True)
# fig.add_scatter(x=bursts.index,y=bursts.to_numpy(),mode='markers')
fig.show()
