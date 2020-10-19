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
how do we background subtract?
"""
date = '1992278'
obj = HiltData(date=date)
data = obj.read(14000,14800)
data = data['Rate1']
# roll = data.rolling(100,center=True,win_type='triang').mean()
# fig = px.line(data)
# trace = go.Scatter(x=roll.index,y=roll.to_numpy())
# fig.add_trace(trace)
# fig.show()
"""
rolling average isnt great
try highpass filter
"""
