import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
import SAMP_Data as sp
import pandas as pd
import plotly.express as px
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np

"""
compare the L histograms of all microbursts and just the flagged bouncing
microbursts
"""
bounce_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/stats.csv"
bounce_stats = pd.read_csv(bounce_file,names = ["time_diff","percent_diff","periods","hemisphere","L","MLT","lat","lon","vx","vy","vz","timestamp"]
                 ,usecols=[1,2,3,4,5,6,7,8,9,10,11,12])
micro_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/obrien_stats.csv"
micro_stats = pd.read_csv(micro_file,header=None,names=["L","MLT"])

bins = [0,1,2,3,4,5,6,7,8]

bounce_hist ,_ = np.histogram(bounce_stats["L"].to_numpy(),bins)
micro_hist ,_ = np.histogram(micro_stats["L"].to_numpy(),bins)

comp = bounce_hist / micro_hist * 100
fig = go.Figure(
    data = go.Bar(
        x=[bin-.5 for bin in bins[1:]],y=comp,
        width=1
    ),
    layout_xaxis_title_text = "L Shell",
    layout_yaxis_title_text ="Percent",
    layout_title_text="# Bouncing Microbursts / # Microbursts"
)
fig.write_image("/home/wyatt/Documents/SAMPEX/bounce_figures/bounce_div_micro.png")
fig.show()
