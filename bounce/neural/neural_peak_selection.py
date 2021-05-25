import pandas as pd
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
from SAMP_Data import *
import plotly.express as px
from itertools import groupby
import itertools
import plotly.graph_objects as go
from ast import literal_eval
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

bounces = pd.read_csv("/home/wyatt/Documents/SAMPEX/generated_Data/accepted_predictions_94.csv",
                          header=None,names=["Burst","Time","Counts"],usecols=[0,1,2])
indices = bounces['Burst'].drop_duplicates().to_numpy()
bounces = bounces.set_index(['Burst','Time'])

def on_pick(event):
    artist = event.artist
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    x, y = artist.get_xdata(), artist.get_ydata()
    ind = event.ind
    peak_times.append(x[ind[0]])
    print(x[ind[0]])
peak_times_master = []
for index in indices:
    peak_times = []
    curr_data = bounces.loc[index]
    fig, ax = plt.subplots()

    ax.plot(curr_data.index,curr_data['Counts'],'-ro',picker=10)
    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.show()

    peak_times_master.append(peak_times)

print(peak_times_master)
df = pd.DataFrame(data = {"Peaks":peak_times_master,"Burst":indices})
save_file = '/home/wyatt/Documents/SAMPEX/generated_Data/neural_peaks_94.csv'
df.to_csv(save_file)
