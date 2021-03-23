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

"""
1994 bouncing microbursts
"""
year = "1994"
days =       [138,138,153,153,153,153,150,150,150,150,150,150,150,227,227,151,226,226,145,161,161,158,158,158]
start_hrs =  [21,21,10, 7, 7, 5,22,22,22,22,22,22, 7,13, 0,13,14,14, 7,11,11,12,12,12]
start_mins = [58,58,31,15,15,37,17,17,16,16,16,16, 2,23,33,27,49,49,34,48,48,57,56,56]
start_secs = [25,15, 5,52,40, 5, 5, 0,55,48,46,43, 0, 5,55, 6,25,40,50,43,46, 5,47,55]
end_hrs =    [21,21,10, 7, 7, 5,22,22,22,22,22,22, 7,13, 0,13,14,14, 7,11,11,12,12,12]
end_mins =   [58,58,31,15,15,37,17,17,17,16,16,16, 2,23,34,27,49,49,34,48,48,57,56,56]
end_secs =   [35,20,10,58,46,10,10, 5, 0,51,48,46, 7, 8, 0,10,27,45,55,46,50,10,50,58]

days = [str(day) for day in days]
start_hrs = [str(hr) if len(str(hr))==2 else "0"+str(hr) for hr in start_hrs ]
start_mins = [str(item) if len(str(item))==2 else "0"+str(item) for item in start_mins ]
start_secs = [str(item) if len(str(item))==2 else "0"+str(item) for item in start_secs ]
end_hrs = [str(item) if len(str(item))==2 else "0"+str(item) for item in end_hrs ]
end_mins = [str(item) if len(str(item))==2 else "0"+str(item) for item in end_mins ]
end_secs = [str(item) if len(str(item))==2 else "0"+str(item) for item in end_secs ]

peaks_94 = [ [140,168,199,226],[122,149,183],[152,179,215],[117,154,192],[169,207,239,279],
          [78,113,148],[65,90,116,144],[62,90,116],[153,179,206],[38,52,65,81,94,107,119,137],
          [38,56,68],[35,68,92],[199,254,300],[40,68,103],[18,56,91,129,166,205],[50,86,146],
          [6,43,74],[54,77,114,149],[97,143,190],[55,85],[37,77,108,140],[58,93,125],[34,68,102],
          [7,44,75]
        ]

starts_94 = [pd.to_datetime(year+days[i]+start_hrs[i]+start_mins[i]+start_secs[i],format="%Y%j%H%M%S",utc=True) for i in range(len(days))]
ends_94   = [pd.to_datetime(year+days[i]+end_hrs[i]+end_mins[i]+end_secs[i],format="%Y%j%H%M%S",utc=True) for i in range(len(days))]

"""
1993 bouncing microbursts
"""

def create_smaller_file():
    """
    makes a smaller file so that accessing the bouncing microbursts is quicker,
    just saves their time series in a csv
    really should have been done in graphical_bounce_search.py, move this there
    if I need to run again/generalize to arb. years
    """
    file_93 = '/home/wyatt/Documents/SAMPEX/generated_Data/human_filtered_bounces93.csv'
    df = pd.read_csv(file_93,names= ['group','date'],header=0)
    df['group'] = df['group'].apply(literal_eval)
    save_path = '/home/wyatt/Documents/SAMPEX/generated_Data/93bounces_quicker_read.csv'
    for i in range(len(df.index)):
        obj= HiltData(date=str(df['date'].iloc[i]))
        data = obj.read(None,None)
        group = df['group'].iloc[i]
        #how much to pad the data - lets do 3s on each side
        ind1 = group[0] - 30
        ind2 = group[-1] + 30
        start = data.index[ind1]
        end = data.index[ind2]
        burst_num = [i]*len(data.index[ind1:ind2])
        df_to_save = pd.DataFrame(data={"Burst":burst_num,"Time":data.index[ind1:ind2],"Counts":data['Rate1'].iloc[ind1:ind2]})
        df_to_save = df_to_save.set_index(['Burst','Time'])
        df_to_save.to_csv(save_path,mode='a')
        print(i)

data_path = '/home/wyatt/Documents/SAMPEX/generated_Data/93bounces_quicker_read.csv'
"""
now that these are organized in a smaller file, we can graphically choose each
peak
"""
bounces = pd.read_csv(data_path)
bounces = bounces.set_index(['Burst','Time'])
indices = [str(i) for i in range(94)]
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

    curr_data['Counts'] = curr_data['Counts'].apply(literal_eval)
    ax.plot(curr_data.index,curr_data['Counts'],'-ro',picker=10)
    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.show()
    peak_times_master.append(peak_times)

print(peak_times_master)
df = pd.DataFrame(data = {"Peaks":peak_times_master})
save_file = '/home/wyatt/Documents/SAMPEX/generated_Data/peaks_93.csv'
df.to_csv(save_file)
