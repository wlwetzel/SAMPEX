import pandas as pd
from ast import literal_eval
import plotly.express as px
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
from SAMP_Data import *
import plotly.graph_objects as go

year = "1994"
days = [138,138,153,153,153,153,150,150,150,150,150,150,150,227,227,151,226,226,145,161,161,158,158,158]
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

starts = [pd.to_datetime(year+days[i]+start_hrs[i]+start_mins[i]+start_secs[i],format="%Y%j%H%M%S") for i in range(len(days))]
ends = [pd.to_datetime(year+days[i]+end_hrs[i]+end_mins[i]+end_secs[i],format="%Y%j%H%M%S") for i in range(len(days))]
counter = 0

for day in days:
    obj = HiltData(date=str(year)+str(day))
    data = obj.read(None,None)
    data = data[starts[counter]:ends[counter]]
    peaks,_ = signal.find_peaks()
    fig = px.line(data)
    fig.show()
    counter+=1
