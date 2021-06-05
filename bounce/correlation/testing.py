import pandas as pd
from ast import literal_eval
import plotly.express as px
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
import SAMP_Data as sp
from scipy import signal

# read = pd.read_csv("/home/wyatt/Documents/SAMPEX/generated_Data/94_training_data_bounces.csv",converters={"data":literal_eval},usecols=[1])

year = "1994"
days =       [138,138,153,153,153,153,150,150,150,150,150,150,150,227,227,151,226,226,161,158,158,158]
start_hrs =  [21,21,10, 7, 7, 5,22,22,22,22,22,22, 7,13, 0,13,14,14,11,12,12,12]
start_mins = [58,58,31,15,15,37,17,17,16,16,16,16, 2,23,33,27,49,49,48,57,56,56]
start_secs = [27,16, 6,54,42, 6, 5, 0,56,47,45,43, 3, 5,55, 6,25,40,46, 5,47,54]
end_hrs =    [21,21,10, 7, 7, 5,22,22,22,22,22,22, 7,13, 0,13,14,14,11,12,12,12]
end_mins =   [58,58,31,15,15,37,17,17,17,16,16,16, 2,23,33,27,49,49,48,57,56,56]
end_secs =   [31,20,10,58,46,10, 9, 4, 0,51,49,47, 7, 9,59,10,29,44,50, 9,51,58]
#            [00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21]
days = [str(day) for day in days]
start_hrs = [str(hr) if len(str(hr))==2 else "0"+str(hr) for hr in start_hrs ]
start_mins = [str(item) if len(str(item))==2 else "0"+str(item) for item in start_mins ]
start_secs = [str(item) if len(str(item))==2 else "0"+str(item) for item in start_secs ]
end_hrs = [str(item) if len(str(item))==2 else "0"+str(item) for item in end_hrs ]
end_mins = [str(item) if len(str(item))==2 else "0"+str(item) for item in end_mins ]
end_secs = [str(item) if len(str(item))==2 else "0"+str(item) for item in end_secs ]

starts = [pd.to_datetime(year+days[i]+start_hrs[i]+start_mins[i]+start_secs[i],format="%Y%j%H%M%S",utc=True) for i in range(len(days))]
ends = [pd.to_datetime(year+days[i]+end_hrs[i]+end_mins[i]+end_secs[i],format="%Y%j%H%M%S",utc=True) for i in range(len(days))]

def correlate(x):
    #transform data
    subtracted = x - x.rolling(10,min_periods=1).quantile(.1)
    correlated = signal.correlate(subtracted,kernel,mode='same') / len(subtracted)
    return correlated.flatten()


# kernels = [0,3,6,9,10,11,14,19,21]
# counter=0
# kernel_df_list = []
# for id in kernels:
#     day = days[id]
#     obj  = sp.HiltData(date=str(year)+str(day))
#     data = obj.read(None,None)
#     data = data[starts[id]:ends[id]]
#     data['bounce'] = counter
#     kernel_df_list.append(data)
#     counter+=1
# kernel_df = pd.concat(kernel_df_list)
# kernel_df.to_csv("/home/wyatt/Documents/SAMPEX/bounce/correlation/data/kernels.csv")
id = 0
day = days[id]
obj  = sp.HiltData(date=str(year)+str(day))
data = obj.read(None,None)
kernel = data[starts[id]:ends[id]]
kernel = kernel - kernel.rolling(10,min_periods=1).quantile(.1)


id = -4
day = days[id]
obj  = sp.HiltData(date=str(year)+str(day))
data = obj.read(None,None)

data = data[starts[id]-pd.Timedelta("150s"):ends[id]+pd.Timedelta("150s")]
correlated = correlate(data)

data = data/data.max()
data['corr']= correlated
fig=px.line(data)
fig.show()
