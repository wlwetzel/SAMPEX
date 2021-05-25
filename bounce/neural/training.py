from sklearn.neural_network import MLPClassifier
import numpy as np
import plotly.express as px
import pandas as pd
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
from SAMP_Data import *
import plotly.graph_objects as go
from scipy import signal
from spacepy.time import Ticktock
from ast import literal_eval
from joblib import dump, load

"""
training the neural network on a bunch of sample data. We'll start with 1994

"""

Re = 6371 #km
year = "1994"
days =       [138,138,153,153,153,153,150,150,150,150,150,150,150,227,227,151,226,226,145,161,161,158,158,158]
start_hrs =  [21,21,10, 7, 7, 5,22,22,22,22,22,22, 7,13, 0,13,14,14, 7,11,11,12,12,12]
start_mins = [58,58,31,15,15,37,17,17,16,16,16,16, 2,23,33,27,49,49,34,48,48,57,56,56]
start_secs = [25,15, 5,52,40, 5, 5, 0,55,48,46,43, 0, 5,55, 6,25,40,50,43,46, 5,47,55]
end_hrs =    [21,21,10, 7, 7, 5,22,22,22,22,22,22, 7,13, 0,13,14,14, 7,11,11,12,12,12]
end_mins =   [58,58,31,15,15,37,17,17,17,16,16,16, 2,23,34,27,49,49,34,48,48,57,56,56]
end_secs =   [35,20,10,58,46,10,10, 5, 0,51,50,46, 7, 8, 0,10,28,45,55,46,50,10,50,58]

days = [str(day) for day in days]
start_hrs = [str(hr) if len(str(hr))==2 else "0"+str(hr) for hr in start_hrs ]
start_mins = [str(item) if len(str(item))==2 else "0"+str(item) for item in start_mins ]
start_secs = [str(item) if len(str(item))==2 else "0"+str(item) for item in start_secs ]
end_hrs = [str(item) if len(str(item))==2 else "0"+str(item) for item in end_hrs ]
end_mins = [str(item) if len(str(item))==2 else "0"+str(item) for item in end_mins ]
end_secs = [str(item) if len(str(item))==2 else "0"+str(item) for item in end_secs ]

peaks = [ [140,168,199,226],[122,149,183],[152,179,215],[117,154,192],[169,207,239,279],
          [78,113,148],[65,90,116,144],[62,90,116],[153,179,206],[38,52,65,81,94,107,119,137],
          [38,56,68],[35,68,92],[199,254,300],[40,68,103],[18,56,91,129,166,205],[50,86,146],
          [6,43,74],[54,77,114,149],[97,143,190],[55,85],[37,77,108,140],[58,93,125],[34,68,102],
          [7,44,75]
        ]
lengths = [len(item) for item in peaks]
starts = [pd.to_datetime(year+days[i]+start_hrs[i]+start_mins[i]+start_secs[i],format="%Y%j%H%M%S",utc=True) for i in range(len(days))]
ends = [pd.to_datetime(year+days[i]+end_hrs[i]+end_mins[i]+end_secs[i],format="%Y%j%H%M%S",utc=True) for i in range(len(days))]

non_bounce_starts = [start + pd.Timedelta('15s') for start in starts]
non_bounce_ends = [end +pd.Timedelta('15s') for end in ends]
#some of these training spots actually have bounces in them whoops so I'm
#manually choosing non bounce sections
non_bounce_starts[4] = non_bounce_starts[4]+pd.Timedelta('65s')
non_bounce_ends[4] = non_bounce_ends[4]+pd.Timedelta('65s')
non_bounce_starts[10] = non_bounce_starts[10]+pd.Timedelta('8s')
non_bounce_ends[10] = non_bounce_ends[10]+pd.Timedelta('8s')
non_bounce_starts[11] = non_bounce_starts[11]+pd.Timedelta('35s')
non_bounce_ends[11] = non_bounce_ends[11]+pd.Timedelta('35s')
non_bounce_starts[15] = non_bounce_starts[15]+pd.Timedelta('8s')
non_bounce_ends[15] = non_bounce_ends[15]+pd.Timedelta('8s')
non_bounce_starts[16] = non_bounce_starts[16]+pd.Timedelta('8s')
non_bounce_ends[16] = non_bounce_ends[16]+pd.Timedelta('8s')
non_bounce_starts[18] = non_bounce_starts[18]+pd.Timedelta('8s')
non_bounce_ends[18] = non_bounce_ends[18]+pd.Timedelta('8s')
non_bounce_starts[21] = non_bounce_starts[21]+pd.Timedelta('9s')
non_bounce_ends[21] = non_bounce_ends[21]+pd.Timedelta('9s')
non_bounce_starts[22] = non_bounce_starts[22]+pd.Timedelta('9s')
non_bounce_ends[22] = non_bounce_ends[22]+pd.Timedelta('9s')


make_file = 1
if make_file:
    counter=0
    master_data_list = []
    minimum = 100000
    for day in days:
        obj  = HiltData(date=str(year)+str(day))
        data = obj.read(None,None)
        data = data[starts[counter]:ends[counter]]
        print(len(data))
        data = data.to_numpy().flatten()
        if len(data)<minimum:
            minimum = len(data)
        master_data_list.append(data)
        counter+=1
    #we neeed to add some non-bounces to the training data
    counter=0
    for day in days:
        obj  = HiltData(date=str(year)+str(day))
        data = obj.read(None,None)
        data = data[non_bounce_starts[counter]:non_bounce_ends[counter]]
        data = data.to_numpy().flatten()
        if len(data)<minimum:
            minimum = len(data)
        master_data_list.append(data)
        counter+=1

    data_list_to_write = []
    for item in master_data_list:
        data_list_to_write.append(item[0:minimum].tolist())

    path = "/home/wyatt/Documents/SAMPEX/generated_Data/94_training_data.csv"
    print(data_list_to_write)
    df = pd.DataFrame(data = {"lengths":lengths + [0]*len(lengths),"data":data_list_to_write})
    print(df)
    df.to_csv(path,quotechar='"',encoding='ascii')

path = "/home/wyatt/Documents/SAMPEX/generated_Data/94_training_data.csv"
bounce94_df = pd.read_csv(path,converters={"data":literal_eval})
data =np.vstack([np.array(item) for item in  bounce94_df["data"].to_numpy()]) #had to reformat the damn thing
data = data / np.max(data)
lengths = lengths + [0]*len(lengths)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,random_state=1)
clf.fit(data, lengths)
dump(clf,"/home/wyatt/Documents/SAMPEX/generated_Data/model.joblib")
clf.predict([[0]])
