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
from more_itertools import chunked
"""
purpose is to loop over data chunks, day by day and identify bounces with
trained model in "/home/wyatt/Documents/SAMPEX/generated_Data/model.joblib"
"length of data chunks was train_size"
TODO: maybe widen section lengths
"""
def obrien(dat):
    dat = pd.DataFrame(data={"Counts":dat})
    a_500 = dat.rolling(window=25,center=True).mean()
    n_100 = dat.rolling(window=5,center=True).mean()
    burst_param = (n_100-a_500)/np.sqrt(1+a_500)
    return (burst_param>5).any()["Counts"]

train_size = 151
clf = load("/home/wyatt/Documents/SAMPEX/generated_Data/model.joblib")

#1994 data
dates = ['1994'+str(i) for i in range(137,238)]
path = "/home/wyatt/Documents/SAMPEX/generated_Data/predictions_94.csv"
for date in dates:
    print(date)
    try:
        obj  = HiltData(date=date)
        data = obj.read(None,None)
    except:
        print("No data available for this day")
        continue
    #loop over train_size length sections
    data_length = len(data.index)
    num_chunks = data_length/train_size
    # data["Counts"] = data["Counts"]/data["Counts"].max()
    data_max = data["Counts"].max()
    prediction_list = []
    df_dict = {n: data.iloc[n:n+train_size,:] for n in range(0,len(data.index),train_size)}

    for key,chunk in df_dict.items():
        time_series = (chunk["Counts"] / data_max).to_numpy().flatten()
        if obrien(chunk["Counts"].to_numpy().flatten()):
            print("Found a microburst")
            if len(time_series)==train_size:
                prediction = clf.predict([time_series])[0]
                prediction_list.append(prediction)
            else:
                prediction_list.append(0)
                print("Incorrect array size")
        else:
            prediction_list.append(0)
    write_df = pd.DataFrame(data = {"Date":date , "Predictions":[prediction_list]})
    write_df.to_csv(path,mode='a',header=False)
