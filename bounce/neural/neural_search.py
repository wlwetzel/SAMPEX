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
import os
import copy

"""
purpose is to loop over data chunks, day by day and identify bounces with
trained model in "/home/wyatt/Documents/SAMPEX/generated_Data/model.joblib"
"length of data chunks was train_size"

TODO: -maybe widen section lengths DONE
      -save data when found instead of separately DONE
"""

def obrien(dat):
    dat = pd.DataFrame(data={"Counts":dat})
    a_500 = dat.rolling(window=25,center=True).mean()
    n_100 = dat.rolling(window=5,center=True).mean()
    burst_param = (n_100-a_500)/np.sqrt(1+a_500)
    return (burst_param>5).any()["Counts"]

def neural_search(dates):
    """
    dates: list of strings, format YYYYDDD
    only intended for use in one year at a time
    """
    path = "/home/wyatt/Documents/SAMPEX/generated_Data/predictions_"+dates[0][2:4] +".csv"
    train_size = 201
    clf = load("/home/wyatt/Documents/SAMPEX/generated_Data/model.joblib")
    try:
        os.remove(path)
    except:
        pass
    date_write_list=[]
    counter = 0
    for date in dates:
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
        df_dict = {n: data.iloc[n:n+train_size,:] for n in range(0,len(data.index),train_size)}

        for key,chunk in df_dict.items():
            time_series = (chunk["Counts"] / data_max).to_numpy().flatten()
            if obrien(chunk["Counts"].to_numpy().flatten()):
                if len(time_series)==train_size:
                    prediction = clf.predict([time_series])[0]
                    #set up to return 1 if net thinks there is a bounce
                    if prediction:
                        os.system("clear")
                        print(date)
                        print(f"Found {counter} bounces")
                        write_df = copy.deepcopy(chunk)
                        write_df["Burst"] = counter
                        write_df.to_csv(path,mode='a',header=False)
                        counter+=1
                else:
                    print("Incorrect array size")
            else:
                pass

#96 dates
days = ['00' + str(i) for i in range(1,10)] + ['0' + str(i) for i in range(10,100)] + [str(i) for i in range(100,366)]
dates = ["1996" + day for day in days]
neural_search(dates)
