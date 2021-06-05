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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
"""
training the neural network on a bunch of sample data. We'll start with 1994

"""

Re = 6371 #km
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

peaks = [ [140,168,199,226],[122,149,183],[152,179,215],[117,154,192],[169,207,239,279],
          [78,113,148],[65,90,116,144],[62,90,116],[153,179,206],[38,52,65,81,94,107,119,137],
          [38,56,68],[35,68,92],[199,254,300],[40,68,103],[18,56,91,129,166,205],[50,86,146],
          [6,43,74],[54,77,114,149],[97,143,190],[55,85],[37,77,108,140],[58,93,125],[34,68,102],
          [7,44,75]
        ]
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
non_bounce_starts[20] = non_bounce_starts[20]+pd.Timedelta('8s')
non_bounce_ends[20] = non_bounce_ends[20]+pd.Timedelta('8s')


"""
write data to file for quicker access (Sampex data takes a while to load)
TODO: rewrite so that non bounces are written to a separate file
"""

bounce_path = "/home/wyatt/Documents/SAMPEX/generated_Data/accepted_predictions_94.csv"
reject_path = "/home/wyatt/Documents/SAMPEX/generated_Data/flagged_Data.csv"
train_bounce_path = "/home/wyatt/Documents/SAMPEX/generated_Data/94_training_data_bounces.csv"
train_reject_path = "/home/wyatt/Documents/SAMPEX/generated_Data/94_training_data_rejects.csv"

make_file = 1
if make_file:
    try:
        os.remove(train_bounce_path)
        os.remove(train_reject_path)
    except:
        pass
    counter=0
    master_data_list = []
    minimum = 100000
    for day in days:
        obj  = HiltData(date=str(year)+str(day))
        data = obj.read(None,None)
        data = data[starts[counter]:ends[counter]]
        data = data.to_numpy().flatten()
        if len(data)<minimum:
            minimum = len(data)

        master_data_list.append(data)
        counter+=1

    data_list_to_write = []
    for item in master_data_list:
        data_list_to_write.append(item[0:minimum].tolist())

    try:
        os.remove(train_bounce_path)
    except:
        pass
        df = pd.DataFrame(data = {"data":data_list_to_write})
        df.to_csv(train_bounce_path,quotechar='"',encoding='ascii')

    #we neeed to add some non-bounces to the training data
    counter=0
    master_data_list = []
    minimum = 100000
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

    try:
        os.remove(train_reject_path)
    except:
        pass
        df = pd.DataFrame(data = {"data":data_list_to_write})
        df.to_csv(train_reject_path,quotechar='"',encoding='ascii')
    """
    Adding more data from 1994 to training data-especially non bouncing microbursts
    that trigger the obrien parameter
    """
    bounce_df = pd.read_csv(bounce_path,names=["bounce","time","counts"])
    reject_df = pd.read_csv(reject_path,names=["bounce","time","counts"])

    bounce_df = bounce_df.set_index(["bounce"])
    reject_df = reject_df.set_index(["bounce"])

    bounce_indices = bounce_df.index.drop_duplicates()
    reject_indices = reject_df.index.drop_duplicates()

    for ind in bounce_indices:
        write = pd.DataFrame(data = {"data":[bounce_df.loc[ind]["counts"].to_numpy().tolist()]} )
        write.to_csv(train_bounce_path,mode="a",header=False)

    for ind in reject_indices:
        if len(reject_df.loc[ind]["counts"].to_numpy().tolist())==201:

            write = pd.DataFrame(data = {"data":[reject_df.loc[ind]["counts"].to_numpy().tolist()]} )
            write.to_csv(train_reject_path,mode="a",header=False)


"""
Training
"""
train_bounce_path = "/home/wyatt/Documents/SAMPEX/generated_Data/94_training_data_bounces.csv"
train_reject_path = "/home/wyatt/Documents/SAMPEX/generated_Data/94_training_data_rejects.csv"

bounce_df = pd.read_csv(train_bounce_path,converters={"data":literal_eval},usecols=[1])
num_bounces = len(bounce_df.index)
reject_df = pd.read_csv(train_reject_path,converters={"data":literal_eval},usecols=[1])
num_rejects = len(reject_df.index)
classifications = [1] * num_bounces + [0] * num_rejects
data = pd.concat([bounce_df,reject_df])
data['class'] = classifications
data = data.sample(frac=1) #shuffle the rows

training_set, validation_set = train_test_split(data, test_size = 0.1, random_state = 21)
X_train = training_set.iloc[:,0:-1].values
Y_train = training_set.iloc[:,-1].values
X_val = validation_set.iloc[:,0:-1].values
y_val = validation_set.iloc[:,-1].values
#Xvals need reformatted
X_train = np.array([item[0] for item in np.array(X_train)])
X_val = np.array([item[0] for item in np.array(X_val)])
X_train = X_train/X_train.max()
X_val = X_val/X_val.max()
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

classifier = MLPClassifier(hidden_layer_sizes=(2**9,2**8,2**7), max_iter=500,
                           activation = 'relu',solver='adam',random_state=1)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_val)
cm = confusion_matrix(y_pred, y_val)
print("Accuracy of MLPClassifier : ", accuracy(cm))

try:
    os.remove("/home/wyatt/Documents/SAMPEX/generated_Data/model.joblib")
except:
    pass
dump(classifier,"/home/wyatt/Documents/SAMPEX/generated_Data/model.joblib")
