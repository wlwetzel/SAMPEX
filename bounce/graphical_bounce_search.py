"""
GOAL: gather a bunch of bouncing microbursts
I think I should use my algortihm, and identify them graphically and then further filter
RUN IN TERMINAL
"""
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
from bounce_algorithms import peak_algo
from bounce_algorithms import peak_algo_v2
from bounce_algorithms import peak_algo_v2_high_res
from bounce_algorithms import candidate_fft


file = '/home/wyatt/Documents/SAMPEX/generated_Data/bouncing_93_v2.csv'
cols = ['date','num','groups']
sub_list = []
date_list = []
save_path = "/home/wyatt/Documents/SAMPEX/generated_Data/human_filtered_bounces93.csv"

bounces = pd.read_csv(file,names=cols,usecols=[1,2,3])
bounces['groups'] = bounces['groups'].apply(literal_eval)
bounces = bounces[bounces['num']!=0]
print(bounces['num'].sum())
fig = go.Figure()
for i in range(len(bounces.index)):
    date =str(bounces['date'].iloc[i])
    obj = HiltData(date=date)
    data = obj.read(None,None)
    data = data['Rate1']
    group = bounces['groups'].iloc[i]
    #we need to pick a range for the data, loop through groups
    for sub in group:
        lower = sub[0]-20
        upper = sub[-1]+20
        peaks_df = data.iloc[sub]
        current_data = data.iloc[lower:upper]
        fig.add_scatter(x = current_data.index,y=current_data.to_numpy())
        fig.add_scatter(x=peaks_df.index,y=peaks_df.to_numpy(),mode='markers',
                        name="Identified Peaks")
        fig.update_layout(title_text=date)
        fig.show()
        x = input("Contains Bouncing Burst? y/N \n")
        if x=='y':
            #keep in a list: subgroup, date? I think thats it
            sub_list.append(sub)
            date_list.append(date)
            #save
            df = pd.DataFrame(data = {"group":[sub],"date":date})
            df.to_csv(save_path,mode='a',header=False)

        elif x=='N' or x==None:
            pass
        fig.layout = {}
        fig.data = []
