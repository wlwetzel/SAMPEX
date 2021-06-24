import pandas as pd
import plotly.express as px
import spacepy.coordinates as spc
import spacepy.irbempy as irb
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
import SAMP_Data as sp
import numpy as np
from spacepy.time import Ticktock
from ast import literal_eval
import plotly.graph_objects as go
import os

Re = 6371
years = [1994,1996,1997,1998,1999,2000,2001,2002,2003,2004]
magField = "T96"
path  = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/irbem_comparisons_"+magField+".csv"
def calc_Lvals(year):

    counts_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/accepted_"+str(year)+".csv"
    times = pd.read_csv(counts_file,header=None,names=["Burst","Time"],usecols=[0,1])
    times['Time'] = pd.to_datetime(times['Time'])
    times = times.set_index('Burst')
    irb_L_list = []
    samp_L_list = []
    indices = times.index.drop_duplicates()
    for ind in indices:
        print(ind)
        start = times.loc[ind].iloc[0]["Time"]
        end = times.loc[ind].iloc[-1]["Time"]
        dataObj = sp.OrbitData(date=start)
        orbitInfo = dataObj.read_time_range(pd.to_datetime(start),pd.to_datetime(end),parameters=['GEI_X','GEI_Y','GEI_Z','L_Shell'])

        X = (orbitInfo['GEI_X'].to_numpy() / Re)[0]
        Y = (orbitInfo['GEI_Y'].to_numpy() / Re)[0]
        Z = (orbitInfo['GEI_Z'].to_numpy() / Re)[0]
        position  = np.array([X,Y,Z])
        ticks = Ticktock(start)
        coords = spc.Coords(position,'GEI','car')
        irb_Lstar = irb.get_Lstar(ticks,coords,extMag=magField)
        irb_Lstar = irb_Lstar['Lm'][0]
        samp_Lstar = orbitInfo['L_Shell'].to_numpy()[0]
        irb_L_list.append(irb_Lstar[0])
        samp_L_list.append(samp_Lstar)
    return irb_L_list,samp_L_list

def gen_stats():
    try:
        os.remove(path)
    except:
        pass

    for year in years:
        print(year)
        irb_L_list,samp_L_list = calc_Lvals(year)
        write_df = pd.DataFrame(data = {"irb_L" :irb_L_list , "samp_L":samp_L_list})
        write_df.to_csv(path,mode="a",index=False,header=None)

#generate the comparisons
gen_stats()

stats = pd.read_csv(path,names = ["irb_L","samp_L"])
stats["diff"] = stats["irb_L"].abs() - stats["samp_L"].abs()

# fig = px.histogram(stats["diff"])
fig = go.Figure(
    data = [go.Scatter(x = stats['samp_L'].to_numpy() , y= stats["diff"].to_numpy(),mode="markers" )]
)
fig.update_layout(title_text="Difference b/w irbem and sampex L, With "+magField,
                  xaxis_title_text="Sampex L",yaxis_title_text="Difference")
fig.write_html("/home/wyatt/Documents/SAMPEX/bounce_figures/irb_comp_"+magField+".html",include_plotlyjs="cdn")

fig.show()
