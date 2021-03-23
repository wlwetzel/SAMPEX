import pandas as pd
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
from SAMP_Data import *
import plotly.express as px
from itertools import groupby
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bounce_algorithms import peak_algo
from bounce_algorithms import peak_algo_v2
from bounce_algorithms import peak_algo_v2_high_res
from bounce_algorithms import candidate_fft

"""
Runs a search over SAMPEX data files for bouncing microbursts, using the peak-
find based algorithm
Preliminary search of 1993
Save results in generated_Data/bouncing_93.csv as YYYYDDD,#found,indices found

v2 using burst parameter in bouncing_93_v2.csv
"""
# year = 1992
# day = 278
# date = '1992278'
# obj = HiltData(date=date)
# data = obj.read(14200,14400)
# data = data[['Rate1','Rate2','Rate3','Rate4']]
# x = data['Rate1'].to_numpy()
# grouped,prominences = peak_algo(x)

fft_alg = 0
if fft_alg:
    date = '1994138'
    obj = HiltData(date = date)
    data = obj.read(None,None)
    #60s chunks, with 20ms bins
    num = 60*50
    #i guess just convert to a numpy array
    data = data.rolling(window=30,min_periods=1).mean()

    data = data.to_numpy().flatten()
    #number of bins to loop through
    bins = int(len(data)/num)
    print("num of bins" , bins)
    for i in range(bins):
        print(i / bins*100 ,"%")
        if candidate_fft(data[i*num:(i+1)*num]):
            #we've found a bouncing microburst candidate, so for now just plot and save
            fig = px.line(data[i*num:(i+1)*num])
            fig.update_layout(title_text = "start index" +str(i*num))
            fig.write_html("/home/wyatt/Documents/SAMPEX/bounce_figures/fft/bounce_fft_"+str(i)+".html",include_plotlyjs="cdn")

low_res = 1
if low_res:
    write_path = '/home/wyatt/Documents/SAMPEX/generated_Data/bouncing_93_v2.csv'
    days = ['00' + str(i) for i in range(1,10)] + ['0' + str(i) for i in range(10,100)] + [str(i) for i in range(100,366)]
    dates = ['1993'+day for day in days]
    dates.insert(0,'1992278')
    errors = 0
    for date in dates:
        try:
            print(date)
            obj = HiltData(date=date)
            data = obj.read(None,None)
            data = data[['Rate1','Rate2','Rate3','Rate4']]
            x = data['Rate1'].to_numpy()

            # grouped,prominences = peak_algo(x)

            grouped,prominences = peak_algo_v2(data['Rate1'])

            print("number found: ", len(grouped))
            write_df = pd.DataFrame({'date':date,'number':len(grouped) , 'groups':[grouped]})
            write_df.to_csv(write_path,mode='a',header=False)
        except:
            errors+=1
            pass
    print(errors)

high_res=0
if high_res:
    write_path = '/home/wyatt/Documents/SAMPEX/generated_Data/bouncing_94_v2.csv'
    file_path = '/home/wyatt/Documents/SAMPEX/SAMPEX_Data/HILThires/State2'
    dates = [name[-11:-4] for name in os.listdir(file_path)]
    num_files = len(dates)
    curr = 1
    for date in dates:
        print(date)
        obj = HiltData(date=date)
        data = obj.read(None,None)
        data = data.rolling(window=5,min_periods=1).mean()
        grouped,prominences = peak_algo_v2_high_res(data['Counts'])
        write_df = pd.DataFrame({'date':date,'number':len(grouped) , 'groups':[grouped]})
        write_df.to_csv(write_path,mode='a',header=False)
        print("number found: ", len(grouped))
        print(curr/num_files*100,"%")
        curr+=1
