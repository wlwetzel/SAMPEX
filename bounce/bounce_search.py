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

"""
searching through low res data
"""
low_res = 0
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

high_res=1
if high_res:
    write_path = '/home/wyatt/Documents/SAMPEX/generated_Data/bouncing_94_v2.csv'
    file_path = '/home/wyatt/Documents/SAMPEX/SAMPEX_Data/HILThires/State2'
    dates = [name[-11:-4] for name in os.listdir(file_path)]

    errors = 0
    for date in dates:
        # try:
        print(date)
        obj = HiltData(date=date)
        data = obj.read(None,None)

        grouped,prominences = peak_algo_v2_high_res(data['Counts'])

        print("number found: ", len(grouped))
        write_df = pd.DataFrame({'date':date,'number':len(grouped) , 'groups':[grouped]})
        write_df.to_csv(write_path,mode='a',header=False)
        # except:
        #     errors+=1
        #     pass
    print(errors)
