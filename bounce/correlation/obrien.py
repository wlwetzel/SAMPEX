import numpy as np
import pandas as pd
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
import SAMP_Data as sp
import os
import spacepy.coordinates as spc
import spacepy.irbempy as irb
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time as t_mod
import plotly.graph_objects as go
import plotly.express as px
# date_today = datetime.now()
# days = pd.date_range(date_today, date_today + timedelta(7), periods=8, tz='UTC')
# data = [0, 1, 2, 3, 4, 5, 6, 7]
#
# df = pd.DataFrame({'timestamp': days, 'values': data})
# df = df.set_index('timestamp')
# dt='2022-02-07T23:18:06.08349'
# dt2='2023-02-07T23:18:06.08349'
#
# idx = abs((df.index - pd.to_datetime(dt2,utc=True))).argmin()
# print(df.iloc[idx])
# quit()
class obrienSearch:
    """docstring for corrSearch."""

    def __init__(self, year):
        """
        year: int, YYYY, year of SAMPEX data to search through
        """
        self.year = year
        self.path = f"/home/wyatt/Documents/SAMPEX/bounce/correlation/data/obrien_{self.year}.csv"
        try:
            os.remove(self.path)
        except:
            pass

    def _transform_data(self,data):
        """
        data: pandas dataframe, 20ms SAMPEX count data
        return: data with rolling 10th percentile subtracted
        TESTING: *Subtracting min value off data chunks to try to get more
                 similar vals for correlation
                 *also trying some small boxcar smoothing
        """
        rolled = data.rolling(5,min_periods=1).mean()
        subtracted = data - data.rolling(10,min_periods=1).quantile(.1)

        return subtracted

    def _group_candidates(self,candidates):
        """
        for organizing the candidates into groupes of times close to one another
        then pick the first time and I'll just look at a 4s window
        """
        group=[]
        master_list=[]
        for ind in range(len(candidates.index)-1):

            if (candidates.index[ind+1] - candidates.index[ind])<pd.Timedelta('1s'):
                group.append(candidates.index[ind])
            else:
                if len(group)!=0: master_list.append(group)
                group=[]

        #the last group won't get added to the master list
        if len(group)!=0: master_list.append(group)
        times = [item[0] for item in master_list ]
        return times

    def _obrien(self,dat):
        # dat = pd.DataFrame(data={"Counts":dat})
        #transform to 100ms looking data
        n_100 = dat.rolling(window=5,center=True).sum()
        a_500 = n_100.rolling(window=25,center=True).mean()
        burst_param = (n_100-a_500)/np.sqrt(1+a_500)
        return pd.DataFrame(data = {"Time":burst_param[burst_param["Counts"]>10].index.to_numpy()})

    def search(self):
        days = ['00' + str(i) for i in range(1,10)] + ['0' + str(i) for i in range(10,100)] + [str(i) for i in range(100,366)]
        #loop through one day at a time
        bounce_num = 0
        for day in days:
            try:
                obj  = sp.HiltData(date=str(self.year)+day)
                data = obj.read(None,None)
                print(day)
            except:
                print("No data for this day.")
                continue

            bursts = self._obrien(data)
            bursts.to_csv(self.path,mode='a',header=None)

class obrienClean:
    """
    overidentified Microbursts, so we're going to pick one out of every group

    basic idea for sorting below

    d = [23,67,110,25,69,24,102,109]

    d.sort()

    diff = [y - x for x, y in zip(*[iter(d)] * 2)]
    avg = sum(diff) / len(diff)

    m = [[d[0]]]

    for x in d[1:]:
        if x - m[-1][0] < avg: #we will do m[-1][-1], so its serial like
            m[-1].append(x)
        else:
            m.append([x])


    print m
    ## [[23, 24, 25], [67, 69], [102, 109, 110]]
    """
    def __init__(self):
        self.years = [1994,1996,1997,1999,2001,2002,2003,2004]
        self.infiles = [f"/home/wyatt/Documents/SAMPEX/bounce/correlation/data/obrien_{year}.csv" for year in self.years]
        self.outfile = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/obrien_times.csv"

    def sort(self):
        for file in self.infiles:
            master_list = []
            print(file)
            df = pd.read_csv(file,parse_dates=True,usecols=[1],header=None,names=["Times"],skiprows=[0])
            times = pd.to_datetime(df["Times"])
            time_diff = pd.Timedelta(60,unit="milli") #ms group with three time bins
            groups = [[times[0]]]
            for time in times[1:]:
                if time - groups[-1][-1] < time_diff:
                    groups[-1].append(time)
                else:
                    groups.append([time])
            #now take the first element from each group
            for group in groups:
                master_list.append(group[0])
            out_df = pd.DataFrame({"Times":master_list})
            out_df.to_csv(file) #overwrites the file we were working on-not a great solution

class obrienDat:
    def __init__(self,years):
        self.years = years
        self.path = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/obrien_"
        self.write_path = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/obrien_stats.csv"

    def _get_L_MLT(self,time):
        timeDelta = pd.Timedelta("5S")
        dataObj = sp.OrbitData(date=time)
        orbitInfo = dataObj.read_time_range(startDate = time,endDate=time+timeDelta, parameters=['GEI_X','GEI_Y','GEI_Z','L_Shell','GEO_Lat',"MLT"])
        return orbitInfo["L_Shell"][0],orbitInfo["MLT"][0]

    def calc_stats(self):
        orb = sp.OrbitData(filename="/home/wyatt/Documents/SAMPEX/OrbitData/PSSet_6sec_1993364_1994025.txt")

        for year in self.years:
            l_list = []
            mlt_list = []
            #need to remove the "time" rows from the data
            times = pd.read_csv(self.path+str(year)+".csv",parse_dates=True,usecols=[1]).to_numpy().flatten()
            data = orb.load_year(year,parameters=["L_Shell","MLT"])
            tot = len(times)
            for ind in range(tot):
                s = data.iloc[abs(data.index - pd.to_datetime(times[ind],utc=True)).argmin()]
                # following is slow af
                # s = data.loc[data.index.unique()[data.index.unique().get_loc(time, method='nearest')]]
                l_list.append(s['L_Shell'])
                mlt_list.append(s["MLT"])
                print(float(ind)/tot * 100,end='\r')
            l_mlt_frame = pd.DataFrame(data = {"L":l_list,"MLT":mlt_list})
            l_mlt_frame.to_csv(self.write_path,mode='a',header=False)
            print(f"{year} done")

    def plot_stats(self):
        df = pd.read_csv(self.write_path,header=None,names=["L","MLT"])
        df = df[df["L"]<10]
        # fig = go.Figure(data=
        #     go.Scatterpolar(
        #         r = df["L"].to_numpy(),
        #         theta = (df["MLT"]*360/24.0).to_numpy(),
        #         mode = 'markers',
        #         marker=dict(
        #             size=10,
        #             showscale=True
        #         )
        #     ))
        # fig.update_layout(
        #     title_text="L vs MLT",
        #     polar = dict(
        #       angularaxis = dict(
        #           tickmode = "array",
        #           tickvals = [int(mlt*360/24.0) for mlt in range(0,24,4)],
        #           ticktext = ["0","4","8","12","16","20"]
        #         )))
        # fig.update_traces(marker_colorbar=dict(title="Time Diff (Bounces)"),
        #                   marker=dict(size=5),
        #                   text=df["period_comp"].to_numpy())
        hist,r_edges,mlt_edges = np.histogram2d(df["L"],df["MLT"])
        R,T = np.meshgrid(r_edges[0:-1],mlt_edges[0:-1])
        hist = hist.T
        fig = go.Figure(data=
            go.Barpolar(
                r = R.ravel(),
                theta =(T.ravel() *360/24),
                marker_color=hist.ravel()
            ))
        fig.update_layout(
            title_text="L - MLT Occurence of Microbursts",
            polar = dict(
              angularaxis = dict(
                  tickmode = "array",
                  tickvals = [int(mlt*360/24.0) for mlt in range(0,24,4)],
                  ticktext = ["0","4","8","12","16","20"]
                )))

        fig.show()
        fig = px.histogram(df["MLT"],nbins=24)
        fig.update_layout(xaxis_title_text="MLT",title_text="MLT Occurence of Microbursts")
        fig.show()
        fig = px.histogram(df["L"],nbins=20)
        fig.update_layout(xaxis_title_text="L-Shell",title_text="L-Shell Occurence of Microbursts")
        fig.show()

#[1994,1996,1997,1999,2001,2002
# years = [1994,1996,
years=[1994,1996,1997,1999,2001,2002,2003,2004]
# years = [2003,2004]
"""
DO NOT TOUCH BELOW
"""
# for year in years:
#     blah = obrienSearch(year)
#     blah.search()
#
# ob = obrienClean()
# ob.sort()

dat = obrienDat(years)
# dat.calc_stats()
dat.plot_stats()
