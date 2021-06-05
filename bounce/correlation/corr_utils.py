from sklearn.neural_network import MLPClassifier
import numpy as np
import plotly.express as px
import pandas as pd
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
import SAMP_Data as sp
import plotly.graph_objects as go
from scipy import signal
from spacepy.time import Ticktock
from ast import literal_eval
from joblib import dump, load
from more_itertools import chunked
import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import copy
import numpy as np
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class corrSearch:
    """docstring for corrSearch."""

    def __init__(self, year):
        """
        year: int, YYYY, year of SAMPEX data to search through
        """
        self.year = year
        self.candidate_path = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/candidate_"+str(year)+".csv"
        self.kernel_path = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/kernels.csv"
        try:
            os.remove(self.candidate_path)
        except:
            pass

    def _transform_data(self,data):
        """
        data: pandas dataframe, 20ms SAMPEX count data
        return: data with rolling 10th percentile subtracted
        """
        subtracted = data - data.rolling(10,min_periods=1).quantile(.1)
        return subtracted


    def _correlate(self,data,kernel):
        """
        data: pd dataframe, 20ms count data
        kernel: bounce to compare data to
        """
        length = len(data.index)
        #divide by length of the data to normalize the correlation
        subtracted_data = self._transform_data(data)
        subtracted_kernel = self._transform_data(kernel)
        #zncc normalization
        data_mean = subtracted_data.mean().to_numpy()[0]
        data_std = subtracted_data.std().to_numpy()[0]
        kernel_mean = subtracted_kernel.mean().to_numpy()[0]
        kernel_std = subtracted_kernel.std().to_numpy()[0]
        correlation = signal.correlate(subtracted_data - data_mean,
                                       subtracted_kernel-kernel_mean,
                                       mode='same')/(length*kernel_std*data_std)
        return correlation.flatten()

    def _load_kernel(self,which_kernel):
        kernel = pd.read_csv(self.kernel_path,parse_dates=['Time'])
        kernel = kernel.set_index(['bounce','Time'])
        return kernel.loc[which_kernel]

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
        dat = pd.DataFrame(data={"Counts":dat})
        a_500 = dat.rolling(window=25,center=True).mean()
        n_100 = dat.rolling(window=5,center=True).mean()
        burst_param = (n_100-a_500)/np.sqrt(1+a_500)
        return (burst_param>6).any()["Counts"]

    def search(self):
        chunk_size = 15000 #300s / 20ms * 1000ms/s
        kernels = [self._load_kernel(i) for i in range(4)]
        days = ['00' + str(i) for i in range(1,10)] + ['0' + str(i) for i in range(10,100)] + [str(i) for i in range(100,366)]
        #loop through one day at a time
        bounce_num = 0
        for day in days:
            try:
                obj  = sp.HiltData(date=str(self.year)+day)
                data = obj.read(None,None)
            except:
                print("No data for this day.")
                continue
            #try multiple kernels, sum results and check
            # corr_df = pd.DataFrame({"Corr":[]})
            # for kern in kernels:
            #     temp_df = pd.DataFrame(data = {"Corr":self._correlate(data,kern)},
            #                        index=data.index)
            #
            #     corr_df = corr_df.add(temp_df,fill_value=0)

            #stick together kernels
            kern = pd.concat(kernels)
            corr_df = pd.DataFrame(data = {"Corr":self._correlate(data,kern)},
                                   index=data.index)

            # corr_df['count'] = data["Counts"]
            # fig = make_subplots(specs=[[{"secondary_y": True}]])
            # fig.add_trace(
            #     go.Scatter(x=corr_df.index, y=corr_df["count"].to_numpy(), name="count"),
            #     secondary_y=False,
            #     )
            #
            # fig.add_trace(
            #     go.Scatter(x=corr_df.index, y=corr_df["Corr"].to_numpy(), name="corr"),
            #     secondary_y=True,
            #     )
            # fig.show()
            # quit()
            cutoff = .001 #determined via trial and error
            bounce_candidates = corr_df[corr_df["Corr"] > cutoff]
            if len(bounce_candidates.index)>0:
                grouped_times = self._group_candidates(bounce_candidates)
                #we want the counts, times and a unique number assigned to each
                #loop through each pair in grouped_times, and write each time
                for time in grouped_times:
                    write_df = data.loc[time-pd.Timedelta('3s'):time+pd.Timedelta('3s'),:]
                    #use obrien parameter to filter things that arent bursts
                    #should improve false positive rate
                    if self._obrien(write_df["Counts"].to_numpy().flatten()):
                        write_df.loc[:,'Bounce']= bounce_num
                        write_df.to_csv(self.candidate_path,mode='a',header=None)
                        bounce_num+=1
                print(f"{bounce_num} Bounces found so far")

            print(f"{bounce_num} Bounces found so far")

class verifyGui(Tk):
    """docstring for Window."""

    def __init__(self,  parent, year):
        Tk.__init__(self,parent)
        self.parent = parent
        self.path =  "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/candidate_"+str(year)+".csv"
        self.accepted_path = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/accepted_"+str(year)+".csv"
        self.num_accepted=0
        self.num_rejected=0
        # try:
        #     os.remove(self.accepted_path)
        # except:
        #     pass
        self.initialize()

    def initialize(self):
        #load in data
        predictions = pd.read_csv(self.path,header=None,
                                  names=["Time","Counts","bounce"],usecols=[0,1,2])
        self.predictions = predictions.set_index('bounce',append=False)
        self.total_bounces = len(self.predictions.index.drop_duplicates())
        self.bounce_num = 0 #for keeping track of which bounce we're looking at

        #make figure and place button
        fig = Figure(figsize = (7, 7),
                     dpi = 100)
        plot1 = fig.add_subplot(111)
        self.line1, = plot1.plot(self.predictions.loc[self.bounce_num]['Time'],
                                 self.predictions.loc[self.bounce_num]['Counts'])

        self.canvas = FigureCanvasTkAgg(fig,
                                   master = self)
        self.canvas.draw()
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()
        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas,
                                       self)
        toolbar.update()
        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().pack()

        #we need two buttons, which will store whether or not I think the
        #data contains a bounce
        acceptButton = Button(self,text="Accept",command = self.acceptBounce)
        rejectButton = Button(self,text="Reject",command = self.rejectBounce)
        flagButton   = Button(self,text="Flag",command = self.flagBounce)
        acceptButton.pack()
        rejectButton.pack()
        flagButton.pack()

        self.update()

    def refreshFigure(self):
        self.bounce_num+=1
        os.system('clear')
        print(f"{self.bounce_num} / {self.total_bounces}")
        false_pos = float(self.num_rejected) / (self.num_accepted + self.num_rejected)
        print(f"False Positive Rate: {false_pos * 100.0}")

        y = self.predictions.loc[self.bounce_num]['Counts'].to_numpy()

        date = self.predictions.loc[self.bounce_num]['Time'].iloc[0]
        x = np.arange(len(y))
        self.line1.set_data(x,y)
        ax = self.canvas.figure.axes[0]
        ax.set_title(date)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        self.canvas.draw()

    def acceptBounce(self):
        self.predictions.loc[self.bounce_num].to_csv(self.accepted_path,
                                                     mode='a',header=False)
        self.num_accepted+=1
        self.refreshFigure()

    def flagBounce(self):
        self.predictions.loc[self.bounce_num].to_csv("/home/wyatt/Documents/SAMPEX/generated_Data/flagged_Data.csv",
                                                     mode='a',header=False)
        self.refreshFigure()

    def rejectBounce(self):
        #pass on anything here for now
        self.num_rejected+=1
        self.refreshFigure()

if __name__ == '__main__':
    year = 1996
    blah = corrSearch(year)
    blah.search()
    gu = verifyGui(None,year)
    gu.mainloop()
    #1998 08 16 164238



    # df_dict = {n: data.iloc[n:n+chunk_size,:] for n in range(0,len(data.index),chunk_size)}
    # for key,chunk in df_dict.items():
    #     current_time_series = copy.deepcopy(chunk) #chunk is immutable
    #     #reunite the correlation with timestamps
    #     corr_df = pd.DataFrame(data = {"Corr":self._correlate(current_time_series,kernel)},
    #                            index=current_time_series.index)
    #     bounce_candidates = corr_df[corr_df["Corr"] > 130]
    #     if len(bounce_candidates.index)>0:
    #         grouped_times = self._group_candidates(bounce_candidates)
    #         #we want the counts, times and a unique number assigned to each
    #         #loop through each pair in grouped_times, and write each time
    #         for time in grouped_times:
    #             write_df = data.loc[time-pd.Timedelta('2s'):time+pd.Timedelta('2s'),:]
    #             write_df.loc[:,'Bounce']= bounce_num
    #             write_df.to_csv(self.candidate_path,mode='a',header=None)
    #             bounce_num+=1
