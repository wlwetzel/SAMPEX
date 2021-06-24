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
import spacepy.coordinates as spc
import spacepy.irbempy as irb


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
import matplotlib.pyplot as plt


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
        chunk_size = 5000 #100s / 20ms * 1000ms/s
        kernels = [self._load_kernel(i) for i in range(5)]
        days = ['00' + str(i) for i in range(1,10)] + ['0' + str(i) for i in range(10,100)] + [str(i) for i in range(100,366)]
        #loop through one day at a time
        bounce_num = 0
        for day in days[255:]:
            try:
                obj  = sp.HiltData(date=str(self.year)+day)
                data = obj.read(None,None)
            except:
                print("No data for this day.")
                continue

            #stick together kernels
            kern = pd.concat(kernels)
            # corr_df = pd.DataFrame(data = {"Corr":self._correlate(data,kern)},
            #                        index=data.index)

            df_dict = {n: data.iloc[n:n+chunk_size,:] for n in range(0,len(data.index),chunk_size)}
            for key,chunk in df_dict.items():
                current_time_series = copy.deepcopy(chunk) #chunk is immutable
                #reunite the correlation with timestamps
                corr_df = pd.DataFrame(data = {"Corr":self._correlate(current_time_series,kern)},
                                       index=current_time_series.index)
                bounce_candidates = corr_df[corr_df["Corr"] > .05]
                if len(bounce_candidates.index)>0:
                    grouped_times = self._group_candidates(bounce_candidates)
                    #we want the counts, times and a unique number assigned to each
                    #loop through each pair in grouped_times, and write each time
                    for time in grouped_times:
                        write_df = data.loc[time-pd.Timedelta('2s'):time+pd.Timedelta('2s'),:]
                        #use obrien parameter to filter things that arent bursts
                        #should improve false positive rate
                        if self._obrien(write_df["Counts"].to_numpy().flatten()):

                            write_df.loc[:,'Bounce']= bounce_num
                            write_df.to_csv(self.candidate_path,mode='a',header=None)
                            bounce_num+=1

            #try multiple kernels, sum results and check
            # corr_df = pd.DataFrame({"Corr":[]})
            # for kern in kernels:
            #     temp_df = pd.DataFrame(data = {"Corr":self._correlate(data,kern)},
            #                        index=data.index)
            #
            #     corr_df = corr_df.add(temp_df,fill_value=0)

            # corr_df['count'] = data["Counts"]
            # fig = make_subplots(specs=[[{"secondary_y": True}]])
            # fig.add_trace(
            #     go.Scatter(x=corr_df.index, y=self._transform_data(corr_df["count"]).to_numpy(), name="count"),
            #     secondary_y=False,
            #     )
            #
            # fig.add_trace(
            #     go.Scatter(x=corr_df.index, y=corr_df["Corr"].to_numpy(), name="corr"),
            #     secondary_y=True,
            #     )
            # fig.show()
            # quit()
            # cutoff = .0012 #determined via trial and error
            # bounce_candidates = corr_df[corr_df["Corr"] > cutoff]
            # if len(bounce_candidates.index)>0:
            #     grouped_times = self._group_candidates(bounce_candidates)
            #     #we want the counts, times and a unique number assigned to each
            #     #loop through each pair in grouped_times, and write each time
            #     for time in grouped_times:
            #         write_df = data.loc[time-pd.Timedelta('3s'):time+pd.Timedelta('3s'),:]
            #         #use obrien parameter to filter things that arent bursts
            #         #should improve false positive rate
            #         if self._obrien(write_df["Counts"].to_numpy().flatten()):
            #             write_df.loc[:,'Bounce']= bounce_num
            #             write_df.to_csv(self.candidate_path,mode='a',header=None)
            #             bounce_num+=1
            #     print(f"{bounce_num} Bounces found so far")


class verifyGui(Tk):
    """docstring for Window."""

    def __init__(self,  parent, year):
        Tk.__init__(self,parent)
        self.parent = parent
        self.path =  "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/candidate_"+str(year)+".csv"
        self.accepted_path = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/accepted_"+str(year)+".csv"
        self.num_accepted=0
        self.num_rejected=0
        try:
            os.remove(self.accepted_path)
        except:
            pass
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

class peak_select:
    def __init__(self,year):
        self.year=year
        self.counts_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/accepted_"+str(year)+".csv"
        self.peaks_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/peaks_"+str(year)+".csv"
        try:
            os.remove(self.peaks_file)
        except:
            pass

    def _on_pick(self,event):
        artist = event.artist
        xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
        x, y = artist.get_xdata(), artist.get_ydata()
        ind = event.ind
        self.peak_times.append(x[ind[0]])
        print(x[ind[0]])

    def select(self):
        bounces = pd.read_csv(self.counts_file,header=None,
                              names=["Burst","Time","Counts"],usecols=[0,1,2])
        indices = bounces['Burst'].drop_duplicates().to_numpy()
        bounces = bounces.set_index(['Burst','Time'])

        peak_times_master = []
        for index in indices:
            self.peak_times = []
            curr_data = bounces.loc[index]
            fig, ax = plt.subplots()

            ax.plot(curr_data.index,curr_data['Counts'],'-b',picker=10)
            fig.canvas.callbacks.connect('pick_event', self._on_pick)
            plt.show()

            peak_times_master.append(self.peak_times)

        print(peak_times_master)
        df = pd.DataFrame(data = {"Peaks":peak_times_master,"Burst":indices})
        df.to_csv(self.peaks_file)


class stats:
    def __init__(self):
        self.stats_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/stats.csv"
        self.peaks_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/peaks_"
        self.counts_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/accepted_"
        self.Re = 6371

    def _to_equatorial(self,position,time,pitch):
        """
        take in spacepy coord class and ticktock class
        """

        blocal = irb.get_Bfield(time,position,extMag='T89')['Blocal']
        beq = irb.find_magequator(time,position,extMag='T89')['Bmin']
        eq_pitch = np.arcsin(np.sqrt(np.sin(np.deg2rad(pitch))**2 * beq / blocal))
        return np.rad2deg(eq_pitch)

    def _find_loss_cone(self,position,time):
        foot = irb.find_footpoint(time,position,extMag='T89')['Bfoot']
        eq = irb.find_magequator(time,position,extMag='T89')['Bmin']

        pitch=90 #for particles mirroring at 100km
        return np.rad2deg(np.arcsin(np.sqrt(np.sin(np.deg2rad(pitch))**2 * eq / foot)))

    def _bounce_period(self,times,energy = 1):
        """
        calculates electron bounce period at edge of loss cone
        energy in MeV
        """
        start = times[0]
        end = times[-1]
        dataObj = sp.OrbitData(date=start)
        orbitInfo = dataObj.read_time_range(pd.to_datetime(start),pd.to_datetime(end),parameters=['GEI_X','GEI_Y','GEI_Z','L_Shell','GEO_Lat'])

        X = (orbitInfo['GEI_X'].to_numpy() / self.Re)[0]
        Y = (orbitInfo['GEI_Y'].to_numpy() / self.Re)[0]
        Z = (orbitInfo['GEI_Z'].to_numpy() / self.Re)[0]
        position  = np.array([X,Y,Z])
        ticks = Ticktock(times[0])
        coords = spc.Coords(position,'GEI','car')

        #something bad is happening with IRBEM, the L values are crazy for a lot of
        #these places, so for now I'll use sampex's provided L vals
        Lstar = irb.get_Lstar(ticks,coords,extMag='0')
        Lstar = abs(Lstar['Lm'][0])

        # Lstar = orbitInfo['L_Shell'].to_numpy()[0]
        self.Lstar = Lstar
        loss_cone = self._find_loss_cone(coords,ticks) #in degrees
        period = 5.62*10**(-2) * Lstar / np.sqrt(energy) * (1-.43 * np.sin(np.deg2rad(loss_cone)))
        return period[0]

    def _compute_bounce_stats(self,data,peaks,times):
        """
        peaks are times of where bursts are
        """
        time_diff = pd.Series(np.diff(peaks)).mean(numeric_only=False).total_seconds()

        #how much burst has decreased
        subtracted = data - data.rolling(10,min_periods=1).quantile(.1)

        first_peak = float(subtracted.loc[peaks[0]])
        last_peak = float(subtracted.loc[peaks[1]])
        percent_diff = (first_peak - last_peak) / first_peak * 100

        #need to find bounce period of spacecraft
        period = self._bounce_period(times)

        time_in_period = time_diff/period

        return time_diff,percent_diff,time_in_period

    def generate_stats(self,use_years="All"):
        try:
            os.remove(self.stats_file)
        except:
            pass

        if use_years=="All":
            years = [1994,1996,1997,1998,1999,2000,2001,2002,2003,2004]
        else:
            years=use_years

        for year in years:
            counts_file = self.counts_file+str(year)+".csv"
            counts = pd.read_csv(counts_file,header=None,names=["Burst","Time","Counts"],usecols=[0,1,2])
            counts['Time'] = pd.to_datetime(counts['Time'])
            counts = counts.set_index(['Burst','Time'])

            peaks_file = self.peaks_file+str(year)+".csv"
            peaks = pd.read_csv(peaks_file,usecols=[1,2])
            peaks['Peaks'] = peaks['Peaks'].apply(literal_eval)
            peaks = peaks.set_index('Burst')
            indices = peaks.index
            times_list    = []
            percents_list = []
            periods_list  = []
            for index in indices:
                data = counts.loc[index]
                curr_peak_times = [pd.Timestamp(peak) for peak in peaks.loc[index][0]]
                if not curr_peak_times:
                    continue
                    print("found rejected bounce")

                start = curr_peak_times[0]
                end   = curr_peak_times[-1]

                time_diff,percent_diff,time_in_period = self._compute_bounce_stats(data,curr_peak_times,(start,end))

                if self.Lstar < 7.0:
                    print(self.Lstar)
                    if percent_diff!=None:
                        times_list.append(time_diff)
                        percents_list.append(percent_diff)
                        periods_list.append(time_in_period)
                else:
                    continue

                # if 3.45 < time_in_period < 3.55:
                #     fig = px.line(data)
                #     fig.update_layout(title_text= f"Example, peak dist = {time_in_period:.2f} bounces, L={self.Lstar:.1f}")
                #     fig.show()
                #
                # if .6 < time_in_period < .65:
                #     fig = px.line(data)
                #     fig.update_layout(title_text= f"Example, peak dist = {time_in_period:.2f} bounces, L={self.Lstar:.1f}")
                #     fig.show()


            df = pd.DataFrame(data = {'time_diff':times_list,'percent_diff':percents_list,'period_comp':periods_list})
            df.to_csv(self.stats_file,mode="a",header=None)

    def plot(self):
        df = pd.read_csv(self.stats_file,names = ["time_diff","percent_diff","period_comp"],usecols=[1,2,3])

        num_bounces = len(df.index)
        fig = px.histogram(np.abs(df['percent_diff'][np.abs(df['percent_diff'])<100]),nbins=50)

        fig.update_layout(title_text = "Percent Difference Between 1st Two Peaks",
                    xaxis_title_text = "Percent")
        fig.write_html("/home/wyatt/Documents/SAMPEX/bounce_figures/PercentDiff.html",include_plotlyjs="cdn")
        fig.show()

        fig = px.histogram(np.abs(df['time_diff']),nbins=40)
        fig.update_layout(title_text = f"Average Time Diff Between Peaks, Total Number of Bouncing Microbursts: {num_bounces}",
                            xaxis_title_text = "Time Diff (s)")
        fig.write_html("/home/wyatt/Documents/SAMPEX/bounce_figures/TimeDiff.html",include_plotlyjs="cdn")
        fig.show()

        fig = px.histogram(np.abs(df['period_comp']),nbins=100)
        fig.update_layout(title_text = "Time Between Peaks Divided By Bounce Period",
                            xaxis_title_text = "(Arb Units)")
        fig.write_html("/home/wyatt/Documents/SAMPEX/bounce_figures/Periods.html",include_plotlyjs="cdn")
        fig.show()



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
