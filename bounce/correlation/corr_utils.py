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
        TESTING: *Subtracting min value off data chunks to try to get more
                 similar vals for correlation
                 *also trying some small boxcar smoothing
        """
        rolled = data.rolling(5,min_periods=1).mean()
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

    def _load_artifical_kernel(self,distance):
        #distnace should be provided in seconds
        burst = lambda t,amp,dist: amp * np.exp(-.5*(t-dist)**2 / (.05**2) )
        total_time = 5 #s
        samples = int(5/(20*10**-3))
        times = np.linspace(0,total_time,samples)
        bounce = None
        for n in range(1,5):
            if bounce is None:
                bounce = burst(times,1/n,n*distance)
            else:
                bounce += burst(times,1/n,n*distance)
        return pd.DataFrame(bounce)

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
        #transform to 100ms looking data
        n_100 = dat.rolling(window=5,center=True).sum()
        a_500 = n_100.rolling(window=25,center=True).mean()
        burst_param = (n_100-a_500)/np.sqrt(1+a_500)
        return (burst_param>5).any()["Counts"]

    def search(self):
        chunk_size = 250 #5s / 20ms * 1000ms/s
        # kernels = [self._load_kernel(i) for i in range(5)]
        distances = [.05,.75,.1,.15,.2,.3,.5,.75,.8,1.0]
        kernels = [self._load_artifical_kernel(dist) for dist in distances]
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

            corr_master_dict = {i:pd.DataFrame(data = {"Corr":self._correlate(data,kernels[i])},
                                       index=data.index) for i in range(len(kernels))}
            data_dict = {n: data.iloc[n:n+chunk_size,:] for n in range(0,len(data.index),chunk_size)}
            keys = [key for key in data_dict]
            for master_key in corr_master_dict:
                corr_df = corr_master_dict[master_key]
                # corr_df = corr_df/corr_df.max()
                bounce_candidates = corr_df[corr_df>.0006]
                # corr_df["count"] = data
                # fig = make_subplots(specs=[[{"secondary_y": True}]])
                # fig.add_trace(
                #     go.Scatter(x=corr_df.index, y=corr_df["count"].to_numpy(), name="count"),
                #     secondary_y=False,
                #     )
                #
                # fig.add_trace(
                #     go.Scatter(x=corr_df.index - pd.Timedelta('2s'), y=corr_df["Corr"].to_numpy(), name="corr"),
                #     secondary_y=True,
                #     )
                # fig.show()
                # quit()
                corr_dict = {n: bounce_candidates.iloc[n:n+chunk_size,:] for n in range(0,len(bounce_candidates.index),chunk_size)}
                keys_to_remove = []
                for key in keys:
                    if corr_dict[key]["Corr"].any():
                        write_df = copy.deepcopy(data_dict[key])
                        if self._obrien(write_df["Counts"].to_numpy().flatten()):
                            write_df.loc[:,'Bounce']= bounce_num
                            write_df.to_csv(self.candidate_path,mode='a',header=None)
                            bounce_num+=1
                            print(f"{bounce_num} bounces found so far")
                        keys_to_remove.append(key)
                for rem in keys_to_remove:
                    keys.remove(rem)


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

class doubleCheck(Tk):
    """
    for going through the bounces fuc k i dont care
    """

    def __init__(self,  parent, year):
        Tk.__init__(self,parent)
        self.parent = parent
        self.accepted_path = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/accepted_"+str(year)+".csv"
        self.reviewed_path = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/reviewed_"+str(year)+".csv"
        self.peaks_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/peaks_"+str(year)+".csv"

        self.counter = 0
        try:
            os.remove(self.reviewed_path)
        except:
            pass

        self.initialize()

    def _on_pick(self,event):
        artist = event.artist
        xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
        x, y = artist.get_xdata(), artist.get_ydata()
        ind = event.ind
        self.reselected_peaks.append(x[ind[0]])
        print(x[ind[0]])

    def initialize(self):
        self.reselected_peaks = [] #for selecting new peaks, have to put it here cause i suck
        #load in data
        predictions = pd.read_csv(self.accepted_path,header=None,
                                  names=["bounce","Time","Counts"],usecols=[0,1,2],
                                  parse_dates=[1])
        peaks = pd.read_csv(self.peaks_file,usecols=[1,2])
        peaks['Peaks'] = peaks['Peaks'].apply(literal_eval)
        self.bursts = peaks["Burst"].to_numpy()

        self.predictions = predictions.set_index(['bounce',"Time"],append=False)
        self.peaks = peaks.set_index('Burst',append=False)
        self.total_bounces = len(self.predictions.index.drop_duplicates())

        self.bounce_num = 0 #for keeping track of which bounce we're looking at

        ind = self.peaks.loc[self.bursts[self.bounce_num]]['Peaks']
        curr_peak_times = [pd.Timestamp(peak,tz="UTC") for peak in ind]
        self.curr_peak_times = curr_peak_times

        #make figure and place button
        fig = Figure(figsize = (7, 7),
                     dpi = 100)
        plot1 = fig.add_subplot(111)
        self.line1,self.line2, = plot1.plot(self.predictions.loc[self.bursts[self.bounce_num]].index,
                                 self.predictions.loc[self.bursts[self.bounce_num]]['Counts'],
                                 "b-",
                                 curr_peak_times,
                                 self.predictions.loc[self.bursts[self.bounce_num]]['Counts'][curr_peak_times],
                                 "rx",
                                 picker=5
                                 )
        # fig.canvas.callbacks.connect('pick_event', self._on_pick)

        self.canvas = FigureCanvasTkAgg(fig,
                                   master = self)
        self.canvas.mpl_connect('pick_event', self._on_pick)
        self.canvas.draw()
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()
        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas,
                                       self)
        toolbar.update()
        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().pack()

        acceptButton = Button(self,text="Accept",command = self.acceptBounce)
        rejectButton = Button(self,text="Reject",command = self.rejectBounce)
        redoButton   = Button(self,text="Redo",command = self.redoBounce)
        acceptButton.pack()
        rejectButton.pack()
        redoButton.pack()

        self.update()

    def refreshFigure(self):
        self.reselected_peaks = [] #for selecting new peaks, have to put it here cause i suck
        self.bounce_num+=1
        os.system('clear')
        ind = self.peaks.loc[self.bursts[self.bounce_num]]['Peaks']
        curr_peak_times = [pd.Timestamp(peak,tz="UTC") for peak in ind]
        self.curr_peak_times = curr_peak_times
        x = self.predictions.loc[self.bursts[self.bounce_num]].index
        y = self.predictions.loc[self.bursts[self.bounce_num]]['Counts']
        self.line1.set_data(x,y)

        self.line2.set_data(
                    curr_peak_times,
                    self.predictions.loc[self.bursts[self.bounce_num]]['Counts'][curr_peak_times]
                    )
        self.line2.set_marker(">")
        ax = self.canvas.figure.axes[0]
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max()+200)
        self.canvas.draw()

    def redraw_new_peaks(self):
        os.system('clear')
        x = self.predictions.loc[self.bursts[self.bounce_num]].index
        y = self.predictions.loc[self.bursts[self.bounce_num]]['Counts']
        print(self.curr_peak_times)
        self.line1.set_data(x,y)
        self.line2.set_data(
                    self.curr_peak_times,
                    self.predictions.loc[self.bursts[self.bounce_num]]['Counts'][self.curr_peak_times]
                    )
        self.line2.set_marker(">")
        ax = self.canvas.figure.axes[0]
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max()+200)
        self.canvas.draw()
        self.reselected_peaks = [] #for selecting new peaks, have to put it here cause i suck

    def acceptBounce(self):
        curr_prediction = self.predictions.loc[self.bursts[self.bounce_num]]
        self.curr_peak_times = list(set(self.curr_peak_times))
        curr_prediction["Peaks"] = 0
        curr_prediction["Peaks"][self.curr_peak_times]=1
        curr_prediction["Burst"] = self.counter
        curr_prediction.to_csv(self.reviewed_path,mode="a",header=False)
        self.counter+=1
        self.refreshFigure()

    def redoBounce(self):
        self.curr_peak_times = self.reselected_peaks
        self.redraw_new_peaks()

    def rejectBounce(self):
        #dont need to do anything on reject, just advance to next bounce
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
    def __init__(self,stats_file = "stats.csv",energy=1,mirr=1):
        """
        use the stats_file kw for messing around
        """
        self.mirror = mirr # if we want to specify manually a mirror lat
        self.energy=energy
        self.stats_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/"+stats_file
        self.counts_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/reviewed_"
        self.figure_path = "/home/wyatt/Documents/SAMPEX/bounce_figures/"
        self.Re = 6371
        if mirr != 1:
            self.name = str(energy*(10**-3))+"keV"+str(mirr) + "deg"
        else:
            self.name = "1MeV"

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

    def _eq_pitch_lat(self,mirror_lat):
        """
        calculates the equatorial pitch angle of a particle mirroring at
        mirror_lat (deg)
        """
        eq = np.arcsin(np.sqrt((((np.cos(np.deg2rad(mirror_lat)))**6) / np.sqrt(1 + 3*np.sin(np.deg2rad(mirror_lat))**2))))
        return np.rad2deg(eq)

    def _bounce_period(self,times,energy = 1):
        """
        calculates electron bounce period at edge of loss cone
        energy in MeV
        """
        start = times[0]
        end = times[-1]
        dataObj = sp.OrbitData(date=start)
        orbitInfo = dataObj.read_time_range(pd.to_datetime(start),pd.to_datetime(end),parameters=['GEI_X','GEI_Y','GEI_Z','L_Shell','GEO_Lat',"MLT"])
        if  (orbitInfo['GEO_Lat'].to_numpy() / self.Re)[0]>0:
            self.hemisphere = "N"
        else:
            self.hemisphere = "S"

        X = (orbitInfo['GEI_X'].to_numpy() / self.Re)[0]
        Y = (orbitInfo['GEI_Y'].to_numpy() / self.Re)[0]
        Z = (orbitInfo['GEI_Z'].to_numpy() / self.Re)[0]
        position  = np.array([X,Y,Z])
        ticks = Ticktock(times[0])
        coords = spc.Coords(position,'GEI','car')

        Lstar = irb.get_Lstar(ticks,coords,extMag='0')
        Lstar = abs(Lstar['Lm'][0])

        #these are going here so I dont have to make another call to orbitData
        self.Lstar = Lstar
        self.mlt   = (orbitInfo["MLT"].to_numpy())[0]
        #here we switch if we're finding based on the mirror lat or saying the
        #particles are lost to the atmosphere
        if not self.mirror==1:
            eq_pitch = self._eq_pitch_lat(self.mirror)
        else:
            eq_pitch = self._find_loss_cone(coords,ticks) #in degrees

        mc2 = .511 #elec rest mass
        beta = np.sqrt(energy/mc2)*np.sqrt(2 + energy/mc2) / (1 + energy/mc2)

        period = .117 * Lstar * (1 - .4635 * np.sin(np.deg2rad(eq_pitch))**.75) / beta
        return period[0]

    def _compute_bounce_stats(self,data,peaks,times,energy=1):
        """
        peaks are times of where bursts are
        """
        time_diff = pd.Series(np.diff(peaks)).mean(numeric_only=False).total_seconds()
        # print(pd.Series(np.diff(peaks)))
        #how much burst has decreased
        subtracted = data - data.rolling(10,min_periods=1).quantile(.1)
        first_peak = float(subtracted.loc[peaks[0]])
        last_peak = float(subtracted.loc[peaks[1]])
        if first_peak == float(0):
            #what the fuck
            print("peak was zero for some reason")
            first_peak = float(subtracted.loc[peaks[1]])
            last_peak = float(subtracted.loc[peaks[2]])

        percent_diff = (first_peak - last_peak) / first_peak * 100

        #need to find bounce period of spacecraft
        period = self._bounce_period(times,energy)

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
            counts = pd.read_csv(counts_file,header=None,names=["Time","Counts","Peaks","Burst"],usecols=[0,1,2,3])
            counts['Time'] = pd.to_datetime(counts['Time'])
            peaks = counts[counts["Peaks"]==1][["Time","Burst"]]
            indices = list(set(peaks["Burst"].to_numpy()))
            peaks = peaks.set_index(["Burst"])
            counts = counts.set_index(['Burst',"Time"])["Counts"]
            times_list    = []
            percents_list = []
            periods_list  = []
            hemisphere_list = []
            lstar_list = []
            mlt_list = []
            for index in indices:
                data = counts.loc[index]
                curr_peak_times = peaks.loc[index]["Time"].to_numpy().tolist()

                if not curr_peak_times:
                    print("found rejected bounce")
                    continue

                start = curr_peak_times[0]
                end   = curr_peak_times[-1]

                time_diff,percent_diff,time_in_period = self._compute_bounce_stats(data,curr_peak_times,(start,end),self.energy)

                if self.Lstar < 7.0:
                    if percent_diff!=None:
                        times_list.append(time_diff)
                        percents_list.append(percent_diff)
                        periods_list.append(time_in_period)
                        hemisphere_list.append(self.hemisphere)
                        lstar_list.append(self.Lstar[0])
                        mlt_list.append(self.mlt)
                else:
                    continue

            df = pd.DataFrame(data = {'time_diff':times_list,
                'percent_diff':percents_list,'period_comp':periods_list,
                'hemisphere':hemisphere_list,'lstars':lstar_list,"MLT":mlt_list})
            df.to_csv(self.stats_file,mode="a",header=None)

    def plot(self):
        df = pd.read_csv(self.stats_file,names = ["time_diff","percent_diff",
                        "period_comp","hemisphere","MLT"],usecols=[1,2,3,4,5])
        df[["percent_diff","time_diff","period_comp"]] = np.abs(df[["percent_diff","time_diff","period_comp"]])
        num_bounces = len(df.index)
        fig = px.histogram(df[['percent_diff','hemisphere']][np.abs(df['percent_diff'])<100],nbins=25,color="hemisphere")
        df_hems = df[np.abs(df['percent_diff'])<100].set_index("hemisphere")["percent_diff"]
        means = df_hems.groupby(level="hemisphere").mean()
        medians = df_hems.groupby(level="hemisphere").median()
        mean_N = means["N"]
        mean_S = means["S"]
        median_N = medians["N"]
        median_S = medians["S"]

        fig.update_layout(title_text = f"Percent Difference Between 1st Two Peaks, Means: N:{mean_N:.0f} S:{mean_S:.0f},Medians: N:{median_N:.0f} S:{median_S:.0f}, " +str(self.energy)+" MeV",
                    xaxis_title_text = "Percent")
        fig.write_html(self.figure_path+self.name + "PercentDiff.html",include_plotlyjs="cdn")
        fig.show()

        fig = px.histogram(df[['time_diff',"hemisphere"]],nbins=40,color="hemisphere")
        fig.update_layout(title_text = f"Average Time Diff Between Peaks, Total Number of Bouncing Microbursts: {num_bounces}, "+str(self.energy)+" MeV",
                            xaxis_title_text = "Time Diff (s)")
        fig.write_html(self.figure_path + self.name + "TimeDiff.html",include_plotlyjs="cdn")
        fig.show()

        fig = px.histogram(df[['period_comp',"hemisphere"]],nbins=40,color="hemisphere")
        fig.update_layout(title_text = "Time Between Peaks Divided By Bounce Period, "+str(self.energy)+" MeV",
                            xaxis_title_text = "Bounce Periods",xaxis_range=[0,3])
        fig.write_html(self.figure_path + self.name + "Periods.html",include_plotlyjs="cdn")
        fig.show()

        fig = px.scatter_polar()
        #DIAL PLOT
class plots:
    def __init__(self,year):
        self.year = year
        self.Re = 6371
        self.counts_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/reviewed_"

        #not sure how to do this i hate myself fuck i suck at everything

    def _to_equatorial(self,position,time,pitch):
        """
        take in spacepy coord class and ticktock class
        """

        blocal = irb.get_Bfield(time,position,extMag='T89')['Blocal']
        beq = irb.find_magequator(time,position,extMag='T89')['Bmin']
        eq_pitch = np.arcsin(np.sqrt(np.sin(np.deg2rad(pitch))**2 * beq / blocal))
        return np.rad2deg(eq_pitch)

    def _find_loss_cone(self,position,time):
        foot = irb.find_footpoint(time,position,extMag='T89')
        foot = foot['Bfoot']
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
        if  (orbitInfo['GEO_Lat'].to_numpy() / self.Re)[0]>0:
            self.hemisphere = "N"
        else:
            self.hemisphere = "S"

        X = (orbitInfo['GEI_X'].to_numpy() / self.Re)[0]
        Y = (orbitInfo['GEI_Y'].to_numpy() / self.Re)[0]
        Z = (orbitInfo['GEI_Z'].to_numpy() / self.Re)[0]
        position  = np.array([X,Y,Z])
        ticks = Ticktock(times[0])
        coords = spc.Coords(position,'GEI','car')

        Lstar = irb.get_Lstar(ticks,coords,extMag='0')
        Lstar = abs(Lstar['Lm'][0])

        # Lstar = orbitInfo['L_Shell'].to_numpy()[0]
        self.Lstar = Lstar
        loss_cone = self._find_loss_cone(coords,ticks) #in degrees
        mc2 = .511 #elec rest mass
        beta = np.sqrt(energy/mc2)*np.sqrt(2 + energy/mc2) / (1 + energy/mc2)

        period = .117 * Lstar * (1 - .4635 * np.sin(np.deg2rad(loss_cone))**.75) / beta
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
        if first_peak == float(0):
            #what the fuck
            print("peak was zero for some reason")
            first_peak = float(subtracted.loc[peaks[1]])
            last_peak = float(subtracted.loc[peaks[2]])

        percent_diff = (first_peak - last_peak) / first_peak * 100

        #need to find bounce period of spacecraft
        period = self._bounce_period(times)

        time_in_period = time_diff/period

        return time_diff,percent_diff,time_in_period

    def _search(self,bounce_period,num):

        counts_file = self.counts_file+str(self.year)+".csv"
        counts = pd.read_csv(counts_file,header=None,names=["Time","Counts","Peaks","Burst"],usecols=[0,1,2,3])
        counts['Time'] = pd.to_datetime(counts['Time'])
        peaks = counts[counts["Peaks"]==1][["Time","Burst"]]
        indices = list(set(peaks["Burst"].to_numpy()))
        peaks = peaks.set_index(["Burst"])
        counts = counts.set_index(['Burst',"Time"])["Counts"]

        times_list    = []
        percents_list = []
        periods_list  = []
        hemisphere_list = []
        counts_master_list = []
        lstar_list = []
        count = 0
        for index in indices:
            print(index)
            data = counts.loc[index]
            curr_peak_times = peaks.loc[index]["Time"].to_numpy().tolist()

            if not curr_peak_times:
                continue
                print("found rejected bounce")

            start = curr_peak_times[0]
            end   = curr_peak_times[-1]

            time_diff,percent_diff,time_in_period = self._compute_bounce_stats(data,curr_peak_times,(start,end))

            if .85*bounce_period < time_in_period < 1.15*bounce_period and self.Lstar<7:
                counts_master_list.append(data)
                hemisphere_list.append(self.hemisphere)
                lstar_list.append(self.Lstar)
                periods_list.append(time_in_period)
                count+=1
            else:
                continue
            if num==count:
                break
        return counts_master_list,hemisphere_list,lstar_list,periods_list
    def plot(self,bounce_period=1,num=2):
        counts,hemispheres,lstars,periods = self._search(bounce_period,num)
        for i in range(num):
            fig = px.line(counts[i])

            fig.update_layout(title_text=f"L val: {lstars[i][0]:.1f}, Hemisphere: {hemispheres[i]}, Periods:{periods[i]:.1f}",
                              xaxis_title_text="UTC",yaxis_title_text="Counts")
            fig.show()

if __name__ == '__main__':
    # gu = doubleCheck(None,2004)
    # gu.mainloop()

    # plot_obj = plots_v2(1999)
    # plot_obj.plot(bounce_period=1.5,num=10)
    # year = 1994
    # blah = corrSearch(year)
    # blah.search()
    # gu = verifyGui(None,year)
    # gu.mainloop()
    #1998 08 16 164238
    pass
    quit()

#the old stuff
#
# class stats:
#     def __init__(self):
#         self.stats_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/stats.csv"
#         self.peaks_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/peaks_"
#         self.counts_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/accepted_"
#         #
#         # self.stats_file = "/media/wyatt/64A5-F009/corr_dat/stats.csv"
#         # self.peaks_file = "/media/wyatt/64A5-F009/corr_dat/peaks_"
#         # self.counts_file = "/media/wyatt/64A5-F009/corr_dat/accepted_"
#
#         self.Re = 6371
#
#     def _to_equatorial(self,position,time,pitch):
#         """
#         take in spacepy coord class and ticktock class
#         """
#
#         blocal = irb.get_Bfield(time,position,extMag='T89')['Blocal']
#         beq = irb.find_magequator(time,position,extMag='T89')['Bmin']
#         eq_pitch = np.arcsin(np.sqrt(np.sin(np.deg2rad(pitch))**2 * beq / blocal))
#         return np.rad2deg(eq_pitch)
#
#     def _find_loss_cone(self,position,time):
#         foot = irb.find_footpoint(time,position,extMag='T89')['Bfoot']
#         eq = irb.find_magequator(time,position,extMag='T89')['Bmin']
#
#         pitch=90 #for particles mirroring at 100km
#         return np.rad2deg(np.arcsin(np.sqrt(np.sin(np.deg2rad(pitch))**2 * eq / foot)))
#
#     def _bounce_period(self,times,energy = 1):
#         """
#         calculates electron bounce period at edge of loss cone
#         energy in MeV
#         """
#         start = times[0]
#         end = times[-1]
#         dataObj = sp.OrbitData(date=start)
#         orbitInfo = dataObj.read_time_range(pd.to_datetime(start),pd.to_datetime(end),parameters=['GEI_X','GEI_Y','GEI_Z','L_Shell','GEO_Lat'])
#         if  (orbitInfo['GEO_Lat'].to_numpy() / self.Re)[0]>0:
#             self.hemisphere = "N"
#         else:
#             self.hemisphere = "S"
#
#         X = (orbitInfo['GEI_X'].to_numpy() / self.Re)[0]
#         Y = (orbitInfo['GEI_Y'].to_numpy() / self.Re)[0]
#         Z = (orbitInfo['GEI_Z'].to_numpy() / self.Re)[0]
#         position  = np.array([X,Y,Z])
#         ticks = Ticktock(times[0])
#         coords = spc.Coords(position,'GEI','car')
#
#         Lstar = irb.get_Lstar(ticks,coords,extMag='0')
#         Lstar = abs(Lstar['Lm'][0])
#
#         # Lstar = orbitInfo['L_Shell'].to_numpy()[0]
#         self.Lstar = Lstar
#         loss_cone = self._find_loss_cone(coords,ticks) #in degrees
#         mc2 = .511 #elec rest mass
#         beta = np.sqrt(energy/mc2)*np.sqrt(2 + energy/mc2) / (1 + energy/mc2)
#
#         period = .117 * Lstar * (1 - .4635 * np.sin(np.deg2rad(loss_cone))**.75) / beta
#         return period[0]
#
#     def _compute_bounce_stats(self,data,peaks,times):
#         """
#         peaks are times of where bursts are
#         """
#         time_diff = pd.Series(np.diff(peaks)).mean(numeric_only=False).total_seconds()
#
#         #how much burst has decreased
#         subtracted = data - data.rolling(10,min_periods=1).quantile(.1)
#         first_peak = float(subtracted.loc[peaks[0]])
#         last_peak = float(subtracted.loc[peaks[1]])
#         if first_peak == float(0):
#             #what the fuck
#             print("peak was zero for some reason")
#             first_peak = float(subtracted.loc[peaks[1]])
#             last_peak = float(subtracted.loc[peaks[2]])
#
#         percent_diff = (first_peak - last_peak) / first_peak * 100
#
#         #need to find bounce period of spacecraft
#         period = self._bounce_period(times)
#
#         time_in_period = time_diff/period
#
#         return time_diff,percent_diff,time_in_period
#
#     def generate_stats(self,use_years="All"):
#         try:
#             os.remove(self.stats_file)
#         except:
#             pass
#
#         if use_years=="All":
#             years = [1994,1996,1997,1998,1999,2000,2001,2002,2003,2004]
#         else:
#             years=use_years
#
#         for year in years:
#             counts_file = self.counts_file+str(year)+".csv"
#             counts = pd.read_csv(counts_file,header=None,names=["Burst","Time","Counts"],usecols=[0,1,2])
#             counts['Time'] = pd.to_datetime(counts['Time'])
#             counts = counts.set_index(['Burst','Time'])
#
#             peaks_file = self.peaks_file+str(year)+".csv"
#             peaks = pd.read_csv(peaks_file,usecols=[1,2])
#             peaks['Peaks'] = peaks['Peaks'].apply(literal_eval)
#             peaks = peaks.set_index('Burst')
#             indices = peaks.index
#             times_list    = []
#             percents_list = []
#             periods_list  = []
#             hemisphere_list = []
#             for index in indices:
#                 data = counts.loc[index]
#                 curr_peak_times = [pd.Timestamp(peak) for peak in peaks.loc[index][0]]
#                 res = []
#                 [res.append(x) for x in curr_peak_times if x not in res]
#
#                 curr_peak_times=res
#
#                 if not curr_peak_times:
#                     continue
#                     print("found rejected bounce")
#
#                 start = curr_peak_times[0]
#                 end   = curr_peak_times[-1]
#
#                 time_diff,percent_diff,time_in_period = self._compute_bounce_stats(data,curr_peak_times,(start,end))
#
#                 if self.Lstar < 5.0:
#                     if percent_diff!=None:
#                         times_list.append(time_diff)
#                         percents_list.append(percent_diff)
#                         periods_list.append(time_in_period)
#                         hemisphere_list.append(self.hemisphere)
#                 else:
#                     continue
#
#             df = pd.DataFrame(data = {'time_diff':times_list,
#                 'percent_diff':percents_list,'period_comp':periods_list,
#                 'hemisphere':hemisphere_list})
#             df.to_csv(self.stats_file,mode="a",header=None)
#
#     def plot(self):
#         df = pd.read_csv(self.stats_file,names = ["time_diff","percent_diff",
#                         "period_comp","hemisphere"],usecols=[1,2,3,4])
#         df[["percent_diff","time_diff","period_comp"]] = np.abs(df[["percent_diff","time_diff","period_comp"]])
#         num_bounces = len(df.index)
#         fig = px.histogram(df[['percent_diff','hemisphere']][np.abs(df['percent_diff'])<100],nbins=25,color="hemisphere")
#         mean1 = (df[['percent_diff','hemisphere']][np.abs(df['percent_diff'])<100]).mean()
#         median1 = (df[['percent_diff','hemisphere']][np.abs(df['percent_diff'])<100]).median()
#         fig.update_layout(title_text = f"Percent Difference Between 1st Two Peaks, mean {mean1},median {median1}",
#                     xaxis_title_text = "Percent")
#         fig.write_html("/home/wyatt/Documents/SAMPEX/bounce_figures/PercentDiff.html",include_plotlyjs="cdn")
#         fig.show()
#
#         fig = px.histogram(df[['time_diff',"hemisphere"]],nbins=40,color="hemisphere")
#         fig.update_layout(title_text = f"Average Time Diff Between Peaks, Total Number of Bouncing Microbursts: {num_bounces}",
#                             xaxis_title_text = "Time Diff (s)")
#         fig.write_html("/home/wyatt/Documents/SAMPEX/bounce_figures/TimeDiff.html",include_plotlyjs="cdn")
#         fig.show()
#
#         fig = px.histogram(df[['period_comp',"hemisphere"]],nbins=37,color="hemisphere")
#         fig.update_layout(title_text = "Time Between Peaks Divided By Bounce Period",
#                             xaxis_title_text = "(Arb Units)",xaxis_range=[0,3])
#         fig.write_html("/home/wyatt/Documents/SAMPEX/bounce_figures/Periods.html",include_plotlyjs="cdn")
#         fig.show()
#
# class plots:
#     def __init__(self,year):
#         self.year = year
#         self.Re = 6371
#         self.counts_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/accepted_"+str(year)+".csv"
#         self.peaks_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/peaks_"+str(year)+".csv"
#
#         #not sure how to do this i hate myself fuck i suck at everything
#
#     def _to_equatorial(self,position,time,pitch):
#         """
#         take in spacepy coord class and ticktock class
#         """
#
#         blocal = irb.get_Bfield(time,position,extMag='T89')['Blocal']
#         beq = irb.find_magequator(time,position,extMag='T89')['Bmin']
#         eq_pitch = np.arcsin(np.sqrt(np.sin(np.deg2rad(pitch))**2 * beq / blocal))
#         return np.rad2deg(eq_pitch)
#
#     def _find_loss_cone(self,position,time):
#         foot = irb.find_footpoint(time,position,extMag='T89')
#         print(foot)
#         foor = foor['Bfoot']
#         eq = irb.find_magequator(time,position,extMag='T89')['Bmin']
#
#         pitch=90 #for particles mirroring at 100km
#         return np.rad2deg(np.arcsin(np.sqrt(np.sin(np.deg2rad(pitch))**2 * eq / foot)))
#
#     def _bounce_period(self,times,energy = 1):
#         """
#         calculates electron bounce period at edge of loss cone
#         energy in MeV
#         """
#         start = times[0]
#         end = times[-1]
#         dataObj = sp.OrbitData(date=start)
#         orbitInfo = dataObj.read_time_range(pd.to_datetime(start),pd.to_datetime(end),parameters=['GEI_X','GEI_Y','GEI_Z','L_Shell','GEO_Lat'])
#         if  (orbitInfo['GEO_Lat'].to_numpy() / self.Re)[0]>0:
#             self.hemisphere = "N"
#         else:
#             self.hemisphere = "S"
#
#         X = (orbitInfo['GEI_X'].to_numpy() / self.Re)[0]
#         Y = (orbitInfo['GEI_Y'].to_numpy() / self.Re)[0]
#         Z = (orbitInfo['GEI_Z'].to_numpy() / self.Re)[0]
#         position  = np.array([X,Y,Z])
#         ticks = Ticktock(times[0])
#         coords = spc.Coords(position,'GEI','car')
#
#         Lstar = irb.get_Lstar(ticks,coords,extMag='T05')
#         Lstar = abs(Lstar['Lm'][0])
#
#         # Lstar = orbitInfo['L_Shell'].to_numpy()[0]
#         self.Lstar = Lstar
#         loss_cone = self._find_loss_cone(coords,ticks) #in degrees
#         period = 5.62*10**(-2) * Lstar / np.sqrt(energy) * (1-.43 * np.sin(np.deg2rad(loss_cone)))
#         return period[0]
#
#     def _compute_bounce_stats(self,data,peaks,times):
#         """
#         peaks are times of where bursts are
#         """
#         time_diff = pd.Series(np.diff(peaks)).mean(numeric_only=False).total_seconds()
#
#         #how much burst has decreased
#         subtracted = data - data.rolling(10,min_periods=1).quantile(.1)
#
#         first_peak = float(subtracted.loc[peaks[0]])
#         last_peak = float(subtracted.loc[peaks[1]])
#         if first_peak == float(0):
#             #what the fuck
#             print("peak was zero for some reason")
#             first_peak = float(subtracted.loc[peaks[1]])
#             last_peak = float(subtracted.loc[peaks[2]])
#
#         percent_diff = (first_peak - last_peak) / first_peak * 100
#
#         #need to find bounce period of spacecraft
#         period = self._bounce_period(times)
#
#         time_in_period = time_diff/period
#
#         return time_diff,percent_diff,time_in_period
#
#     def _search(self,bounce_period,num):
#
#         counts = pd.read_csv(self.counts_file,header=None,names=["Burst","Time","Counts"],usecols=[0,1,2])
#         counts['Time'] = pd.to_datetime(counts['Time'])
#         counts = counts.set_index(['Burst','Time'])
#
#         peaks = pd.read_csv(self.peaks_file,usecols=[1,2])
#         peaks['Peaks'] = peaks['Peaks'].apply(literal_eval)
#         peaks = peaks.set_index('Burst')
#         indices = peaks.index
#
#         times_list    = []
#         percents_list = []
#         periods_list  = []
#         hemisphere_list = []
#         counts_master_list = []
#         lstar_list = []
#         count = 0
#         for index in indices:
#             data = counts.loc[index]
#             curr_peak_times = [pd.Timestamp(peak) for peak in peaks.loc[index][0]]
#             if not curr_peak_times:
#                 continue
#                 print("found rejected bounce")
#
#             start = curr_peak_times[0]
#             end   = curr_peak_times[-1]
#
#             time_diff,percent_diff,time_in_period = self._compute_bounce_stats(data,curr_peak_times,(start,end))
#
#             if .85*bounce_period < time_in_period < 1.15*bounce_period and self.Lstar<7:
#                 counts_master_list.append(data)
#                 hemisphere_list.append(self.hemisphere)
#                 lstar_list.append(self.Lstar)
#                 periods_list.append(time_in_period)
#                 count+=1
#             else:
#                 continue
#             if num==count:
#                 break
#         return counts_master_list,hemisphere_list,lstar_list,periods_list
#     def plot(self,bounce_period=1,num=2):
#         counts,hemispheres,lstars,periods = self._search(bounce_period,num)
#         for i in range(num):
#             fig = px.line(counts[i])
#
#             fig.update_layout(title_text=f"L val: {lstars[i][0]:.1f}, Hemisphere: {hemispheres[i]}, Periods:{periods[i]:.1f}",
#                               xaxis_title_text="UTC",yaxis_title_text="Counts")
#             fig.show()
