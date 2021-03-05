import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
from SAMP_Data import *
import scipy.fft as fft
from scipy import signal
import plotly.express as px
from itertools import groupby
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import time
# data.to_csv(path)

# path = '/home/wyatt/Documents/SAMPEX/test_dat.csv'
# data = pd.read_csv(path)
# time = data.index
"""
high res bouncing microburst example
date = '1994138'
obj = HiltData(date=date)
data = obj.read(75600+3600-120+26,75600+3600-120+32)
fig = px.line(data)
fig.show()
"""


def peak_algo(x):
    """

    """
    peaks, _ = signal.find_peaks(x,prominence=(500,None))
    peaks = list(peaks)
    print(peaks)
    #groups successive peaks
    #i have forgotten entirely how this next line works
    grouped = [list(g) for k, g in groupby(peaks[:-1],lambda x: (peaks[peaks.index(x)+1] -peaks[peaks.index(x)])<5) if k]
    grouped = [item for item in grouped if len(item)>2]

    # filtered = [item for item in grouped if np.all(np.diff(x[item]) >=0)]

    #see if a line with negative slope can be reasonably fit
    line = lambda x,m,b: m*x+b
    line_list = []
    num = 0
    keep_list = []
    for group in grouped:
        popt,pcov = curve_fit(line,group,x[group])
        line_list.append(popt)
        if popt[0]< (-50) and np.sqrt(np.diag(pcov))[0]<100:
            keep_list.append(num)
        num+=1

    grouped = [grouped[i] for i in keep_list]
    line_list = [line_list[i] for i in keep_list]

    filtered = list(itertools.chain.from_iterable(grouped)) #looks like it just flattens
    prominences = signal.peak_prominences(x,filtered)[0]

    return grouped,prominences

def burst_param(x):
    """
    takes in pandas dataframe
    returns indices of peaks
    """
    a_500 = x.rolling(window=5,center=True).mean()
    burst_param = (x-a_500)/np.sqrt(1+a_500)
    bool = burst_param>9.5
    # bursts = x[burst_param>5]
    bursts = np.flatnonzero(bool)
    return bursts

def peak_algo_v2(x):
    """
    takes in pandas dataframe
    """

    peaks = burst_param(x)
    peaks = list(peaks)

    #groups successive peaks
    #i have forgotten entirely how this next line works
    grouped = [list(g) for k, g in groupby(peaks[:-1],lambda x: (peaks[peaks.index(x)+1] -peaks[peaks.index(x)])<5) if k]
    grouped = [item for item in grouped if len(item)>2]

    # filtered = [item for item in grouped if np.all(np.diff(x[item]) >=0)]

    #see if a line with negative slope can be reasonably fit
    line = lambda x,m,b: m*x+b
    line_list = []
    num = 0
    keep_list = []
    for group in grouped:
        popt,pcov = curve_fit(line,group,x[group])
        line_list.append(popt)
        if popt[0]< (-40) and np.sqrt(np.diag(pcov))[0]<150:
            keep_list.append(num)
        num+=1

    grouped = [grouped[i] for i in keep_list]
    line_list = [line_list[i] for i in keep_list]

    filtered = list(itertools.chain.from_iterable(grouped)) #looks like it just flattens
    prominences = signal.peak_prominences(x,filtered)[0]

    return grouped,prominences

def peak_algo_v2_high_res(x):

    peaks, _ = signal.find_peaks(x,prominence=(75,None),distance=25)
    peaks = list(peaks)
    #groups successive peaks
    grouped = []
    small  = []
    start = time.time()
    for i in range(len(peaks)-1):
        if (peaks[i+1]-peaks[i])<75 and (peaks[i+1]-peaks[i])>20 :
            small.append(peaks[i])
        elif len(small) !=0:
            small.append(peaks[i])
            grouped.append(small)
            small = []
    grouped = [item for item in grouped if len(item)>2]
    #try to improve algorithm by requiring that all peaks be strictly decreasing

    # new_grouped = []
    # for item in grouped:
    #     if np.any(np.diff(x[item])<0):
    #         new_grouped.append(item)
    # grouped = [item for item in new_grouped]

    line = lambda x,amp,pow: amp*x**(-pow)
    line_list = []
    num = 0
    keep_list = []
    for group in grouped:
        try:
            #reset each group to start at index=0, to fit to exponential
            reset_group = [el - group[0]+10 for el in group]
            popt,pcov = curve_fit(line,reset_group, x[group]  ,p0=[200,2])
            line_list.append(popt)

            if popt[1]> .1 and np.sqrt(np.diag(pcov))[1]<10:
                keep_list.append(num)
            num+=1
        except:
            line_list.append([])
            num+=1

    grouped = [grouped[i] for i in keep_list]
    line_list = [line_list[i] for i in keep_list]
    filtered = list(itertools.chain.from_iterable(grouped)) #looks like it just flattens
    prominences = signal.peak_prominences(x,filtered)[0]

    return grouped,prominences


def corr_algo(x):
    """
    use a sample decreasing sawtooth and use cross correlation
    """
    sawtooth = np.array([1.5,2.9,1.7,1,1.5,1.9,1.2,.7,1,1.3,.75,.5,.65,.85,.5,.3,.45,.6,.4,.2])
    corr = signal.correlate(sawtooth,x)

    return corr

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=10):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def candidate_fft(x):
    """
    For high res data only-timestep is hardcoded
    param x : counts
    """
    #detrend
    data = x - np.mean(x)
    #take transform
    trans = fft.fft(data)
    power = abs(trans)
    #get frequencies
    timestep = 20 *10**-3 #20ms
    # freqs = fft.fftfreq(len(x))
    #index for .5s and 2s
    length = len(x)
    ind2 = int(length*timestep * 1/.5)
    ind1 = int(length*timestep * 1/2.0)

    #this doesn't really work - maybe using signal.find_peaks
    # mean = np.mean(power)
    # condition_arr = power[ind1:ind2]>mean
    # return np.any(condition_arr)


    #lotta power in the 1 and -1 index, and 2 and -2 and they arent relevant so I'll remove them
    # power[-1]=0
    # power[1]=0
    # power[-2]=0
    # power[2]=0
    # print(np.max(power))
    # power = power / np.max(power)
    # print(power[ind1])
    # peaks,_ = signal.find_peaks(power,prominence = (.2,None))
    # indices = [i for i in range(ind1,ind2+1)]
    # print(peaks)
    # return any(elem in peaks for elem in indices)
    #that didnt really work
    diff = ind2-ind1
    #try comparing power in region of interest to power at high freqs
    interest = np.sum(power[ind1:ind2])
    compare = np.sum(power[int(length/2-diff/2):int(length/2+diff/2)])
    return interest/compare > 15


if __name__ == '__main__':

    fft_flag = 0
    if fft_flag:
        date = '1994138'
        obj = HiltData(date=date)
        data = obj.read(75600+3600-120+60,75600+3600-120+120)
        # data = obj.read(75600+3600-120+0,75600+3600-120+60)
        data = data.rolling(window=10,min_periods=1).mean()
        data = data.to_numpy().flatten()
        #lets try giving this some rolling data
        trans = fft.fft(data-np.mean(data))
        print(candidate_fft(data))
        power = np.abs(trans)
        power = power/np.max(power)
        fig = px.line(power)
        fig.show()
        fig = px.line(np.abs(data))
        fig.show()

    high_res=1
    if high_res:
        """
        testing for the high res
        """
        date = '1994138'
        obj = HiltData(date=date)
        data = obj.read(75600+3600-120,75600+3600-120+100)
        # grouped,prominences = peak_algo_v2_high_res(data['Counts'])
        rolling = data.rolling(window=5,min_periods=1).mean()
        grouped,prominences = peak_algo_v2_high_res(rolling['Counts'])
        print(grouped)
        print(prominences)
        fig = px.line(rolling.to_numpy().flatten())
        # fig.add_trace(go.Scatter(x = rolling.index,y = rolling.to_numpy().flatten()))
        fig.show()

    use_peaks = False
    if use_peaks:
        grouped,prominences = peak_algo_v2(data['Rate1'])
        peaks = list(itertools.chain.from_iterable(grouped))
        heights = x[peaks]-prominences
        fig = px.line(x)
        fig.add_scatter( x=peaks,y=x[peaks] ,mode='markers')
        for i in range(len(peaks)):
            fig.add_shape(
            # Line Vertical
            dict(
            type="line",
            x0=peaks[i],
            y0=heights[i],
            x1=peaks[i],
            y1=x[peaks[i]],
            line=dict(
            color="Black",
            width=1
            )
            ))
            # fig.update_yaxes(type="log")
        # for item in line_list:
        #     yvals = line(indices,item[0],item[1])
        #     fig.add_scatter(x=indices,y=yvals)
        fig.show()

    corr_al = False
    if corr_al:
        filtered_data = butter_highpass_filter(x,.1,fs)
        corr = signal.correlate(filtered_data,x[1120:1150])
        corr = corr_algo(filtered_data)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        indices = [i for i in range(len(x))]
        fig.add_trace(
            go.Scatter(x=indices, y=x, name="yaxis data"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=indices, y=corr, name="yaxis2 data"),
            secondary_y=True,
        )
        # fig.update_yaxes(type="log")

        fig.show()
