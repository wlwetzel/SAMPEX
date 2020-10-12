import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
from SAMP_Data import *
from scipy import signal
import plotly.express as px
from itertools import groupby
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

# data.to_csv(path)

# path = '/home/wyatt/Documents/SAMPEX/test_dat.csv'
# data = pd.read_csv(path)
# time = data.index

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


def corr_algo(x):
    """
    use a sample decreasing sawtooth and use cross correlation
    """
    sawtooth = np.array([1.5,2.9,1.7,1,1.5,1.9,1.2,.7,1,1.3,.75,.5,.65,.85,.5,.3,.45,.6,.4,.2])
    corr = signal.correlate(sawtooth,x)

    return corr

fs = 1/.1 #sample frequency

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=10):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


if __name__ == '__main__':

    year = 1992
    day = 278
    date = datetime_from_doy(year,day)
    date = '1992278'
    obj = HiltData(date=date)
    data = obj.read(14200,14400)
    data = data[['Rate1','Rate2','Rate3','Rate4']]
    use_peaks = True
    # t=np.linspace(0,2,20)
    # sawtooth = (signal.sawtooth(np.pi *5*t,width=.5) + 1)*np.exp(-t) + 1.5*np.exp(-t)

    x = data['Rate1'].to_numpy()
    indices = np.array(list(range(len(x))))
    line = lambda x,m,b: m*x+b

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
