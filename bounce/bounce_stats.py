import pandas as pd
from ast import literal_eval
import plotly.express as px
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
from SAMP_Data import *
import plotly.graph_objects as go
from scipy import signal
from spacepy.time import Ticktock
import spacepy.coordinates as spc
import spacepy.irbempy as irb


"""
peaks and such are in index space, not seconds
TODO: compare time differneces to bounce periods - irbem blegh #done
"""

Re = 6371 #km
year = "1994"
days =       [138,138,153,153,153,153,150,150,150,150,150,150,150,227,227,151,226,226,145,161,161,158,158,158]
start_hrs =  [21,21,10, 7, 7, 5,22,22,22,22,22,22, 7,13, 0,13,14,14, 7,11,11,12,12,12]
start_mins = [58,58,31,15,15,37,17,17,16,16,16,16, 2,23,33,27,49,49,34,48,48,57,56,56]
start_secs = [25,15, 5,52,40, 5, 5, 0,55,48,46,43, 0, 5,55, 6,25,40,50,43,46, 5,47,55]
end_hrs =    [21,21,10, 7, 7, 5,22,22,22,22,22,22, 7,13, 0,13,14,14, 7,11,11,12,12,12]
end_mins =   [58,58,31,15,15,37,17,17,17,16,16,16, 2,23,34,27,49,49,34,48,48,57,56,56]
end_secs =   [35,20,10,58,46,10,10, 5, 0,51,48,46, 7, 8, 0,10,27,45,55,46,50,10,50,58]

days = [str(day) for day in days]
start_hrs = [str(hr) if len(str(hr))==2 else "0"+str(hr) for hr in start_hrs ]
start_mins = [str(item) if len(str(item))==2 else "0"+str(item) for item in start_mins ]
start_secs = [str(item) if len(str(item))==2 else "0"+str(item) for item in start_secs ]
end_hrs = [str(item) if len(str(item))==2 else "0"+str(item) for item in end_hrs ]
end_mins = [str(item) if len(str(item))==2 else "0"+str(item) for item in end_mins ]
end_secs = [str(item) if len(str(item))==2 else "0"+str(item) for item in end_secs ]

peaks = [ [140,168,199,226],[122,149,183],[152,179,215],[117,154,192],[169,207,239,279],
          [78,113,148],[65,90,116,144],[62,90,116],[153,179,206],[38,52,65,81,94,107,119,137],
          [38,56,68],[35,68,92],[199,254,300],[40,68,103],[18,56,91,129,166,205],[50,86,146],
          [6,43,74],[54,77,114,149],[97,143,190],[55,85],[37,77,108,140],[58,93,125],[34,68,102],
          [7,44,75]
        ]

starts = [pd.to_datetime(year+days[i]+start_hrs[i]+start_mins[i]+start_secs[i],format="%Y%j%H%M%S",utc=True) for i in range(len(days))]
ends = [pd.to_datetime(year+days[i]+end_hrs[i]+end_mins[i]+end_secs[i],format="%Y%j%H%M%S",utc=True) for i in range(len(days))]

def to_equatorial(position,time,pitch):
    """
    take in spacepy coord class and ticktock class
    """

    blocal = irb.get_Bfield(time,position,extMag='T89')['Blocal']
    beq = irb.find_magequator(time,position,extMag='T89')['Bmin']
    eq_pitch = np.arcsin(np.sqrt(np.sin(np.deg2rad(pitch))**2 * beq / blocal))
    return np.rad2deg(eq_pitch)

def find_loss_cone(position,time):
    foot = irb.find_footpoint(time,position,extMag='T89')['Bfoot']
    eq = irb.find_magequator(time,position,extMag='T89')['Bmin']

    pitch=90 #for particles mirroring at 100km
    return np.rad2deg(np.arcsin(np.sqrt(np.sin(np.deg2rad(pitch))**2 * eq / foot)))

def bounce_period(times,energy = 1):
    """
    calculates electron bounce period at edge of loss cone
    energy in MeV
    """
    start = times[0]
    end = times[-1]

    dataObj = OrbitData(date=start)
    orbitInfo = dataObj.read_time_range(start,end,parameters=['GEI_X','GEI_Y','GEI_Z','L_Shell','GEO_Lat'])

    X = (orbitInfo['GEI_X'].to_numpy() / Re)[0]
    Y = (orbitInfo['GEI_Y'].to_numpy() / Re)[0]
    Z = (orbitInfo['GEI_Z'].to_numpy() / Re)[0]
    position  = np.array([X,Y,Z])
    ticks = Ticktock(times[0])
    coords = spc.Coords(position,'GEI','car')
    #something bad is happening with IRBEM, the L values are crazy for a lot of
    #these places, so for now I'll use sampex's provided L vals
    # Lstar = irb.get_Lstar(ticks,coords,extMag='T89')
    # Lstar = Lstar['Lm']
    Lstar = orbitInfo['L_Shell'].to_numpy()[0]
    loss_cone = find_loss_cone(coords,ticks) #in degrees
    period = 5.62*10**(-2) * Lstar / np.sqrt(energy) * (1-.43 * np.sin(np.deg2rad(loss_cone)))
    return period[0]

def compute_bounce_stats(data,peaks,times):
    """
    peaks are indices of data where bursts are
    """

    #distance between peaks
    diffs = np.diff(peaks)
    time_diff = np.average(diffs)*20 *10**-3 #time in seconds
    #how much burst has decreased
    first_peak = float(data[peaks[0]])
    last_peak = float(data[peaks[1]])
    percent_diff = (first_peak - last_peak) / first_peak * 100

    #need to find bounce period of spacecraft
    period = bounce_period(times)

    time_in_period = time_diff/period

    return time_diff,percent_diff,time_in_period

def compute_bounce_stats_low_res(peaks,data):
    """
    for some reason I did the low res in a completely different way to the high
    res stuff
    """
    #distance between peaks
    diffs = [t2 - t1 for t1,t2 in zip(peaks[:-1], peaks[1:])]
    time_diff = np.mean(diffs).total_seconds()

    #how much burst has decreased
    first_peak = float(data.loc[peaks[0]])
    last_peak = float(data.loc[peaks[1]])
    percent_diff = (first_peak - last_peak) / first_peak * 100

    #need to find bounce period of spacecraft
    period = bounce_period(peaks)
    time_in_period = time_diff/period

    return time_diff,percent_diff,time_in_period


counter = 0
generate_stats = 0
make_plots = 1
if generate_stats:
    times_list    = []
    percents_list = []
    periods_list  = []
    for day in days:
        obj  = HiltData(date=str(year)+str(day))
        data = obj.read(None,None)
        data = data[starts[counter]:ends[counter]]
        data = data.to_numpy().flatten()
        # peaks,_ = signal.find_peaks(data,prominence=(100,None),distance = 10,width = 5)
        time_diff,percent_diff,time_in_period = compute_bounce_stats(data,peaks[counter],(starts[counter],ends[counter]))
        print('\n')
        print(counter)
        print(time_in_period)
        print('\n')
        if percent_diff!=None:
            times_list.append(time_diff)
            percents_list.append(percent_diff)
            periods_list.append(time_in_period)
        #uncomment to plot all
        # fig = px.line(data)
        # trace = go.Scatter(x= peaks[counter],y = data[peaks[counter]],mode = 'markers')
        # fig.add_trace(trace)
        # fig.show()
        counter+=1
    path = "/home/wyatt/Documents/SAMPEX/generated_Data/bounce_stats.csv"
    df = pd.DataFrame(data = {'time_diff':times_list,'percent_diff':percents_list,'period_comp':periods_list})
    df.to_csv(path)

stats_path = "/home/wyatt/Documents/SAMPEX/generated_Data/bounce_stats.csv"
generate_low_res_stats = 0
if generate_low_res_stats:
    times_list    = []
    percents_list = []
    periods_list  = []
    #there are two files to load - the time series, and the peak locations
    series_path = '/home/wyatt/Documents/SAMPEX/generated_Data/93bounces_quicker_read.csv'
    peak_path = '/home/wyatt/Documents/SAMPEX/generated_Data/peaks_93.csv'
    series_df = pd.read_csv(series_path,parse_dates=['Time'])

    series_df = series_df.set_index(['Burst','Time'])
    peak_df = pd.read_csv(peak_path)
    peak_df = peak_df['Peaks'].apply(literal_eval)
    for index in range(94):
        peak_df.index = pd.to_datetime(peak_df.index)
        group = peak_df.iloc[index]
        group = [pd.to_datetime(i,utc=True) for i in group]
        data = series_df.loc[str(index)]
        data.index = pd.to_datetime(data.index)
        time_diff,percent_diff,time_in_period = compute_bounce_stats_low_res(group,data)

        print(index)
        print('\n')
        times_list.append(time_diff)
        percents_list.append(percent_diff)
        periods_list.append(time_in_period)
        #uncomment to plot all
        # fig = px.line(data)
        # trace = go.Scatter(x=pd.to_datetime(group),y=data.loc[group].to_numpy().flatten(),mode='markers',
        #                 name="Identified Peaks")
        # fig.add_trace(trace)
        # fig.show()
    df = pd.DataFrame(data = {'time_diff':times_list,'percent_diff':percents_list,'period_comp':periods_list})
    df.to_csv(stats_path,mode='a',header=False)

if make_plots:
    path = "/home/wyatt/Documents/SAMPEX/generated_Data/bounce_stats.csv"
    df = pd.read_csv(path)
    fig = px.histogram(np.abs(df['percent_diff']),nbins=20)
    fig.update_layout(title_text = "Percent Difference Between 1st Two Peaks",
                xaxis_title_text = "Percent")
    fig.write_html("/home/wyatt/Documents/SAMPEX/bounce_figures/PercentDiff.html",include_plotlyjs="cdn")
    fig.show()

    fig = px.histogram(np.abs(df['time_diff']),nbins=20)
    fig.update_layout(title_text = "Average Time Diff Between Peaks",
                        xaxis_title_text = "Time Diff (s)")
    fig.write_html("/home/wyatt/Documents/SAMPEX/bounce_figures/TimeDiff.html",include_plotlyjs="cdn")
    fig.show()

    fig = px.histogram(np.abs(df['period_comp']),nbins=20)
    fig.update_layout(title_text = "Time Between Peaks Divided By Bounce Period",
                        xaxis_title_text = "(Arb Units)")
    fig.write_html("/home/wyatt/Documents/SAMPEX/bounce_figures/Periods.html",include_plotlyjs="cdn")
    fig.show()

check_series = 0
if check_series:
    #1 bounce period
    which = 0
    obj  = HiltData(date=str(year)+str(days[which]))
    data = obj.read(None,None)
    data = data[starts[which]:ends[which]]
    fig = px.line(data)
    data = data.to_numpy().flatten()
    L=9
    tau = .5
    time_diff,percent_diff,time_in_period = compute_bounce_stats(data,peaks[which],(starts[which],ends[which]))
    fig.update_layout(title_text=f'Dist b/w Peaks ~ {time_diff:.1f}s , L~{L}, Bounce Period = {tau}s ')
    fig.show()

    #3 bounce period
    which = 5
    obj  = HiltData(date=str(year)+str(days[which]))
    data = obj.read(None,None)
    data = data[starts[which]:ends[which]]
    fig = px.line(data)
    data = data.to_numpy().flatten()
    time_diff,percent_diff,time_in_period = compute_bounce_stats(data,peaks[which],(starts[which],ends[which]))
    L=3.7
    tau = .19
    fig.update_layout(title_text=f'Dist b/w Peaks ~ {time_diff:.1f}s , L~{L}, Bounce Period = {tau}s')
    fig.show()
    #4,5,13,14,15,16,17
