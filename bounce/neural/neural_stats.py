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
Re = 6378

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
    orbitInfo = dataObj.read_time_range(pd.to_datetime(start),pd.to_datetime(end),parameters=['GEI_X','GEI_Y','GEI_Z','L_Shell','GEO_Lat'])

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
    # print(Lstar)
    Lstar = orbitInfo['L_Shell'].to_numpy()[0]
    loss_cone = find_loss_cone(coords,ticks) #in degrees
    period = 5.62*10**(-2) * Lstar / np.sqrt(energy) * (1-.43 * np.sin(np.deg2rad(loss_cone)))
    return period[0]

def compute_bounce_stats(data,peaks,times):
    """
    peaks are times of where bursts are
    """
    time_diff = pd.Series(np.diff(peaks)).mean(numeric_only=False).total_seconds()
    #how much burst has decreased
    first_peak = float(data.loc[peaks[0]])
    last_peak = float(data.loc[peaks[1]])
    percent_diff = (first_peak - last_peak) / first_peak * 100

    #need to find bounce period of spacecraft
    period = bounce_period(times)

    time_in_period = time_diff/period

    return time_diff,percent_diff,time_in_period

generate_stats=0
make_plots=1
if generate_stats:

    counts_file = "/home/wyatt/Documents/SAMPEX/generated_Data/accepted_predictions_94.csv"
    counts = pd.read_csv(counts_file,header=None,names=["Burst","Time","Counts"],usecols=[0,1,2])
    counts['Time'] = pd.to_datetime(counts['Time'])
    counts = counts.set_index(['Burst','Time'])

    peaks_file = '/home/wyatt/Documents/SAMPEX/generated_Data/neural_peaks_94.csv'
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
            print("founf rejected bounce")

        start = curr_peak_times[0]
        end   = curr_peak_times[-1]

        time_diff,percent_diff,time_in_period = compute_bounce_stats(data,curr_peak_times,(start,end))
        print('\n')
        print(time_diff)
        print(time_in_period)
        print(percent_diff)
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
    path = "/home/wyatt/Documents/SAMPEX/generated_Data/neural_bounce_stats.csv"
    df = pd.DataFrame(data = {'time_diff':times_list,'percent_diff':percents_list,'period_comp':periods_list})
    df.to_csv(path)

if make_plots:
    path = "/home/wyatt/Documents/SAMPEX/generated_Data/neural_bounce_stats.csv"
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
