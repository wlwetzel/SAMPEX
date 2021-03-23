# %%
import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from scipy import signal
sys.path.append('/home/wyatt/pyModules')
import SAMP_Data
import SAMPEXreb
import datetime
import matplotlib.animation as animation
from Wavelet import wavelet,wave_signif,invertWave
from scipy.integrate import simps

%matplotlib inline

#%%
start0 = pd.Timestamp('1997-01-10 05:40:00' , tz = 'utc')
end0 = pd.Timestamp('1997-01-10 05:47:00' , tz = 'utc')
start1  = pd.Timestamp('1997-03-01 00:17:00' , tz = 'utc')
end1  = pd.Timestamp('1997-03-01 00:40:00' , tz = 'utc')
start2  = pd.Timestamp('1997-03-01 01:00:00' , tz = 'utc')
end2  = pd.Timestamp('1997-03-01 01:10:00' , tz = 'utc')
start3  = pd.Timestamp('1997-03-01 01:15:00' , tz = 'utc')
end3  = pd.Timestamp('1997-03-01 01:25:00' , tz = 'utc')
start4  = pd.Timestamp('1997-03-01 01:45:00' , tz = 'utc')
end4  = pd.Timestamp('1997-03-01 02:25:00' , tz = 'utc')
start5  = pd.Timestamp('1997-03-01 02:25:00' , tz = 'utc')
end5  = pd.Timestamp('1997-03-01 03:25:00' , tz = 'utc')
start6 = pd.Timestamp('1997-10-01 10:35:00' , tz = 'utc')
end6 = pd.Timestamp('1997-10-01 10:45:00' , tz = 'utc')
start7 = pd.Timestamp('1997-01-07 01:42:00' , tz = 'utc')
end7 = pd.Timestamp('1997-01-07 01:44:00' , tz = 'utc')
start8 = pd.Timestamp('1997-01-09 07:07:00' ,tz = 'utc')
end8 =  pd.Timestamp('1997-01-09 07:12:00' , tz = 'utc')
start9  = pd.Timestamp('1997-03-01 00:15:00' , tz = 'utc')
end9  = pd.Timestamp('1997-03-01 00:50:00' , tz = 'utc')
start10  = pd.Timestamp('1997-03-01 01:00:00' , tz = 'utc')
end10  = pd.Timestamp('1997-03-01 01:10:00' , tz = 'utc')
start11  = pd.Timestamp('1997-03-01 01:15:00' , tz = 'utc')
end11  = pd.Timestamp('1997-03-01 01:25:00' , tz = 'utc')
start12  = pd.Timestamp('1997-03-01 01:50:00' , tz = 'utc')
end12  = pd.Timestamp('1997-03-01 02:25:00' , tz = 'utc')
start13  = pd.Timestamp('1997-03-01 02:25:00' , tz = 'utc')
end13  = pd.Timestamp('1997-03-01 03:25:00' , tz = 'utc')
start14  = pd.Timestamp('1997-03-01 03:25:00' , tz = 'utc')
end14  = pd.Timestamp('1997-03-01 04:25:00' , tz = 'utc')
start15  = pd.Timestamp('1997-03-01 04:25:00' , tz = 'utc')
end15  = pd.Timestamp('1997-03-01 05:25:00' , tz = 'utc')

eventList = []
for i in range(0,15):
    eventList.append([eval('start' + str(i)),eval('end'+str(i))])
#%%
"""
Lets look at whole days
"""
hours = ['0' + str(i) if i<10 else str(i) for i in range(24)]
eventList = [[pd.Timestamp('1997-03-01'+ ' ' +hour+ ':00:00' , tz = 'utc'),
              pd.Timestamp('1997-03-01'+ ' ' +hour+ ':59:59' , tz = 'utc')]
             for hour in hours]
#%%
def wavAnalysis(data):
    mean = np.mean(data)
    maxScale=5
    variance = np.std(data, ddof=1) ** 2
    n = len(data)
    dt = .1
    dj = .1
    s0 = 2*dt
    j1 = int(10 / dj)
    mother = 'DOG'
    pad = 1
    lag1 = .1

    wave, period, scale, coi = wavelet(data, dt, pad, dj, s0, j1, mother)
    power = np.abs(wave)**2

    signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale,
                                            lag1=lag1, mother=mother)
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand signif --> (J+1)x(N) array
    sig95 = power / sig95  # where ratio > 1, power is significant
    mask = sig95>1
    denoisedWav = mask * wave
    depower = np.abs(denoisedWav)**2
    mask = scale<maxScale
    mask = mask[:,np.newaxis].dot(np.ones(n)[np.newaxis,:])
    denoisedWav *=mask
    newdat = invertWave(mother, scale, dt, dj, denoisedWav)
    return newdat


def compare(range1,range2):
    if range1[0]<=range2[0] and range1[1]>=range2[1]: #if 1 encapsulates 2
        return True
    elif range1[0]>=range2[0] and range1[1]<=range2[1]: #if 2 encapsulates 1
        return True
    elif abs(range1[0]-range2[0])<2 and abs(range1[1]-range2[1])<2: #both bounds are within one of each other
        return True
    else:
        return False

"""
pitch Info for each detector
"""
def findPitches(times):
    real_Start = pd.to_datetime(times[0],utc=True)
    real_End = pd.to_datetime(times[-1],utc=True)

    dataObj = SAMP_Data.OrbitData(date=real_Start)
    df = dataObj.read_time_range(real_Start,real_End,parameters=['A11', 'A21', 'A31', 'A12', 'A22',
    'A32', 'A13', 'A23', 'A33','B_X','B_Y','B_Z','B_Mag','Loss_Cone_2'])
    #use the direction cosine array to convert the Bfield to body fixed coordinates
    mag_angle = np.degrees(np.arcsin(-1/df['B_Mag'] * np.abs(df['A11']*df['B_X'] + df['A12']*df['B_Y']+ df['A13']*df['B_Z']))).to_numpy()
    #the viewing angles for each detector
    det_1_Alpha = -8.879 - mag_angle
    det_1_Beta = 32  - mag_angle
    det_2_Alpha = -17.351 - mag_angle
    det_2_Beta = 25.110 - mag_angle
    det_3_Alpha = -25.110 - mag_angle
    det_3_Beta = 17.351 - mag_angle
    det_4_Alpha = -32 - mag_angle
    det_4_Beta = 8.879 - mag_angle
    d = {'det_1_Alpha' : det_1_Alpha , 'det_1_Beta' : det_1_Beta,'det_2_Alpha' : det_2_Alpha , 'det_2_Beta' : det_2_Beta,'det_3_Alpha' : det_3_Alpha , 'det_3_Beta' : det_3_Beta,'det_4_Alpha' : det_4_Alpha , 'det_4_Beta' : det_4_Beta,'LossCone' : df['Loss_Cone_2'].to_numpy()}
    pitchInfo = pd.DataFrame(data=d,index=df.index.values)
    pitchInfo = pitchInfo.resample('100ms').asfreq()
    return pitchInfo.interpolate(method='polynomial',order=7)

def microBins(start,end):
    """
    takes a start and end time, backgound subtracts via wavelet analysis,
    finds peaks, finds the count rate for each peak in each channel, finds the
    angle of the magnetic field and the angle range each channel looks in,
    divides each channel's angle range and returns a list of "bins" (i.e. angle
    values) and weights (countrates) for use in making a histogram
    """
    workDir = '/home/wyatt/Documents/SAMPEX/'
    filename = "/home/wyatt/Documents/SAMPEX/data/HILThires/"
    if len(str(start.dayofyear))==1:
        day = '00'+str(start.dayofyear)
    elif len(str(start.dayofyear))==2:
        day = '0'+str(start.dayofyear)
    else:
        day = str(start.dayofyear)
    filename = filename + 'State4/hhrr1997' + day+'.txt'
    data = SAMPEXreb.quick_read(filename,start,end)
    times = data.index.values
    burstDF = pd.DataFrame(data=None,index = times)

    """
    wavelet analysis
    """

    cols = ['Rate1','Rate2','Rate3','Rate4']
    for col in cols:
        burstDF[col] = wavAnalysis(data[col].to_numpy())

    """
    peak finding
    """
    resDF = pd.DataFrame(data=None)
    #finds peaks and the areas under each in each channel
    for col in cols:
        dat = burstDF[col].to_numpy()
        peaks, _ = signal.find_peaks(burstDF[col],height=300,distance=3,rel_height=.1)
        results = signal.peak_widths(burstDF[col],peaks)
        resArr = np.array(results[2:])
        resArr[0,:] = np.floor(resArr[0,:])
        resArr[1,:] = np.floor(resArr[1,:])
        resArr = resArr.astype(int)
        resArr = np.vstack((resArr,peaks) )
        resDF[col]=[resArr]

    # resDF has three rows, the first two are the FWHM and the third is the peak location

    first = resDF['Rate1'].to_numpy()[0]
    second = resDF['Rate2'].to_numpy()[0]
    third = resDF['Rate3'].to_numpy()[0]
    fourth = resDF['Rate4'].to_numpy()[0]

    rangeDF = pd.DataFrame(columns = cols)
    timeDF = pd.DataFrame(columns = cols)
    #logic for associating peaks to one another
    for i in range(np.shape(first)[1]):
        #now we need to compare this to each other channel
        for j in range(np.shape(second)[1]):
            if compare(second[0:2,j],first[0:2,i]):
                for k in range(np.shape(third)[1]):
                    if compare(third[0:2,k],first[0:2,i]):
                        for l in range(np.shape(fourth)[1]):
                            if compare(fourth[0:2,l],first[0:2,i]):
                                tempDF = pd.DataFrame(data={'Rate1':[first[0:2,i]],'Rate2':[second[0:2,j]],'Rate3':[third[0:2,k]],'Rate4':[fourth[0:2,l]]})
                                tempTimeDF = pd.DataFrame(data={'Rate1':[first[2,i]],'Rate2':[second[2,j]],'Rate3':[third[2,k]],'Rate4':[fourth[2,l]]})
                                rangeDF = rangeDF.append(tempDF,ignore_index=True)
                                timeDF = timeDF.append(tempTimeDF,ignore_index=True)

    #times at peaks
    timeDF = timeDF.loc[timeDF.astype(str).drop_duplicates().index]
    rangeDF = rangeDF.loc[rangeDF.astype(str).drop_duplicates().index]

    peakTimes = times[timeDF['Rate1'].to_numpy().astype(int)]
    """
    get the countrates for each peak
    """
    countrates = rangeDF
    for i in range(len(rangeDF.index)):
        for col in cols:
            ran = rangeDF[col].iloc[i]
            countrates[col].iloc[i] = simps(dat[ran[0]:ran[1]+1] ,dx=.1)

    pitchInfo = findPitches(times)
    #get max and min look directions
    maxLook = pitchInfo.drop(['LossCone'],axis=1).max().max()
    minLook = pitchInfo.drop(['LossCone'],axis=1).min().min()
    """
    binning each microburst
    """
    binNum = 10
    binList = []
    weightList = []
    names = pd.DataFrame(data={'Rate1':['det_1_Alpha','det_1_Beta'] ,'Rate2':['det_2_Alpha','det_2_Beta'],'Rate3':['det_3_Alpha','det_3_Beta'] ,'Rate4':['det_4_Alpha','det_4_Beta']})
    pitchAtPeaks = pitchInfo.loc[peakTimes]

    for col in cols:
        for i in range(len(countrates.index)):
            testPitches = pitchAtPeaks.iloc[i]
            alpha = testPitches[names[col][0]]
            beta = testPitches[names[col][1]]
            flux = countrates.iloc[i].loc[col]
            bins = np.linspace(alpha, beta, binNum)
            weights = [flux / binNum] * binNum
            binList.append(bins)
            weightList.append(weights)


    weightList = np.array(weightList).flatten()
    binList = np.array(binList).flatten()

    return binList,weightList,minLook,maxLook,pitchInfo,peakTimes,countrates

#%%
def plot_Garbage(pitchInfo,peakTimes,countrates):
    pitchAtPeaks = pitchInfo.loc[peakTimes]
    #start with one
    for i in range(len(countrates.index)):
        testPitches = pitchAtPeaks.iloc[i]
        currentRates = countrates.iloc[i]
        rateList =np.array( [currentRates['Rate1'],currentRates['Rate2'],currentRates['Rate3'],currentRates['Rate4']])
        rateList = rateList / np.max(rateList)

        fig,ax = plt.subplots()

        ax.hlines(y= rateList[0], xmin = testPitches['det_1_Alpha'],xmax = testPitches['det_1_Beta'],label='1')
        ax.hlines(y= rateList[1], xmin = testPitches['det_2_Alpha'],xmax = testPitches['det_2_Beta'],color='b',label='2')
        ax.hlines(y= rateList[2], xmin = testPitches['det_3_Alpha'],xmax = testPitches['det_3_Beta'],color='g',label='3')
        ax.hlines(y= rateList[3], xmin = testPitches['det_4_Alpha'],xmax = testPitches['det_4_Beta'],color='r',label='4')

        ax.hlines(y= rateList[0], xmin = -testPitches['det_1_Alpha'],xmax = -testPitches['det_1_Beta'],label='1')
        ax.hlines(y= rateList[1], xmin = -testPitches['det_2_Alpha'],xmax = -testPitches['det_2_Beta'],color='b',label='2')
        ax.hlines(y= rateList[2], xmin = -testPitches['det_3_Alpha'],xmax = -testPitches['det_3_Beta'],color='g',label='3')
        ax.hlines(y= rateList[3], xmin = -testPitches['det_4_Alpha'],xmax = -testPitches['det_4_Beta'],color='r',label='4')
        ax.hlines(y= .5 ,xmin= -testPitches['LossCone'],xmax = testPitches['LossCone'],linestyles='--',label = 'loss cone')
        ax.set_xlim(0,180)
        ax.legend()
        ax.set_xlabel("Local Pitch Angle, Degrees")
        ax.set_ylabel("Normalized Count Rate Integrated Over Full Width Half Max")

        plt.show()
# start = eventList[0][0]
# end = eventList[0][1]
# _,_,_,_,pitchInfo,peakTimes,countrates = microBins(start, end)
#plot_Garbage(pitchInfo, peakTimes, countrates)
#%%
binList = np.array([])
weightList = np.array([])
testing = np.array([])
j = 0
for event in eventList[0:12]:
    print(j)
    j+=1
    start = event[0]
    end = event[1]
    bins, weights,minLook,maxLook , _ , _  ,_ = microBins(start, end)
    testing = np.append(testing,minLook)
    testing = np.append(testing,maxLook)
    binList = np.append(binList,bins)
    weightList = np.append(weightList,weights)

minLook = testing.min()
maxLook = testing.max()
#%%
plt.hist(binList,bins=np.linspace(-180,180,100),density=True,weights=weightList )
plt.vlines(minLook,0,.03)
plt.vlines(maxLook, 0, .03)
plt.xlabel("Local Pitch Angle")
plt.title("Microbursts binned and weigthed by relative detector counts")
plt.show()
#%%
"""
compare look direction to Microbursts
"""
lookbins = np.load('/home/wyatt/Documents/SAMPEX/lookbins.npy')
ndat ,bdat,_= plt.hist(binList,bins=np.linspace(-180,180,100),density=True,weights=weightList ,label= 'Microbursts')
nlook,blook,_= plt.hist(lookbins,bins= np.linspace(-180,180,100) ,density=True,alpha = .5,label='Look Direction')
plt.title("Example Microburst/Look Direction Comparison 97/03/01")
plt.xlabel("Pitch Angle")
plt.legend()
plt.savefig("/home/wyatt/Documents/SAMPEX/histogram.png")
plt.show()
#%%
counts,bins = np.histogram(binList,bins=np.linspace(-180,180,100),density=True,weights=weightList)
lcount,lbin = np.histogram(lookbins,bins= np.linspace(-180,180,100) ,density=True)
altcount = np.nan_to_num(counts/lcount,nan=0.0)
plt.hist(bins[:-1], bins, weights=altcount,density=True)
plt.hist(binList,bins=np.linspace(-180,180,100),density=True,weights=weightList ,label= 'Microbursts',alpha=.4)
plt.show()
