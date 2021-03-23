import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from scipy import signal
sys.path.append('/home/wyatt/pyModules')
import SAMPEXreb
#import Wavelet
import datetime
import matplotlib.animation as animation
from Wavelet import wavelet,wave_signif,invertWave
workDir = '/home/wyatt/Documents/SAMPEX/'
filename = "/home/wyatt/Documents/SAMPEX/data/HILThires/"
new = 0
analysis = 0
analysis2 = 0
filter = 0
jan10 = 1
if jan10:
    start  = pd.Timestamp('1997-01-10 05:40:00' , tz = 'utc')
    end  = pd.Timestamp('1997-01-10 05:47:00' , tz = 'utc')
    start  = pd.Timestamp('1997-01-10 09:35:00' , tz = 'utc')
    end  = pd.Timestamp('1997-01-10 10:05:00' , tz = 'utc')

    if len(str(start.dayofyear))==1:
        day = '00'+str(start.dayofyear)
    elif len(str(start.dayofyear))==2:
        day = '0'+str(start.dayofyear)

    filename = filename + 'State4/hhrr1997' + day+'.txt'
    data = SAMPEXreb.quick_read(filename,start,end)
    wavedat =  data['Rate1'].to_numpy()
    maxScale = 5 #scale in seconds, above which we throw out
    fig, axs = plt.subplots(2,2)
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[1,0]
    ax4 = axs[1,1]
    ax1.set_title(str(start.asm8))
    time = data.index.values
    ax1.semilogy(time,wavedat,'k:')
    variance = np.std(wavedat, ddof=1) ** 2
    n = len(wavedat)
    dt = .1
    dj = .1
    s0 = 2*dt
    j1 = int(15 / dj)
    mother = 'PAUL'
    pad = 1
    lag1 = .1

    wave, period, scale, coi = wavelet(wavedat, dt, pad, dj, s0, j1, mother)
    power = np.abs(wave)**2

    signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale,
                                            lag1=lag1, mother=mother)
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand signif --> (J+1)x(N) array
    sig95 = power / sig95  # where ratio > 1, power is significant
    mask = sig95>1
    denoisedWav = mask * wave
    dePower = np.abs(denoisedWav)**2
    ax2.contour(time,period,np.log2(dePower),50)

    """
    removing large scale stuff
    """
    mask = scale<maxScale
    mask = mask[:,np.newaxis].dot(np.ones(n)[np.newaxis,:])
    denoisedWav *=mask

    dePower = np.abs(denoisedWav)**2

    newdat = invertWave(mother, scale, dt, dj, denoisedWav)
    ax3.contour(time,period,np.log2(dePower), 50)
    ax3.set_yscale('log')
    ax3.invert_yaxis()
    ax2.set_yscale("log")
    ax2.invert_yaxis()
    ax4.plot(time,newdat)
    ax1.set_xlabel("TIme,(UTC)")
    ax2.set_xlabel("TIme,(UTC)")
    ax3.set_xlabel("TIme,(UTC)")
    ax4.set_xlabel("TIme,(UTC)")
    ax2.set_ylabel("Scale (s)")
    ax3.set_ylabel("Scale (s)")
    ax3.set_title('Large Scales Set to Zero')
    ax4.set_title("Reconstructed Signal")
    ax2.set_title("Wavelet Power "+mother)
    plt.tight_layout()
    plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax4.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.show()

if new:
        """
        read in data and plot
        """
        start  = pd.Timestamp('1992-10-04 00:00:00' , tz = 'utc') + pd.to_timedelta(13600,unit='s')
        end  = pd.Timestamp('1992-10-04 00:00:00' , tz = 'utc')  + pd.to_timedelta(14600 , unit='s')
        filename = filename + 'State1/hhrr1992' + str(start.dayofyear)+'.txt'


        data = SAMPEXreb.quick_read(filename,start,end)
        wavedat =  data['Rate1'].to_numpy()
        fig, axs = plt.subplots(2,2)
        ax1 = axs[0,0]
        ax2 = axs[0,1]
        ax3 = axs[1,0]
        ax4 = axs[1,1]
        ax1.set_title(str(start.asm8))
        time = data.index.values
        ax1.plot(time,wavedat,'k:')
        variance = np.std(wavedat, ddof=1) ** 2
        n = len(wavedat)
        dt = .1
        dj = .1
        s0 = 2*dt
        j1 = int(15 / dj)
        mother = 'PAUL'
        pad = 1
        lag1 = .1

        wave, period, scale, coi = wavelet(wavedat, dt, pad, dj, s0, j1, mother)
        power = np.abs(wave)**2

        signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale,
                                                lag1=lag1, mother=mother)
        sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand signif --> (J+1)x(N) array
        sig95 = power / sig95  # where ratio > 1, power is significant
        mask = sig95>1
        denoisedWav = mask * wave
        dePower = np.abs(denoisedWav)**2
        ax2.contour(time,period,np.log2(dePower),50)
        ax2.set_yscale("log")
        #inverting data
        denoisedWav[80:,:] =0
        dePower = np.abs(denoisedWav)**2
        ax3.contour(time,period,np.log2(dePower), 50)
        ax3.set_yscale('log')
        ax3.invert_yaxis()
        ax2.invert_yaxis()
        newdat = invertWave(mother, scale, dt, dj, denoisedWav)
        ax4.plot(time,newdat)
        ax1.set_xlabel("TIme,(UTC)")
        ax2.set_xlabel("TIme,(UTC)")
        ax3.set_xlabel("TIme,(UTC)")
        ax4.set_xlabel("TIme,(UTC)")
        ax2.set_ylabel("Scale (s)")
        ax3.set_ylabel("Scale (s)")
        ax3.set_title('Large Scales Set to Zero')
        ax4.set_title("Reconstructed Signal")
        ax2.set_title("Wavelet Power (PAUL)")
        plt.tight_layout()
        plt.savefig(workDir + 'NoSpinPAUL.png')
        plt.show()
if analysis:
    start  = pd.Timestamp('1992-10-04 00:00:00' , tz = 'utc') + pd.to_timedelta(13600,unit='s')
    end  = pd.Timestamp('1992-10-04 00:00:00' , tz = 'utc')  + pd.to_timedelta(14600 , unit='s')
    filename = filename + 'State1/hhrr1992' + str(start.dayofyear)+'.txt'


    data = SAMPEXreb.quick_read(filename,start,end)
    wavedat =  data['Rate1'].to_numpy()
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4)
    ax1.set_title(str(start.asm8))
    times = data.index.values
    ax1.plot(times,wavedat,'k:')

    """
    calculate lag1 autocorellation
    """
    var = np.var(wavedat)
    mean = np.mean(wavedat)
    lag1 = pd.Series(wavedat[:4000]).autocorr(lag=1)
    #ax2.plot(wavedat[:4000],'k-')
    print(lag1)

    """
    wavelet analysis
    """
    dt = .1
    s0 = 2*dt
    dj = .1
    num_octaves = 15
    wavObj = Wavelet.WaveletObject(wavedat, dt, dj=dj,J=int(num_octaves/dj))
    wav = wavObj.waveletTransform()
    power = np.abs(wav)**2
    periods = wavObj.fourierPeriods()


    """
    denoise
    """
    denoisedWav = wavObj.denoise()
    denoisedDat = wavObj.invWaveletTransform()

    # #set a lot of entries to zero?
    #denoisedWav[80:,:]=0
    wavObj.changeWavelet(denoisedWav)
    transDat = wavObj.invWaveletTransform()
    denoisedPower = np.abs(denoisedWav)**2

    """
    plotting
    """

    ax2.contour(times,periods,np.log2(power),100)
    ax3.contour(times,periods,np.log2(denoisedPower),20)
    #ax4.plot(times,transDat,'k-')
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    ax1.set_ylabel("counts")
    ax4.set_ylabel("counts")
    ax3.set_ylabel("scale (s) ")
    ax2.set_ylabel("scale (s) ")
    plt.tight_layout()
    #plt.savefig("/home/wyatt/Documents/SAMPEX/oct92wav.png",format='png')
    plt.show()
if filter:
    start  = pd.Timestamp('1997-10-01 10:25:00' , tz = 'utc')
    end  = pd.Timestamp('1997-10-01 10:45:00' , tz = 'utc')

    filename = filename + 'State4/hhrr1997' + str(start.dayofyear)+'.txt'
    data = SAMPEXreb.quick_read(filename,start,end)
    wavedat =  data['Rate1'].to_numpy()
    times = data.index.values

    T = 60.0
    fcrit = 1/T * 2
    samFreq = 1/.1
    F_data = np.fft.fft(wavedat)
    freqs = np.fft.fftfreq(len(F_data),.1)
    sos = signal.butter(10, [fcrit*.8 , fcrit*1.2], btype='bandstop', fs=samFreq, output='sos')
    filtered = signal.sosfilt(sos, wavedat)

    fig,axs = plt.subplots(3)

    axs[0].plot(times,wavedat)
    axs[1].plot(times,filtered)
    axs[2].semilogy(freqs,np.abs(F_data))
    axs[2].axvline(fcrit,color = 'red')
    axs[2].axvline(fcrit*1.2,color = 'red')
    axs[2].axvline(fcrit*.8,color = 'red')

    axs[2].set_xlim(-.2,.2)
    plt.show()
if analysis2:
    start  = pd.Timestamp('1997-10-01 10:25:00' , tz = 'utc')
    end  = pd.Timestamp('1997-10-01 10:45:00' , tz = 'utc')
    # start  = pd.Timestamp('1997-12-03 14:23:20' , tz = 'utc')
    # end  = pd.Timestamp('1997-12-03 14:24:00' , tz = 'utc')
    # start  = pd.Timestamp('1997-10-01 02:25:00' , tz = 'utc')
    # end  = pd.Timestamp('1997-10-01 02:45:00' , tz = 'utc')



    filename = filename + 'State4/hhrr1997' + str(start.dayofyear)+'.txt'

    data = SAMPEXreb.quick_read(filename,start,end)
    wavedat =  data['Rate1'].to_numpy()
    fig, axs = plt.subplots(2,2)
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[1,0]
    ax4 = axs[1,1]

    ax1.set_title(str(start.asm8))
    time = data.index.values
    ax1.semilogy(time,wavedat,'k:')

    """
    Filtering
    """
    # T = 60.0
    # fcrit = 1/T * 2
    # samFreq = 1/.1
    # F_data = np.fft.fft(wavedat)
    # freqs = np.fft.fftfreq(len(F_data),.1)
    # sos = signal.butter(10, [fcrit*.9 , fcrit*1.1], btype='bandstop', fs=samFreq, output='sos')
    # filtered = signal.sosfilt(sos, wavedat)
    # ax1.plot(time,filtered)
    # wavedat = filtered

    """
    wavelet analysis
    """
    variance = np.std(wavedat, ddof=1) ** 2
    n = len(wavedat)
    dt = .1
    dj = .1
    s0 = 2*dt
    j1 = int(12 / dj)
    pad = 1
    mother = 'PAUL'
    lag1 = .1


    wave, period, scale, coi = wavelet(wavedat, dt, pad, dj, s0, j1, mother)
    power = np.abs(wave)**2

    signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale,
                                            lag1=lag1, mother=mother)
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand signif --> (J+1)x(N) array
    sig95 = power / sig95  # where ratio > 1, power is significant
    mask = sig95>1
    denoisedWav = mask * wave
    dePower = np.abs(denoisedWav)**2
    ax2.contour(time,period,np.log2(dePower),50,cmap= 'Greys')
    ax2.set_yscale("log")
    denoisedWav[80:,:] =0
    dePower = np.abs(denoisedWav)**2
    ax3.contour(time,period,np.log2(dePower), 50,cmap='binary')
    ax3.set_yscale('log')
    ax3.invert_yaxis()
    ax2.invert_yaxis()
    newdat = invertWave(mother, scale, dt, dj, denoisedWav)
    ax4.plot(time,newdat)
    plt.tight_layout()

    plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax4.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax1.set_xlabel("TIme,(UTC)")
    ax2.set_xlabel("TIme,(UTC)")
    ax3.set_xlabel("TIme,(UTC)")
    ax4.set_xlabel("TIme,(UTC)")
    ax2.set_ylabel("Scale (s)")
    ax3.set_ylabel("Scale (s)")
    ax3.set_title('Large Scales Set to Zero')
    ax4.set_title("Reconstructed Signal")
    ax2.set_title("Wavelet Power "+ mother)
    plt.tight_layout()
    plt.show()
