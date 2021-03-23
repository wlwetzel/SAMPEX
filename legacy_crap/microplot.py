import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from scipy import signal
sys.path.append('/home/wyatt/pyModules')
import SAMPEXreb
import datetime
import matplotlib.animation as animation
from Wavelet import wavelet,wave_signif,invertWave
#%matplotlib inline
# start  = pd.Timestamp('1992-10-04 00:00:00' , tz = 'utc') + pd.to_timedelta(13800,unit='s')
# end  = pd.Timestamp('1992-10-04 00:00:00' , tz = 'utc')  + pd.to_timedelta(14400 , unit='s')
# start  = pd.Timestamp('1997-10-01 10:35:00' , tz = 'utc')
# end  = pd.Timestamp('1997-10-01 10:45:00' , tz = 'utc')
# start  = pd.Timestamp('1997-10-01 02:25:00' , tz = 'utc')
# end  = pd.Timestamp('1997-10-01 02:45:00' , tz = 'utc')

# start  = pd.Timestamp('1997-01-07 01:42:00' , tz = 'utc')
# end  = pd.Timestamp('1997-01-07 01:44:00' , tz = 'utc')
# start  = pd.Timestamp('1997-01-09 07:07:00' , tz = 'utc')
# end  = pd.Timestamp('1997-01-09 07:12:00' , tz = 'utc')
start  = pd.Timestamp('1997-01-10 05:40:00' , tz = 'utc')
end  = pd.Timestamp('1997-01-10 05:47:00' , tz = 'utc')
start = pd.Timestamp('1997-03-05 00:00:00+00:00' ,tz='utc')
end = pd.Timestamp('1997-03-05 00:29:59+00:00' ,tz='utc')


# start  = pd.Timestamp('1997-03-01 00:17:00' , tz = 'utc')
# end  = pd.Timestamp('1997-03-01 00:40:00' , tz = 'utc')
# start  = pd.Timestamp('1997-03-01 01:00:00' , tz = 'utc')
# end  = pd.Timestamp('1997-03-01 01:10:00' , tz = 'utc')
# start  = pd.Timestamp('1997-03-01 01:15:00' , tz = 'utc')
# end  = pd.Timestamp('1997-03-01 01:25:00' , tz = 'utc')
# start  = pd.Timestamp('1997-03-01 01:45:00' , tz = 'utc')
# end  = pd.Timestamp('1997-03-01 02:25:00' , tz = 'utc')
# start  = pd.Timestamp('1997-03-01 08:25:00' , tz = 'utc')
# end  = pd.Timestamp('1997-03-01 09:25:00' , tz = 'utc')
eventList = []
hours = ['0' + str(i) if i<10 else str(i) for i in range(24)]
days = ['0' + str(i) if i<10 else str(i) for i in range(1,29)]
for day in days:
    for hour in hours:
        eventList.append([pd.Timestamp('1997-03-'+day+ ' ' +hour+ ':00:00' , tz = 'utc'),
                          pd.Timestamp('1997-03-'+day+ ' ' +hour+ ':29:59' , tz = 'utc')])
        eventList.append([pd.Timestamp('1997-03-'+day+ ' ' +hour+ ':29:59' , tz = 'utc'),
                          pd.Timestamp('1997-03-'+day+ ' ' +hour+ ':59:59' , tz = 'utc')])
#%%
for event in eventList[760:767]:
    filename = "/home/wyatt/Documents/SAMPEX/data/HILThires/"

    start = event[0]
    end = event[1]
    print(start,end)
    if len(str(start.dayofyear))==1:
        day = '00'+str(start.dayofyear)
    elif len(str(start.dayofyear))==2:
        day = '0'+str(start.dayofyear)
    else:
        day = str(start.dayofyear)
    print(day)
    filename = filename + 'State4/hhrr1997' + day+'.txt'

    data = SAMPEXreb.quick_read(filename,start,end)
    wavedat =  data['Rate1'].to_numpy()
    time = data.index.values

    fig,ax = plt.subplots()
    ax.semilogy(time,wavedat)
    plt.show()
