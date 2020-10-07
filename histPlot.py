import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
#readin data
datFile = ['hist_Jan93.csv','hist_Feb93.csv']
lookFile = ['look_Jan93.csv','look_Feb93.csv']

dat = pd.DataFrame()
lookDir = pd.DataFrame()

for i in range(2):
    tempdat= pd.read_csv('/home/wyatt/Documents/SAMPEX/'+datFile[i],header=None,index_col=0,names=['data','weights'])
    templookDir = pd.read_csv('/home/wyatt/Documents/SAMPEX/'+lookFile[i],index_col=0,names=['look'])
    dat = dat.append(tempdat,ignore_index=True)
    lookDir = lookDir.append(templookDir,ignore_index=True)
#%%
binNum = 80
fact = 180/3.14159
fig,axs = plt.subplots(2)
dat_hist,bins,_  = axs[0].hist(fact*dat['data'],bins=np.linspace(0,8,binNum),weights=dat['weights'],density=True,label= 'Microburst Distribution')
l_hist,lbins ,_= axs[0].hist(fact * lookDir['look'],bins=np.linspace(0,8,binNum),density=True,alpha=.4,label='Look Direction')

axs[0].set_title('Microburst Distribution Weighted by Amplitude')
axs[0].legend()
axs[1].set_xlabel('Equatorial Pitch Angle (Degrees)')

altcount = np.nan_to_num(dat_hist/l_hist,nan=0.0)

axs[1].set_title('Microburst Distribution Weighted by Look Distribution')
axs[1].hist(bins[:-1], bins, weights=altcount,density=True)

plt.tight_layout()
plt.savefig('/home/wyatt/Documents/SAMPEX/correctedPitch.pdf')

plt.show()
