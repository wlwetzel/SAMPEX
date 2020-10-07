import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as m

radToDeg = 180.0 / m.pi

df = pd.read_csv('/home/wyatt/Documents/SAMPEX/generated_Data/procData.csv',header=None,index_col=0,names=['pitch','flux','loss'])
look = pd.read_csv('/home/wyatt/Documents/SAMPEX/generated_Data/look_Jan93.csv',header=None,index_col=0,names=['pitch','loss','lossFrac'])

fig,axs = plt.subplots()
dataPitchHist , dataPitchBins,_ = axs.hist(df['pitch'],bins=np.linspace(0,8,50),weights = df['flux'],density=True,label='Microburst Distribution')
lookPitchHist , lookPitchBins,_ = axs.hist(look['pitch'],bins=np.linspace(0,8,50),alpha = .5,density=True,label='Look Direction')
axs.set_title('Equatorial Pitch Angle Distribution of Microbursts')
axs.legend()
axs.set_xlabel('Equatorial Pitch Angle (Degrees)')
#lossdf = df['loss'] / (df['pitch'] - df['loss'])

#dataLossHist , dataLossBins,_ =  axs[1,0].hist(lossdf,bins=np.linspace(-1.1,1.1,100),weights = df['flux'],density=True)
#lookLossHist , lookLossBins,_ = axs[1,0].hist(look['lossFrac'] / radToDeg,bins=np.linspace(-1.1,1.1,100),alpha=.5,density=True)
plt.savefig('/home/wyatt/Documents/SAMPEX/Figures/eqPitch.png')
plt.show()

#fig,axs = plt.subplots()
#pitchCount = np.nan_to_num(dataPitchHist/lookPitchHist,nan=0.0)
#lossCount =  np.nan_to_num(dataLossHist/lookLossHist,nan=0.0)

#axs.hist(dataPitchBins[:-1], dataPitchBins, weights=pitchCount,density=True)
#axs[1,1].hist(dataLossBins[:-1], dataLossBins, weights=lossCount,density=True)

#axs[1,0].axvline(color='Black')
#axs[1,1].axvline(color='Black')
#axs.set_title('Pitch Angle Dist. Weighted by Look Direction')
#axs[1,1].set_title('Loss Cone Plot Weighted by look direction')
#axs[1,0].text(-4,.25,'Into Loss Cone')
#axs[1,0].text(1,.25,'Out Of Loss Cone')
#axs.set_xlabel('Equatorial Pitch Angle, Degrees')
#axs[1,0].set_title('Microbursts Into and Out of Loss Cone')
#plt.savefig('/home/wyatt/Documents/SAMPEX/Figures/eqPitchCorr.png')
# fg,ax = plt.subplots()
# ax.hist(look['lossFrac'],bins=np.linspace(-1,1,200),density=True)
#plt.show()
