import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as m

df = pd.read_csv('/home/wyatt/Documents/SAMPEX/generated_Data/procData2.csv',header=None,index_col=0,names=['pitch','flux','loss'])

fig,ax = plt.subplots()
bin,edge,_ =ax.hist(df['pitch'],bins=np.linspace(0,5,20),weights=df['flux'],density=True)
ax.set_xlabel('Equatorial Pitch Angle')
ax.set_title('Microburst Pitch Angle Distribution')
plt.show()

fig,ax = plt.subplots()
lossCone = (df['pitch']-df['loss'] )/df['loss']

ax.hist(lossCone,bins=np.linspace(-1,1,20),weights=df['flux'],density=True )
ax.text(-.8,.5,'Into Loss Cone')
ax.text(.5,.5,'Out of Loss Cone')
ax.set_title('Pitch Angle Compared to Loss Cone')
ax.set_xlabel('Distance from Loss Cone as Fraction of Loss Cone Size')
ax.axvline(color='black')
plt.show()
