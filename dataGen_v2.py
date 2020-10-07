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
from pitchAngle_v3 import *

"""
This script is for analysis of identified microbursts
1) read in microbursts
2) find local pitch angle range
"""
filename = '/home/wyatt/Documents/SAMPEX/generated_Data/procData.csv'
radToDeg = 180.0 / m.pi
cols = ['Rate1', 'Rate2', 'Rate3', 'Rate4', 'B_Mag', 'B_X', 'B_Y', 'B_Z', 'Loss_Cone_1', 'Equator_B_Mag', 'A11', 'A21', 'A31', 'A12', 'A22', 'A32', 'A13', 'A23', 'A33']
data = pd.read_csv('/home/wyatt/Documents/SAMPEX/generated_Data/burst_Mar93.csv',header=None,index_col=0,names=cols)
length = len(data.index.values)
#rotate b to body fixed
BX = data['B_X']
BY = data['B_Y']
BZ = data['B_Z']
A11 = data['A11']
A12 = data['A12']
A13 = data['A13']
A21 = data['A21']
A22 = data['A22']
A23 = data['A23']
A31 = data['A31']
A32 = data['A32']
A33 = data['A33']

#x comp
BX_body = BX * A11 + BY * A12 + BZ * A13
#y comp
BY_body = BX * A21 + BY * A22 + BZ * A23
#z comp
BZ_body = BX * A31 + BY * A32 + BZ * A33
octant = pd.Series(data=BZ_body)

#angle to yz plane
#alpha and beta
# if we take away the abs, we'll get the right sign
yz_angle = radToDeg * np.arcsin(BX_body / data['B_Mag'])

#angle to xz plane
#assoc. with -34 to 34
xz_angle = radToDeg * np.arcsin(BY_body / data['B_Mag'])

# we need to set up the ranges
leftEdge_xz = (-34 - xz_angle) / radToDeg
rightEdge_xz =( 34 - xz_angle) /radToDeg

#note, I think I calculated this wrong previously whoops
#det1 goes from -9.57 to 34
#det2 -18.636 o 26.834
#det3 -26.834 to 18.636
#det4 -34 to 9.57
leftEdge_1_yz = (-9.57 - yz_angle) / radToDeg
rightEdge_1_yz =(34 - yz_angle) / radToDeg

leftEdge_2_yz = (-18.636 - yz_angle) / radToDeg
rightEdge_2_yz = (26.834 - yz_angle) /radToDeg

leftEdge_3_yz = (-26.834 - yz_angle) / radToDeg
rightEdge_3_yz =( 18.636 - yz_angle) /radToDeg

leftEdge_4_yz = (-34 - yz_angle) / radToDeg
rightEdge_4_yz = (9.57 - yz_angle) /radToDeg

def fov_to_pitch(mesh):
    """
    for taking the square fov we have with respect to sampex coordinate planes
    and converting to local pitch angle
    """
    return np.arctan(np.sqrt(np.tan( mesh[0])**2 + np.tan(mesh[1])**2))

def eqPitch(eq_B, local_B,angle):
    pitch =  np.arcsin( np.sqrt(np.sin(angle)**2 * eq_B / local_B))
    return pitch


# i think the next bit, converting to pitch angle, will be easiest in a for loop
for i in range(length):
    num = 10
    mesh1 = np.meshgrid(np.linspace(leftEdge_1_yz[i] , rightEdge_1_yz[i],num) ,
                        np.linspace(leftEdge_xz[i] , rightEdge_xz[i] , num))
    mesh2 = np.meshgrid(np.linspace(leftEdge_2_yz[i] , rightEdge_2_yz[i],num) ,
                        np.linspace(leftEdge_xz[i] , rightEdge_xz[i] , num))
    mesh3 = np.meshgrid(np.linspace(leftEdge_3_yz[i] , rightEdge_3_yz[i],num) ,
                        np.linspace(leftEdge_xz[i] , rightEdge_xz[i] , num))
    mesh4 = np.meshgrid(np.linspace(leftEdge_4_yz[i] , rightEdge_4_yz[i],num) ,
                        np.linspace(leftEdge_xz[i], rightEdge_xz[i] , num))
    #convert to pitchangle
    eq_B = data['Equator_B_Mag'].iloc[i]
    local_B = data['B_Mag'].iloc[i]

    pitch1 = fov_to_pitch(mesh1)
    pitch2 = fov_to_pitch(mesh2)
    pitch3 = fov_to_pitch(mesh3)
    pitch4 = fov_to_pitch(mesh4)

    #loss cone
    loss = data['Loss_Cone_2'].iloc[i] / radToDeg
    loss = eqPitch(eq_B, local_B, loss)

    pitch1 = eqPitch(eq_B, local_B, pitch1.flatten())
    pitch2 = eqPitch(eq_B, local_B, pitch2.flatten())
    pitch3 = eqPitch(eq_B, local_B, pitch3.flatten())
    pitch4 = eqPitch(eq_B, local_B, pitch4.flatten())

    # now we can just assign each element of the pitches a hundredth of the flux
    flux1 = np.array([data['Rate1'].iloc[i] / num**2 ] * num**2)
    flux2 = np.array([data['Rate2'].iloc[i] / num**2 ] * num**2)
    flux3 = np.array([data['Rate3'].iloc[i] / num**2 ] * num**2)
    flux4 = np.array([data['Rate4'].iloc[i] / num**2 ] * num**2)

    pitches = np.append(pitch1, (pitch2,pitch3,pitch4))
    fluxes = np.append(flux1, (flux2,flux3,flux4))
    lossPitches = pitches - loss

    df = pd.DataFrame(data={'pitch':pitches * radToDeg , 'flux':fluxes, 'loss' : lossPitches*radToDeg})
    with open(filename,'a') as f:
        df.to_csv(f,header=None)



#what quadrant are we in
# for i in range(length):
#     if BX_body.iloc[i]>0:
#         if BY_body.iloc[i]>0:
#             if BZ_body.iloc[i]>0:
#                     #+++ = 1
#                     octant.iloc[i]=1
#             else:
#                 #++-=5
#                 octant.iloc[i]=5
#         else:
#             if BZ_body.iloc[i]>0:
#                     #+-+ = 4
#                     octant.iloc[i]=4
#             else:
#                 #+-- = 7
#                 octant.iloc[i]=7
#     else:
#         if BY_body.iloc[i]>0:
#             if BZ_body.iloc[i]>0:
#                     #-++ = 2
#                     octant.iloc[i]=2
#             else:
#                 #-+-= 6
#                 octant.iloc[i]=6
#         else:
#             if BZ_body.iloc[i]>0:
#                     #--+ = 3
#                     octant.iloc[i]=3
#             else:
#                 #--- = 8
#                 octant.iloc[i]=8
