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
# TODO: Clean up this nonsense so it's less scripty and more functional cause
        its a bitch to read
"""
radToDeg = 180.0 / m.pi

def fov_to_pitch(mesh):
    """
    for taking the square fov we have with respect to sampex coordinate planes
    and converting to local pitch angle
    """
    return np.arctan(np.sqrt(np.tan( mesh[0])**2 + np.tan(mesh[1])**2))

def eqPitch(eq_B, local_B,angle):
    pitch =  np.arcsin( np.sqrt(np.sin(angle)**2 * eq_B / local_B))
    return pitch

"""
Change this section to generate sampex look Distribution for a certain month
"""
month = '01'
eventList = []
hours = ['0' + str(i) if i<10 else str(i) for i in range(24)]
days = ['0' + str(i) if i<10 else str(i) for i in range(1,15)]
for day in days:
    for hour in hours:
        eventList.append([pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':00:00' , tz = 'utc'),
                          pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':29:59' , tz = 'utc')])
        eventList.append([pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':29:59' , tz = 'utc'),
                          pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':59:59' , tz = 'utc')])
outFile = '/home/wyatt/Documents/SAMPEX/generated_Data/look_Jan93.csv'
j=1
for event in eventList:
    print(j/48)
    j+=1
    start = event[0]
    end = event[1]

    workDir = '/home/wyatt/Documents/SAMPEX/'
    filename = "/home/wyatt/Documents/SAMPEX/data/HILThires/"
    if len(str(start.dayofyear))==1:
        day = '00'+str(start.dayofyear)
    elif len(str(start.dayofyear))==2:
        day = '0'+str(start.dayofyear)
    else:
        day = str(start.dayofyear)
    filename = filename + 'State1/hhrr1993' + day+'.txt'
    data = SAMPEXreb.quick_read(filename,start,end)
    data = data.drop(columns=['Rate5','Rate6'])
    times = data.index

    if not times.empty:
        data = findPitches(times,interpolate=False)
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

            #loss cone
            loss = data['Loss_Cone_1'].iloc[i] / radToDeg
            loss = eqPitch(eq_B, local_B, loss)

            pitch1 = fov_to_pitch(mesh1)
            pitch2 = fov_to_pitch(mesh2)
            pitch3 = fov_to_pitch(mesh3)
            pitch4 = fov_to_pitch(mesh4)

            pitch1 = eqPitch(eq_B, local_B, pitch1.flatten())
            pitch2 = eqPitch(eq_B, local_B, pitch2.flatten())
            pitch3 = eqPitch(eq_B, local_B, pitch3.flatten())
            pitch4 = eqPitch(eq_B, local_B, pitch4.flatten())

            pitches = np.append(pitch1, (pitch2,pitch3,pitch4))
            lossPitches =  (pitches - loss )
            lossPitchesFrac = (pitches - loss )/ loss

            df = pd.DataFrame(data={'pitch':pitches * radToDeg , 'loss':lossPitches*radToDeg , 'lossFrac':lossPitchesFrac})
            with open(outFile,'a') as f:
                df.to_csv(f,header=None)
