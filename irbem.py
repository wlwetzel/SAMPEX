import matplotlib.pyplot as plt
import sys
sys.path.append('/home/wyatt/Downloads/irbem-code-r620-trunk/python')
import IRBEM
import pandas as pd
from spacepy.time import Ticktock
import spacepy.coordinates as spc
import spacepy.irbempy as irb
import SAMP_Data
import numpy as np
model = 'T87SHORT'
Re = 6371 #km
col_names = ['Year', 'Day-of-year', 'Sec_of_day',
              'Sec_of_day_psset', 'Flag_rstime',
              'Orbit_Number', 'GEO_Radius', 'GEO_Long',
              'GEO_Lat', 'Altitude', 'GEI_X', 'GEI_Y',
              'GEI_Z', 'GEI_VX', 'GEI_VY', 'GEI_VZ',
              'ECD_Radius', 'ECD_Long', 'ECD_Lat',
              'ECD_MLT', 'L_Shell', 'B_Mag', 'MLT',
              'Invariant_Lat', 'B_X', 'B_Y', 'B_Z',
              'B_R', 'B_Theta', 'B_Phi', 'Declination',
              'Dip', 'Magnetic_Radius', 'Magnetic_Lat',
              'Loss_Cone_1', 'Loss_Cone_2',
              'Dipole_Moment_X', 'Dipole_Moment_Y',
              'Dipole_Moment_Z', 'Dipole_Disp_X',
              'Dipole_Disp_Y', 'Dipole_Disp_Z',
              'Mirror_Alt', 'Mirror_Long',
              'Mirror_Lat', 'Equator_B_Mag',
              'Equator_Alt', 'Equator_Long',
              'Equator_Lat', 'North100km_B_Mag',
              'North100km_Alt',
              'North100km_Long', 'North100km_Lat',
              'South100km_B_Mag', 'South100km_Alt',
              'South100km_Long', 'South100km_Lat',
              'Vertical_Cutoff', 'SAA_Flag',
              'A11', 'A21', 'A31', 'A12', 'A22',
              'A32', 'A13', 'A23', 'A33', 'Pitch',
              'Zenith', 'Azimuth', 'Att_Flag']

dataPath = '/home/wyatt/Documents/SAMPEX/OrbitData/PSSet_6sec_1993040_1993066.txt'
month = '02'
eventList = []
hours = ['0' + str(i) if i<10 else str(i) for i in range(24)]
days = ['0' + str(i) if i<10 else str(i) for i in range(1,20)]
for day in days:
    for hour in hours:
        eventList.append([pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':00:00' , tz = 'utc'),
                          pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':29:59' , tz = 'utc')])
        eventList.append([pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':29:59' , tz = 'utc'),
                          pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':59:59' , tz = 'utc')])
start=eventList[-30][0]
end=eventList[-30][1]
ind = 10
orbitObj = SAMP_Data.OrbitData(date=start)
orbit = orbitObj.read_time_range(start,end,parameters=['B_X','B_Y','B_Z','B_Mag'
,'Equator_B_Mag','GEI_X','GEI_Y','GEI_Z','GEO_Radius','GEO_Lat','GEO_Long','B_R'
,'B_Theta','B_Phi','Loss_Cone_1'])
times = orbit.index
orbit = orbit.iloc[ind]
time = times[ind]
y =[orbit['GEO_Radius'] / Re , orbit['GEO_Lat'] , orbit['GEO_Long'] ]
"""
spacepy doesnt fuckin work so I gotta learn the other one fuck
"""
time = time.to_pydatetime()
model = IRBEM.MagFields(options=[0,0,0,0,0],verbose=False,sysaxes=8)
pos = {}
pos['x1'] = y[0]
pos['x2'] = y[1]
pos['x3'] = y[2]
pos['dateTime'] = time

foot = model.find_foot_point(pos,{'Kp':0},100,0)
footPos = {}
footPos['x1'] = foot['XFOOT'][0] / Re + 1
footPos['x2'] = foot['XFOOT'][1]
footPos['x3'] = foot['XFOOT'][2]
footPos['dateTime'] = time

def to_equatorial(position,pitch):
    lstar = model.make_lstar(position,{'Kp':0})
    bmin = lstar['bmin']
    blocal = lstar['blocal']
    eq_pitch = np.arcsin(np.sqrt(np.sin(np.deg2rad(pitch))**2 * bmin[0] / blocal[0]))
    return np.rad2deg(eq_pitch)



"""
spacepy version
"""
sublist = eventList[0:2]
start=sublist[0][0]
end = sublist[0][1]
orbitObj = SAMP_Data.OrbitData(date=start)
orbit = orbitObj.read_time_range(start,end,parameters=['B_X','B_Y','B_Z','B_Mag'
,'Equator_B_Mag','GEI_X','GEI_Y','GEI_Z','GEO_Radius','GEO_Lat','GEO_Long','B_R'
,'B_Theta','B_Phi','Loss_Cone_1'])
time = orbit.index[0]
orbit = orbit.iloc[0]
y = [orbit['GEI_X']/Re,orbit['GEI_Y']/Re,orbit['GEI_Z']/Re]
coord = spc.Coords(y,'GEI','car')
time = Ticktock(time)

print(irb.find_magequator(time,coord,extMag='T89'))
print(irb.find_footpoint(time,coord,extMag='T89'))
