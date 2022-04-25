from spacepy.time import Ticktock
import spacepy.coordinates as spc
import spacepy.irbempy as irb
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import pandas as pd
import sys
import dask.dataframe as dd
import math as m
sys.path.append('/home/wyatt/py_Modules')
from os import listdir
from os.path import isfile, join
import os
from glob import glob

class SampexDateConverter:
    """
    Used to convert SAMPEX file dates to Pandas
    datetime objects. Must have a persistent year
    since that info is only in the filename
    """
    def __init__(self, date):
        self.date = date

    def sec_to_date(self, secs):
        return pd.to_datetime(secs, unit='s',
                              origin=self.date,
                              utc=True)

def datetime_from_filename(filename):
    # Pull the date from the filename
    name = os.path.basename(filename)

    year = pd.Timestamp(name[4:8])

    days_off = int(name[8:11]) - 1  # Days after Jan 1
    days = pd.DateOffset(days=days_off)

    return year + days

def datetime_from_doy(year,doy):
    year = pd.Timestamp(str(year))
    days = pd.DateOffset(days = (doy-1))
    return year+ days

def quick_read(filename,start,end):
    startDate = start.tz_localize(None)
    endDate = end.tz_localize(None)
    date = datetime_from_filename(filename)
    time_converter = SampexDateConverter(date)
    totalRows = m.ceil((endDate-startDate).total_seconds() / .1)
    rowsToSkip = m.ceil((startDate-date).total_seconds() / .1)
    cols = ['Time','Rate1','Rate2','Rate3','Rate4','Rate5','Rate6']
    try:

        data = pd.read_csv(filename, sep=' ', header=0,
                           date_parser=time_converter.sec_to_date,
                           names=cols,
                           parse_dates=['Time'], index_col=['Time'],
                           skiprows=rowsToSkip,
                           nrows=totalRows,
                           dtype={"Time": np.float128,
                                  "Rate1": np.uint16,
                                  "Rate2": np.uint16,
                                  "Rate3": np.uint16,
                                  "Rate4": np.uint16,
                                  "Rate5": np.uint16,
                                  "Rate6": np.uint16})
        if not data.empty:
            #there's a lot of missing data, so trying to acquire data a half
            #hour at a time doesnt actually work, it grabs 30*60/.1 data points
            #thus we hit the end of the file too soon, so as a stupid solution
            #i just return the empty data frame and all function calls
            #downstream just skip through when they get it
            data['Time'] = data.index
            data.index = data.index.tz_convert('UTC')

    except FileNotFoundError:
        print(filename + ' not found')
        return None

    return data

class HiltData:
    _data_dir = '/home/wyatt/Documents/SAMPEX/SAMPEX_Data/HILThires/State1'
    # _data_dir = "/media/wyatt/64A5-F009/SAMPEX_Data/HILThires/State1"

    def __init__(self,filename=None,date=None):
        if filename is None and date is None:
            raise TypeError("Either a time or a file is needed")
        elif filename is None:
            self._init_date(date)
        elif date is None:
            self._init_filename(filename)
        else:
            self._init_filename(filename)
            if not self.date_in(date):
                raise TypeError("Two input arguments provided, "
                                "but don't agree")

    def _init_filename(self, filename):
        """
        Initializes with a given file.
        :param filename: HILT data file
        TODO: doesnt work yet
        """

        self.filename = filename
        start_year = pd.to_datetime(filename[-19:-15], utc=True)
        start_days = pd.to_timedelta(int(filename[-15:-12])-1, 'D')
        self.start = start_year + start_days

        end_year = pd.to_datetime(filename[-11:-7], utc=True)
        end_days = pd.to_timedelta(int(filename[-7:-4]), 'D')
        self.end = end_year + end_days

    def _init_date(self, date):
        """
        Initializes to use hilt data for a specific time
        :param date: str, YYYYDDD
        """
        year = pd.Timestamp(date[0:4])
        days_off = int(date[4:])-1
        days = pd.DateOffset(days = days_off)
        self.date = year+days

        files = []
        # start_dir = os.getcwd()
        start_dir = '/home/wyatt/Documents/SAMPEX/SAMPEX_Data'
        # start_dir = "/media/wyatt/64A5-F009/SAMPEX_Data"
        pattern   = "*.txt"

        for dir,_,_ in os.walk(start_dir):
            files.extend(glob(os.path.join(dir,pattern)))

        # files = [f for f in listdir(self._data_dir)
        #          if isfile(join(self._data_dir, f))]
        found_file = False

        for data_filename in files:

            # Checks each file in in the catlog to see if it's valid.
            if date in data_filename:
                self.filename = data_filename
                self.state = data_filename[-17] #different states
                found_file = True
                break

        if not found_file:
            raise ValueError("File for ", date, " could not be found")

    def read(self,start,end):
        """
        :param start: int, start second of day
        :param end: int, end second of day
        or
        send start and end as None
        """
        time_converter = SampexDateConverter(self.date)
        cols = ['Time','Rate1','Rate2','Rate3','Rate4','Rate5','Rate6']
        try:
            data = pd.read_csv(self.filename, sep=' ', header=0,
                               date_parser=time_converter.sec_to_date,
                               names=cols,
                               parse_dates=['Time'], index_col=['Time'],
                               dtype={"Time": np.float128,
                                      "Rate1": np.uint16,
                                      "Rate2": np.uint16,
                                      "Rate3": np.uint16,
                                      "Rate4": np.uint16,
                                      "Rate5": np.uint16,
                                      "Rate6": np.uint16})
        except FileNotFoundError:
            return None
            print(filename + ' not found')
        if self.state=='2':
            """
            state 2 rate1:time to time +20ms rate2:time+20 to time+40
                    rate3:time+40 to time+60 rate5:time+80 to time+100
                    rate6:time+60 to time+80
            """
            times = data.index
            delta = pd.Timedelta(20,unit="milli")
            rate1 = pd.DataFrame({"Counts":data['Rate1'].to_numpy()},index=times)
            rate2 = pd.DataFrame({"Counts":data['Rate2'].to_numpy()},index=times+delta)
            rate3 = pd.DataFrame({"Counts":data['Rate3'].to_numpy()},index=times+2*delta)
            rate5 = pd.DataFrame({"Counts":data['Rate5'].to_numpy()},index=times+4*delta)
            rate6 = pd.DataFrame({"Counts":data['Rate6'].to_numpy()},index=times+3*delta)
            data = pd.concat([rate1,rate2,rate3,rate5,rate6])
            data = data.sort_index()

        if self.state=='4':
            """
            state 4 rate1:time to time +20ms rate2:time+20 to time+40
                    rate3:time+40 to time+60 rate4:time+60 to time+80
                    rate6:time+80 to time+100
            """
            times = data.index
            delta = pd.Timedelta(20,unit="milli")
            rate1 = pd.DataFrame({"Counts":data['Rate1'].to_numpy()},index=times)
            rate2 = pd.DataFrame({"Counts":data['Rate2'].to_numpy()},index=times+delta)
            rate3 = pd.DataFrame({"Counts":data['Rate3'].to_numpy()},index=times+2*delta)
            rate4 = pd.DataFrame({"Counts":data['Rate4'].to_numpy()},index=times+3*delta)
            rate6 = pd.DataFrame({"Counts":data['Rate6'].to_numpy()},index=times+4*delta)
            data = pd.concat([rate1,rate2,rate3,rate4,rate6])
            data = data.sort_index()



        if type(start)==int:
            start_time = self.date + pd.Timedelta(start,unit='seconds')
            end_time = self.date + pd.Timedelta(end,unit='seconds')
            return data[start_time.tz_localize('UTC'):end_time.tz_localize('UTC')]
        else:
            return data

    def quick_read(self,start,end):
        """
        :param start: int, start second of day
        :param end: int, end second of day
        """
        start_time = self.date + pd.Timedelta(start,unit='seconds')
        end_time = self.date + pd.Timedelta(end,unit='seconds')
        time_converter = SampexDateConverter(self.date)
        totalRows = m.ceil((end_time-start_time).total_seconds() / .1)
        rowsToSkip = m.ceil((start_time-self.date).total_seconds() / .1)
        cols = ['Time','Rate1','Rate2','Rate3','Rate4','Rate5','Rate6']
        try:
            data = pd.read_csv(self.filename, sep=' ', header=0,
                               date_parser=time_converter.sec_to_date,
                               names=cols,
                               parse_dates=['Time'], index_col=['Time'],
                               skiprows=rowsToSkip,
                               nrows=totalRows,
                               dtype={"Time": np.float128,
                                      "Rate1": np.uint16,
                                      "Rate2": np.uint16,
                                      "Rate3": np.uint16,
                                      "Rate4": np.uint16,
                                      "Rate5": np.uint16,
                                      "Rate6": np.uint16})
            # if not data.empty:
            #     #there's a lot of missing data, so trying to acquire data a half
            #     #hour at a time doesnt actually work, it grabs 30*60/.1 data points
            #     #thus we hit the end of the file too soon, so as a stupid solution
            #     #i just return the empty data frame and all function calls
            #     #downstream just skip through when they get it
            #     data['Time'] = data.index
            #     data.index = data.index.tz_convert('UTC')

        except FileNotFoundError:
            print(filename + ' not found')
            return None

        return data

class OrbitData:
    #wrong dir
    _data_dir = '/home/wyatt/Documents/SAMPEX/OrbitData'
    # _data_dir = "/media/wyatt/64A5-F009/OrbitData"
    def __init__(self, filename=None, date=None):
        self.data_dir = '/home/wyatt/Documents/SAMPEX/OrbitData/'
        if filename is None and date is None:
            raise TypeError("Either a time or a file is needed")
        elif filename is None:
            self._init_date(date)
        elif date is None:
            self._init_filename(filename)
        else:
            self._init_filename(filename)
            if not self.date_in(date):
                raise TypeError("Two input arguments provided, "
                                "but don't agree")

    def _init_filename(self, filename):
        """
        Initializes with a given file.
        :param filename: The orbit data file
        """
        self.filename = filename
        start_year = pd.to_datetime(filename[-19:-15], utc=True)
        start_days = pd.to_timedelta(int(filename[-15:-12])-1, 'D')
        self.start = start_year + start_days

        end_year = pd.to_datetime(filename[-11:-7], utc=True)
        end_days = pd.to_timedelta(int(filename[-7:-4]), 'D')
        self.end = end_year + end_days

    def _init_date(self, date):
        """
        Initializes to use orbit data for a specific time
        :param date: Datetime64 object for the time of orbital data to use
        """
        files = [f for f in listdir(self._data_dir)
                 if isfile(join(self._data_dir, f))]
        found_file = False
        for data_filename in files:
            # Checks each file in in the catlog to see if it's valid.
            self._init_filename(join(self._data_dir, data_filename))
            if self.date_in(date):
                found_file = True
                break

        if not found_file:
            raise ValueError("File for ", date, " could not be found")

    def __str__(self):
        return 'DataFile: ' + str(self.start) + ' - ' + str(self.end)

    def date_in(self, date):
        """
        Tests if date is in the range of the object's orbit datafile
        :param date: Datetime64 object
        :return: Bool: if date is contained
        """
        return (self.start <= date) & (date < self.end)

    def read_data(self, parameters=None):
        """
        Creates a pandas dataframe from the orbit data file
        :param parameters:
        :param approx_start [day sec]
        :param approx_end   [day sec]
        :return:
        """
        if parameters is None:
            col_list = []
        else:
            col_list = parameters.copy()
        # Add date information
        col_list.extend(self._col_names[0:3])
        data = pd.read_csv(self.filename, sep=' ', header=None,
                               parse_dates={'Time': self._col_names[0:3]},
                               date_parser=self.date_converter,
                               keep_date_col=True,
                               names=self._col_names,
                               usecols=col_list,
                               skiprows=list(range(0, 60)),
                               error_bad_lines=False,
                               skipinitialspace=True).set_index('Time')

        data = data[
            ~data.index.duplicated(keep='first')]
        data.sort_index()
        return data

    def read_time_range(self , startDate,endDate,parameters=None):
        #sometimes not enough data is read in, so I extend the range added in
        #by 20 rows before and after the time range I actually want

        totalRows = m.ceil((endDate-startDate).total_seconds() / 6.0) + 200

        if parameters is None:
            col_list = []
        else:
            col_list = parameters.copy()

        # Add date information
        col_list.extend(self._col_names[0:3])

        with open(self.filename,'r') as f :
            lineNumber = 0
            for line in f:
                if lineNumber==61:
                    firstLine = line.strip().split(' ')[0:3]
                lineNumber+=1

        firstDate = pd.to_datetime(firstLine[0] + ' ' + firstLine[1] , utc=True , format= '%Y %j') + pd.to_timedelta(int(firstLine[2]) , unit='s')

        rowsToSkip = abs(m.ceil((startDate - firstDate).total_seconds() / 6.0))
        if rowsToSkip>21:
            rowsToSkip-=20

        data = pd.read_csv(self.filename, sep=' ', header=None,
                               parse_dates={'Time': self._col_names[0:3]},
                               date_parser=self.date_converter,
                               keep_date_col=True,
                               names=self._col_names,
                               usecols=col_list,
                               skiprows=list(range(0, 60+rowsToSkip)),
                               nrows=totalRows,
                               error_bad_lines=False,
                               skipinitialspace=True).set_index('Time')
        data = data[
            ~data.index.duplicated(keep='first')]
        data.sort_index()
        return data


    # Names of the columns in the orbit data file
    _col_names = ['Year', 'Day-of-year', 'Sec_of_day',
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

    def date_converter(self, time):
        """
        Used to process date in orbit data files
        :param time: string of date from orbit data file
        :return: pandas timestamp of attitude line
        """
        return pd.to_datetime(time[0:8], utc=True,
                              format='%Y %j') + \
               pd.to_timedelta(
                   int(time[9:]),
                    unit='s')

    def load_year(self,year,parameters=None):
        files = [f for f in listdir(self._data_dir)
                 if isfile(join(self._data_dir, f))]
        found_file = False
        file_list = []
        for data_filename in files:
            # Checks each file in in the catlog to see if it's valid.
            self._init_filename(join(self._data_dir, data_filename))
            if str(year) in data_filename:
                file_list.append(data_filename)
        #now we need to load all the files that have that year
        if parameters is None:
            col_list = []
        else:
            col_list = parameters.copy()
        # Add date information
        col_list.extend(self._col_names[0:3])
        data = pd.DataFrame()
        # file_list = [file_list[0],file_list[1]]
        for file in file_list:
            data = data.append(pd.read_csv( self.data_dir + file, sep=' ', header=None,
                                   parse_dates={'Time': self._col_names[0:3]},
                                   date_parser=self.date_converter,
                                   keep_date_col=True,
                                   names=self._col_names,
                                   usecols=col_list,
                                   skiprows=list(range(0, 60)),
                                   error_bad_lines=False,
                                   skipinitialspace=True).set_index('Time')
                                   )
        data = data[
            ~data.index.duplicated(keep='first')]
        data = data.sort_index()
        return data


class sampexStats:
    """
    Contains functions to to get misc stats from a specific time for SAMPEX HILT
    """
    def __init__(self,time):
        orbitObj = OrbitData(date=time)
        start = time-pd.Timedelta("10s")
        end = time+pd.Timedelta("10s")
        self.orbitInfo = orbitObj.read_time_range(pd.to_datetime(start),
                                                  pd.to_datetime(end),
                                                  parameters=['GEI_X','GEI_Y','GEI_Z','L_Shell',
                                                              'GEO_Lat','GEO_Long','GEI_VX', 'GEI_VY', 'GEI_VZ',
                                                              "Magnetic_Lat","Altitude"])
        self.orbitInfo = self.orbitInfo[start:end]
        self.Re = 6371

    def _find_loss_cone(self,position,time):
        foot = irb.find_footpoint(time,position,extMag='T89')
        locus = foot['loci']
        locus.ticks=time
        foot = foot['Bfoot']
        eq = irb.find_magequator(time,position,extMag='T89')['Bmin']

        pitch=90 #for particles mirroring at 100km
        return np.rad2deg(np.arcsin(np.sqrt(np.sin(np.deg2rad(pitch))**2 * eq / foot)))

    def get_L(self):
        return self.orbitInfo["L_Shell"]

    def get_bounce_period(self,energy = 1):
        """
        calculates electron bounce period at edge of loss cone
        energy in MeV
        """

        if  (self.orbitInfo['GEO_Lat'].to_numpy() / self.Re)[0]>0:
            self.hemisphere = "N"
        else:
            self.hemisphere = "S"

        X = (self.orbitInfo['GEI_X'].to_numpy() / self.Re)[0]
        Y = (self.orbitInfo['GEI_Y'].to_numpy() / self.Re)[0]
        Z = (self.orbitInfo['GEI_Z'].to_numpy() / self.Re)[0]
        position  = np.array([X,Y,Z])
        ticks = Ticktock(self.orbitInfo.index[0])
        coords = spc.Coords(position,'GEI','car')

        Lstar = irb.get_Lstar(ticks,coords,extMag='0')
        Lstar = abs(Lstar['Lm'][0])

        # Lstar = self.orbitInfo['L_Shell'].to_numpy()[0]
        self.Lstar = Lstar
        loss_cone = self._find_loss_cone(coords,ticks) #in degrees
        mc2 = .511 #elec rest mass
        beta = np.sqrt(energy/mc2)*np.sqrt(2 + energy/mc2) / (1 + energy/mc2)

        period = .117 * Lstar * (1 - .4635 * np.sin(np.deg2rad(loss_cone))**.75) / beta
        return period[0]

    def get_velocity(self,coord="GEI"):

        vel_df = self.orbitInfo[['GEI_VX', 'GEI_VY', 'GEI_VZ']] #km/s
        vel_df = vel_df / self.Re
        if coord=="GEI":
            #upsample to 20ms
            vel_df = vel_df.resample("20ms").interpolate(method="spline",order=3)
            return vel_df
        else:
            #to make transformation matrix, we need only the time
            time = self.orbitInfo.index[0]
            bases = spc.Coords([[1,0,0],[0,1,0],[0,0,1]],'GEI','car')
            bases.ticks = Ticktock([time]*3,'UTC')
            transformation_matrix = bases.convert(coord,'car').data
            transformation_matrix = np.array(transformation_matrix)
            vectors=np.array(vel_df.values.tolist())
            vel_RLL = vectors.dot(transformation_matrix)
            df = pd.DataFrame({"vr":vel_RLL[:,0],"vlat":vel_RLL[:,1],"vlon":vel_RLL[:,2]}
                              ,index=self.orbitInfo.index)
            df = df.resample("20ms").interpolate(method="spline",order=3)
            return df



    def drift_velocity(self,energy,interp=None):
        """
        bounce averaged drift velocity at sampex's location for field aligned
        particles
        energy in MeV
        result in Re/s, in RLL coords [0,0,vel]

        """
        L = self.get_L() #type dataframe
        time = 60 * 62.7 / (L*energy) #s
        mlat = self.orbitInfo["Magnetic_Lat"]
        alt = self.orbitInfo["Altitude"]
        vel = 2 * np.pi * (1 + alt/self.Re) * np.cos(np.deg2rad(mlat)) / time
        vel = vel.rename("vlon")
        vel = vel.to_frame()
        vel["vr"]=0
        vel["vlat"]=0
        vel = vel[["vr","vlat","vlon"]]
        if interp is None:
            return vel
        else:
            return vel.resample(interp).interpolate(method="spline",order=3)

if __name__ == '__main__':
    quit()
    filename = '/home/wyatt/Documents/SAMPEX/data/HILThires/State1/hhrr1993001.txt'
    month = '01'
    eventList = []
    hours = ['0' + str(i) if i<10 else str(i) for i in range(24)]
    days = ['0' + str(i) if i<10 else str(i) for i in range(1,31)]
    for day in days:
        for hour in hours:
            eventList.append([pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':00:00' , tz = 'utc'),
                              pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':29:59' , tz = 'utc')])
            eventList.append([pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':29:59' , tz = 'utc'),
                              pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':59:59' , tz = 'utc')])
    event = eventList[0]
    data = quick_read(filename,event[0],event[1])

    print(data)
