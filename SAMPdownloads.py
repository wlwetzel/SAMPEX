import requests,io
year92 = False
if year92:
    url = 'http://www.srl.caltech.edu/sampex/DataCenter/DATA/HILThires/State1/hhrr1992'
    workDir = '/home/wyatt/Documents/SAMPEX/SAMPEX_Data/HILThires/State1/'

    days = ['00' + str(i) for i in range(1,10)] + ['0' + str(i) for i in range(10,100)] + [str(i) for i in range(100,366)]
    for day in days:
        print(day)
        down = url+ day+'.txt.zip'
        r = requests.get(down,allow_redirects=True)
        with open(workDir + 'hhrr1993'+day+'.txt.zip','wb') as f:
            f.write(r.content)

"""
State 2: First 20-msec SSD configuration: 1994-137 thru 1994-237.
note* missing data from 169-200
"""

state2=True
if state2:
    # workDir = '/home/wyatt/Documents/SAMPEX/SAMPEX_Data/HILThires/State2/'
    #workDir = "/media/wyatt/64A5-F009/SAMPEX_Data/HILThires/State2/"
    workDir = '/home/wyatt/Projects/SAMPEX/SAMPEX_Data/HILThires/State2/'

    url = 'http://www.srl.caltech.edu/sampex/DataCenter/DATA/HILThires/State2/hhrr1994'
    days = [str(i) for i in range(137,238)]
    for day in days:
        print(day)
        down = url+ day+'.txt.zip'
        r = requests.get(down,allow_redirects=True)
        with open(workDir + 'hhrr1994'+day+'.txt.zip','wb') as f:
            f.write(r.content)

"""
State 4: Second 20-msec SSD configuration: 1996-220 thru 2004-182.
2002129
"""
state4=True
if state4:
    # workDir = '/home/wyatt/Documents/SAMPEX/SAMPEX_Data/HILThires/State4/'
    #workDir = "/media/wyatt/64A5-F009/SAMPEX_Data/HILThires/State4/"
    workDir = '/home/wyatt/Projects/SAMPEX/SAMPEX_Data/HILThires/State4/'

    url = 'http://www.srl.caltech.edu/sampex/DataCenter/DATA/HILThires/State4/hhrr'
    years = [str(i) for i in [1996,1997,1998,1999,2000,2001,2002,2003,2004]]
    days = ['00' + str(i) for i in range(1,10)] + ['0' + str(i) for i in range(10,100)] + [str(i) for i in range(100,366)]
    dates = [year+day for year in years for day in days]
    start_ind = dates.index("2002129")
    for date in dates[start_ind:]:
        print(date)
        down = url+ date+'.txt.zip'
        r = requests.get(down,allow_redirects=True)
        with open(workDir + 'hhrr' + date + '.txt.zip','wb') as f:
            f.write(r.content)

# for day in days:
#     zip_url = url + day + '.txt.zip'
#     r = requests.get(zip_url)
#     z = zipfile.ZipFile(io.BytesIO(r.content))
#     z.extractall()
