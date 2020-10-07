import requests,io
url = 'http://www.srl.caltech.edu/sampex/DataCenter/DATA/HILThires/State1/hhrr1993'
workDir = '/home/wyatt/Documents/SAMPEX/data/HILThires/State1/'

days = ['00' + str(i) for i in range(1,10)] + ['0' + str(i) for i in range(10,100)] + [str(i) for i in range(100,366)]
for day in days:
    print(day)
    down = url+ day+'.txt.zip'
    r = requests.get(down,allow_redirects=True)
    with open(workDir + 'hhrr1993'+day+'.txt.zip','wb') as f:
        f.write(r.content)

# for day in days:
#     zip_url = url + day + '.txt.zip'
#     r = requests.get(zip_url)
#     z = zipfile.ZipFile(io.BytesIO(r.content))
#     z.extractall()
