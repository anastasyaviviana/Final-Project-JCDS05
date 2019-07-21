from yahoo_historical import Fetcher
import matplotlib.pyplot as plt
import numpy as np

    
def dataset(saham,tanggal1,tanggal2):
    real_saham=Fetcher("{}.JK".format(saham), tanggal1, tanggal2, interval="1d")
    real_saham=real_saham.getHistorical()
    real_saham=real_saham.dropna()
    return real_saham

#data machine learning
tanggal1=[2012,7,12]
tanggal2=[2019,7,12]
bbri=dataset('BBRI',tanggal1,tanggal2)
bbca=dataset('BBCA',tanggal1,tanggal2)
bbni=dataset('BBNI',tanggal1,tanggal2)
bmri=dataset('BMRI',tanggal1,tanggal2)

plt.figure(figsize=(12,8))
plt.subplot(1,1,1)
plt.plot(np.arange(len(bbri)),bbri['Open'],color='blue',label='Real stock price of BBRI')
plt.plot(np.arange(len(bbri)),bbca['Open'],color='red',label='Real stock price of BBCA')
plt.plot(np.arange(len(bbri)),bbni['Open'],color='green',label='Real stock price of BBNI')
plt.plot(np.arange(len(bbri)),bmri['Open'],color='orange',label='Real stock price of BMRI')
plt.title('Real Stock Price')
plt.xlabel('Date')
plt.ylabel('Stock price')
# plt.xticks(bbri['Date'],rotation=90)
plt.legend()
plt.tight_layout()
# namafile='realdata'
# addressplot='./storage/{}.png'.format(namafile)
# plt.savefig(addressplot)

#Percentage based on popularity
tanggal1=[2018,7,19]
tanggal2=[2019,7,19]
bbri=dataset('BBRI',tanggal1,tanggal2)
bbca=dataset('BBCA',tanggal1,tanggal2)
bbni=dataset('BBNI',tanggal1,tanggal2)
bmri=dataset('BMRI',tanggal1,tanggal2)

sumbbri=bbri.Volume.sum()
sumbbca=bbca.Volume.sum()
sumbbni=bbni.Volume.sum()
sumbmri=bmri.Volume.sum()

#plot percentage popularity in one year
labels = ['BBRI','BBCA','BBNI','BMRI']
sizes = [sumbbri,sumbbca,sumbbni,sumbmri]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',startangle=90)
plt.title('Percentage of Stock Based On Populatity in 1 Year')
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax1.axis('equal') 
plt.tight_layout()
namafile='plotpopularity'
addressplot='./static/images/{}.png'.format(namafile)
plt.savefig(addressplot)
plt.show()