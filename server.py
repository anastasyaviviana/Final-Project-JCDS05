from flask import redirect, request, Flask, render_template, url_for,send_from_directory
import requests
import pandas as pd
import numpy as np
import joblib
import datetime
from yahoo_historical import Fetcher
import dataset_predict
import os
import matplotlib.pyplot as plt
from keras import backend as K

app=Flask(__name__)
app.config['UPLOAD_FOLDER']='./storage'

@app.route('/')
def home1():
    return render_template('home.html')

@app.route('/prediction')
def home():
    a=datetime.datetime.now()+ datetime.timedelta(days=1)
    a=a.strftime('%Y-%m-%d')
    return render_template('prediction.html',a=a)

@app.route('/hasilprediksi', methods=['POST','GET'])
def post():
    K.clear_session()
    namasaham=request.form['namasaham']
    model=joblib.load('model'+namasaham)
  
    #split tanggal,bulan,tahun
    date=request.form['tanggal']
    date=datetime.datetime.strptime(date,'%Y-%m-%d')
    tanggal=date.strftime('%d')
    if tanggal[0]==str(0):
        tanggal1=int(tanggal[1])
    else:
        tanggal1=int(tanggal)
    print(tanggal1)
    bulan=date.strftime('%m')
    if bulan[0]==str(0):
        bulan1=int(bulan[1])
    else:
        bulan1=int(bulan)
    tahun=date.strftime('%Y')
    tahun1=int(tahun)
    tanggal_predict=str(tahun+'-'+bulan+'-'+tanggal)

    #import data stock price
    real_saham= Fetcher(namasaham+".JK", [2018,1,1], [tahun1,bulan1,tanggal1], interval="1d")
    real_saham=real_saham.getHistorical()
    real_saham=real_saham.iloc[:,0:2]
    real_saham=real_saham.dropna()
    
    #set index and drop data in predict_date
    real_saham= real_saham.set_index("Date")
    if tanggal_predict in real_saham.index.values:
        real_saham= real_saham.drop(tanggal_predict, axis=0)
    real_saham= real_saham.tail(10)
    
    #transform
    from sklearn.preprocessing import MinMaxScaler
    sc=MinMaxScaler()
    saham=sc.fit_transform(real_saham) 

    window = 3
    predictX= dataset_predict.createDataset(saham, window)
    predictX=predictX.reshape(len(saham)-window+1,window,1)
    predictY= model.predict(predictX)

    #denormalisasi
    predictY= sc.inverse_transform(predictY)
    predict_next_day="Rp. {}".format(int(predictY[-1][0]))
    price_previous_day="Rp. {}".format(int(real_saham['Open'][-1]))
    previous_date=real_saham.index[-1]

    return render_template(
        'predict.html',
        predict=predict_next_day,
        price_previous=price_previous_day,
        tanggal_predict=tanggal_predict,
        previous_date=previous_date,
        namasaham=namasaham)

@app.route('/menuhist')
def menuhist():
    a=datetime.datetime.now()
    a=a.strftime('%Y-%m-%d')
    return render_template('menuhist.html',a=a)

@app.route('/histdata',methods=['GET','POST'])
def histdata():
    a=datetime.datetime.now()
    a=a.strftime('%Y-%m-%d')
    namasaham=request.form['namasaham']
    tgl1=request.form['tanggal1']
    tgl2=request.form['tanggal2']
    
    def convert_date(date):
        date=datetime.datetime.strptime(date,'%Y-%m-%d')
        tanggal=date.strftime('%d')
        if tanggal[0]==str(0):
            tanggal1=int(tanggal[1])
        else:
            tanggal1=int(tanggal)
        
        bulan=date.strftime('%m')
        if bulan[0]==str(0):
            bulan1=int(bulan[1])
        else:
            bulan1=int(bulan)
        tahun=date.strftime('%Y')
        tahun1=int(tahun)
        tanggal_lengkap=str(tahun+'-'+bulan+'-'+tanggal)
        return tanggal1,bulan1,tahun1,tanggal_lengkap
    tgl1_1,bulan1_1,tahun1_1,tgl_lengkap1=convert_date(tgl1)
    tgl2_1,bulan2_1,tahun2_1,tgl_lengkap2=convert_date(tgl2)

    #import data stock price
    real_saham= Fetcher(namasaham+".JK", [tahun1_1,bulan1_1,tgl1_1], [tahun2_1,bulan2_1,tgl2_1], interval="1d")
    real_saham=real_saham.getHistorical()
    real_saham=real_saham.dropna()

    #plot data
    plt.plot(real_saham['Date'],real_saham['Open'],'blue')
    plt.title('Plot Stock Price of {}'.format(namasaham))
    plt.xlabel('Date')
    plt.ylabel('Price (Rp)')
    plt.xticks(real_saham['Date'],rotation=90)
    plt.tight_layout()

    #filename
    b=datetime.datetime.now()
    namafile=(str(b).split('.'))[-1]

    addressplot='./storage/{}.png'.format(namafile)
    urlplot='/fileupload/{}.png'.format(namafile)
    plt.savefig(addressplot)
    plot=urlplot
    plt.close()
    return render_template('histdata.html',tables=real_saham.to_html(),a=a,tgl1=tgl_lengkap1,tgl2=tgl_lengkap2,namasaham=namasaham,plot=plot)

@app.route('/fileupload/<path:x>')
def hasilUpload(x):
    return send_from_directory('storage',x)

@app.route('/scoremodel')
def scoremodel():
   
    return render_template('scoremodel.html')

@app.route('/plotdata')
def plotdata():
    return render_template('plotdata.html')
if __name__=='__main__':

    app.run(debug=True)