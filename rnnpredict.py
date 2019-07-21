# load model with joblib
import numpy as np
import pandas as pd
namasaham="bbri"
namasaham=namasaham.upper()
import joblib
model=joblib.load('model{}'.format(namasaham))

#dataset from yahoo finance
from yahoo_historical import Fetcher

real_saham= Fetcher(namasaham+".JK", [2019,6,18], [2019,7,18], interval="1d")
real_saham=real_saham.getHistorical()
date_and_open=real_saham.iloc[:,0:2]
real_saham=real_saham.iloc[:,1:2]
real_saham=real_saham.dropna()
print(real_saham)
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
saham=sc.fit_transform(real_saham)       
print(saham)

def createDataset(data, window):
    dataX= []
    for i in range(len(data)-window):
        temp = []
        for j in range(i, i+window):
            temp.append(data[j,0])
        dataX.append(temp)
       
    return np.array(dataX)
window = 3
predictX= createDataset(saham, window)
predictX=predictX.reshape(predictX.shape[0],window,1)

# #predict value
predict_value = model.predict(predictX)
predict_value= sc.inverse_transform(predict_value)

df_predict=date_and_open[window:]
df_predict['predict']=predict_value.reshape(-1)
print(df_predict)