# load model with joblib
import numpy as np
import pandas as pd
namasaham="bbca"
namasaham=namasaham.upper()
import joblib
model=joblib.load('model{}'.format(namasaham))

#dataset from yahoo finance
from yahoo_historical import Fetcher

real_saham= Fetcher(namasaham+".JK", [2019,7,10], [2019,7,18], interval="1d")
real_saham=real_saham.getHistorical()
real_saham=real_saham.iloc[:,1:2]
real_saham=real_saham.dropna()
print(real_saham)
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
saham=sc.fit_transform(real_saham)       
print(saham)

def createDataset(data, window):
    dataX= []
    for i in range(len(data)-window+1):
        temp = []
        for j in range(i, i+window):
            temp.append(data[j,0])
        dataX.append(temp)
       
    return np.array(dataX)
window = 3
predictX= createDataset(saham, window)
print(predictX)
print(predictX.shape)
predictX=predictX.reshape(len(saham)-window+1,window,1)


predict_value = model.predict(predictX)

predict_value= sc.inverse_transform(predict_value)
print(predict_value)
print(real_saham[window:])
predict_value=predict_value[-1][0]
print("Predict 15/07/19 = {}".format(int(predict_value)))

# # df=pd.DataFrame(dict(real_price=saham[window:],predict=predict_value[0:-1]))
# # print(df)