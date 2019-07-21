import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##Dataset from yahoo finance

from yahoo_historical import Fetcher
namasaham="bmri"
namasaham=namasaham.upper()
saham= Fetcher(namasaham+".JK", [2012,7,12], [2019,7,12], interval="1d")
saham=saham.getHistorical()
#Open Price
saham=saham.iloc[:,1:2]  
# print(saham)

##Data Preprocessing

#delete na values
saham=saham.dropna()            

#split dataset into training and testing
percent_train=0.75
trainingset=saham.iloc[:int(len(saham)*percent_train),:]
testset=saham.iloc[int(len(saham)*percent_train):,:]

# Scale the features 
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
trainingset=sc.fit_transform(trainingset)       
testset=sc.fit_transform(testset)

# Build trainX and trainY
def createDataset(data, window):
    dataX, dataY = [], []
    for i in range(len(data)-window):
        temp = []
        for j in range(i, i+window):
            temp.append(data[j,0])
        dataX.append(temp)
        dataY.append(data[i+window,0])
    return np.array(dataX), np.array(dataY)
window = 3
trainX, trainY = createDataset(trainingset, window)
trainX=trainX.reshape(len(trainingset)-window,window,1)     #reshape to 3D
print(trainX)
print(trainY)       #1 D

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Model
regressor=Sequential()
# hidden layer
regressor.add(LSTM(units=50,activation='tanh',return_sequences=True,input_shape=(trainX.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# Adding the output layer
regressor.add(Dense(units = 1))
regressor.compile(optimizer='adam',loss='mean_squared_error') #adam
regressor.fit(trainX,trainY,epochs=500,batch_size=32)

#built testX into 3D and testY 1D
testX, testY = createDataset(testset, window)   #split x and y
testX=testX.reshape(len(testset)-window,window,1)   #reshape x to 3D

#prediction
predict_price=regressor.predict(testX)

# denormalisasi testY for plotting
testY=sc.inverse_transform(testY.reshape(-1,1))
predict_price=sc.inverse_transform(predict_price)
trainingset=sc.inverse_transform(trainingset)
testset=sc.inverse_transform(testset)
dfpredict=pd.DataFrame(dict(testY=list(testY),predict=list(predict_price)))
print(dfpredict)

def dstat(x,y):
    dstat = 0
    n = len(y)
    for i in range(n-1):
        if ((x[i+1]-y[i])*(y[i+1]-y[i]))>0 :
            dstat += 1
            Dstat = (float(1/(n-1))*dstat)*100
    return float(Dstat)

print('Dstat = {}%'.format(dstat(testY,predict_price)))

# ploting the results
# plot testset
plt.plot(np.arange(len(testY)),testY,color='red',label='Real stock price of {}'.format(namasaham))
# plot predict
plt.plot(np.arange(len(predict_price)),predict_price,color='blue',label='Predicted stock price of {}'.format(namasaham))
plt.title('Predicting Data Testing Stock Price of {}\nScore Dstat = {}%'.format(namasaham,round(dstat(testY,predict_price),3)))
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.savefig('/templates/testing_predict_{}.png'.format(namasaham))
plt.show()

# import joblib
# joblib.dump(regressor,'model{}'.format(namasaham))
