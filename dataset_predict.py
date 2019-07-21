import numpy as np

def createDataset(data, window):
    dataX= []
    for i in range(len(data)-window+1):
        temp = []
        for j in range(i, i+window):
            temp.append(data[j,0])
        dataX.append(temp)
    return np.array(dataX)