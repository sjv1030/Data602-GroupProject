import requests
import pandas as pd
import numpy as np
import cryptocompare
import datetime
from datetime import timedelta
import random
from numpy import concatenate
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from py_scripts import get_data

def get_future_data(crypto_data):
    crypto_data = crypto_data.reset_index()
    columns = ['timestamp', 'average']
    crypto_data_avg = crypto_data[columns]
    price = crypto_data['average']
    last = crypto_data_avg.tail(1)['timestamp'].dt.date
    future = []

    for i in range(35):
        time = last + timedelta(days=1)
        last = time
        future.append(time)

    usage = random.sample(range(int(min(price)), int(max(price)+100)), 35)    
    future_array = np.concatenate(future, axis=0)
    d = {'timestamp': future_array, 'average': usage}
    df = pd.DataFrame(data=d)
    crypto_data_avg_random = crypto_data_avg.append(df)
    prices = crypto_data_avg_random['average']
    
    crypto_data_avg.append(crypto_data_avg_random)
    
    return future_array, prices

def get_train_data(prices):
    values = prices.values.reshape(-1,1)
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    train_size = int(len(scaled) * 0.7)
    test_size = len(scaled) - train_size
    train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
    return train, test, scaler

def create_dataset(dataset, look_back=50):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def run_LSTM(crypto):
    tf.reset_default_graph() 
    crypto_data, hist = get_data.daily_price_historical(crypto, "USD")

    future_array, prices = get_future_data(crypto_data)
    
    train, test, scaler = get_train_data(prices)
    trainX, trainY = create_dataset(train, 50)
    testX, testY = create_dataset(test, 50)
    
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    history = model.fit(trainX, trainY, epochs=500, batch_size=100, validation_data=(testX, testY), verbose=0, shuffle=False)
    
    yhat = model.predict(testX)
    
    yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
    testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))
    
    return crypto_data, yhat_inverse