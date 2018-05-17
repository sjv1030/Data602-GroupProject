import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
import cryptocompare
import datetime
from datetime import timedelta
import random
from numpy import concatenate
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

def set_requirements(num_periods, forecast, inputs, nodes, output, learning_rate, epochs):
    return num_periods, forecast, inputs, nodes, output, learning_rate, epochs

def batch_data(prices, num_periods, forecast):    
    time_series = np.array(prices)
    x_data = time_series[:(len(time_series)-(len(time_series)%num_periods))]
    x_batches = x_data.reshape(-1, num_periods, 1)
    y_data = time_series[:(len(time_series)-(len(time_series)%num_periods))+forecast]
    y_batches = x_data.reshape(-1, num_periods, 1)
    return time_series, x_data, x_batches, y_data, y_batches

def test_data(time_series, forecast, num_periods):
    test_x_setup = time_series[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
    testY = time_series[-(num_periods):].reshape(-1, num_periods, 1)
    tf.reset_default_graph()
    return testX, testY

def create_RNN(nodes, inputs, output, num_periods, learning_rate, x_batches, y_batches, testX):
    X = tf.placeholder(tf.float32, [None, num_periods, inputs])   
    y = tf.placeholder(tf.float32, [None, num_periods, output])
    
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=nodes, activation=tf.nn.relu)   
    rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)  
    stacked_rnn_output = tf.reshape(rnn_output, [-1, nodes])          
    stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        
    outputs = tf.reshape(stacked_outputs, [-1, num_periods, output]) 
    
    loss = tf.reduce_sum(tf.square(outputs - y))    #define the cost function which evaluates the quality of our model
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
    training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 

    init = tf.global_variables_initializer()
    epochs = 1000     #number of iterations or training cycles, includes both the FeedFoward and Backpropogation

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        for ep in range(epochs):
            sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
            if ep % 100 == 0:
                mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
                save_path = saver.save(sess, "/tmp/model.ckpt")

        y_pred_RNN = sess.run(outputs, feed_dict={X: testX})
    return X, y_pred_RNN

def main_RNN(crypto):
    num_periods, forecast, inputs, nodes, output, learning_rate, epochs = set_requirements(num_periods=35, forecast=40, inputs=1, nodes=500, output=1, learning_rate=.0001, epochs=1000)

    crypto_data, hist = get_data.daily_price_historical(crypto, "USD")
    
    future_array, prices = get_future_data(crypto_data)
    time_series, x_data, x_batches, y_data, y_batches = batch_data(prices, num_periods, forecast)
    testX, testY = test_data(time_series, forecast, num_periods)
    X, y_pred_RNN = create_RNN(nodes, inputs, output, num_periods, learning_rate, x_batches, y_batches, testX)
    one_month_RNN  = y_pred_RNN[0][34]

    return one_month_RNN