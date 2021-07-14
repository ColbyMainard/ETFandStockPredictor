import numpy as np

from keras.models import Model, save_model, load_model
from keras.utils.vis_utils import plot_model
from keras.layers import Conv1D, Input, MaxPool1D, Dense, Dropout, Conv1DTranspose, UpSampling1D, Flatten, LSTM
from keras import losses
from keras import regularizers
from keras import callbacks
from keras import activations

import datetime

import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

import pandas as pd
from tensorflow.python.keras.layers.merge import concatenate

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import math

X_train, X_test, y_train, y_test = [], [], [], []

#converts one date in format year-month-day into numerical values
def date_to_float(date):
    date = datetime.datetime.strptime(date, '%Y-%m-%d')
    return float(10000*date.year + 100*date.month + date.day)

#converts list of dates in format year-month-day into numerical values
def date_list_to_floats(date_list):
    dates_as_integers = []
    for idx in range(len(date_list)):
        dates_as_integers.append(date_to_float(date_list[idx]))
    return dates_as_integers

def split_data_into_x_and_y(data_list):
    num_points = len(data_list)
    x_data_list = []
    y_data_list = []
    data_list = minmax_scale(data_list)
    for idx in range(int(num_points / 1000)):
        start_x = idx * 500
        end_x = start_x + 250
        end_y = start_x + 500
        x_data_elem = data_list[start_x:end_x]
        for list_index in range(len(x_data_elem)):
            try:
                x_data_elem[list_index] = math.log2(x_data_elem[list_index])
            except:
                x_data_elem[list_index] = math.log2(0.0001)
        fit_x = LinearRegression().fit(np.array(range(250)).reshape(250,1), x_data_elem)
        x_data_elem = x_data_elem - (2 ** ((range(250) * fit_x.coef_[0]) + fit_x.intercept_))
        x_data_list.append(x_data_elem)
        y_data_elem = data_list[end_x:end_y]
        for list_idx in range(len(y_data_elem)):
            try:
                y_data_elem[list_idx] = math.log2(y_data_elem[list_idx])
            except:
                y_data_elem[list_idx] = math.log2(0.0001)
        fit_y = LinearRegression().fit(np.array(range(250)).reshape(250,1), y_data_elem)
        y_data_elem = y_data_elem - (2 ** ((range(250) * fit_y.coef_[0]) + fit_y.intercept_))
        y_data_list.append(y_data_elem)
    return (x_data_list, y_data_list)

#loads data file
def load_data_file(filename, in_train_set = True):
    data_frame = pd.read_csv(filename)
    prices = list(data_frame["Close"])
    price_splits = split_data_into_x_and_y(prices)
    if in_train_set:
        global X_train, y_train
        for element in price_splits[0]:
            X_train.append(element)
        for element in price_splits[1]:
            y_train.append(element)
    else:
        global X_test, y_test
        for element in price_splits[0]:
            X_test.append(element)
        for element in price_splits[1]:
            y_test.append(element)

def load_file_directory(directory_name, in_train_set = True):
    print("Loading directory", directory_name)
    for filename in os.listdir(directory_name):
        load_data_file(os.path.join(directory_name, filename), in_train_set)

load_file_directory(os.path.join("Stock Data", "train"), in_train_set = True)
load_file_directory(os.path.join("Stock Data", "test"), in_train_set = False)

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

input_layer = Input(shape=(250, 1,))

#conv branch 1
conv_1_1 = Conv1D(16, 5, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(input_layer)
conv_2_1 = Conv1D(32, 5, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(conv_1_1)
conv_3_1 = Conv1D(64, 5, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(conv_2_1)
max_pool_1_1 = MaxPool1D(pool_size=5, padding='same')(conv_3_1)
conv_4_1 = Conv1D(128, 5, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(max_pool_1_1)
conv_5_1 = Conv1D(256, 5, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(conv_4_1)
conv_6_1 = Conv1D(512, 5, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(conv_5_1)
lstm_1_1 = LSTM(256, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(conv_6_1)

#conv branch 2
conv_1_2 = Conv1D(32, 10, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(input_layer)
conv_2_2 = Conv1D(64, 10, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(conv_1_2)
max_pool_1_2 = MaxPool1D(pool_size=10, padding='same')(conv_2_2)
conv_3_2 = Conv1D(128, 10, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(max_pool_1_2)
conv_4_2 = Conv1D(256, 10, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(conv_3_2)
lstm_1_2 = LSTM(256, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(conv_4_2)

#conv branch 3
conv_1_3 = Conv1D(32, 15, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(input_layer)
max_pool_1_3 = MaxPool1D(pool_size=15, padding='same')(conv_1_3)
conv_2_3 = Conv1D(64, 15, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(max_pool_1_3)
lstm_1_3 = LSTM(256, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(conv_2_3)

lstm_input = LSTM(256, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(input_layer)

concat = concatenate([lstm_1_1, lstm_1_2, lstm_1_3, lstm_input], axis=-1)
dropout_1 = Dropout(0.2)(concat)
hidden_dense_1 = Dense(512, activation=activations.relu, kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(dropout_1)
dropout_2 = Dropout(0.2)(hidden_dense_1)
hidden_dense_2 = Dense(512, activation=activations.relu, kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(dropout_2)
dropout_3 = Dropout(0.2)(hidden_dense_2)
output_layer = Dense(250, activation=activations.linear, kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(dropout_3)

network = Model(input_layer, output_layer)
network.compile(loss=losses.MAPE)
plot_model(network, to_file='stock_network.png', show_shapes=True)
print(network.summary())
callbacks_list_stocks = [callbacks.EarlyStopping(monitor='val_loss', patience=1,), callbacks.ModelCheckpoint(filepath='stock_predictor.h5', monitor='val_loss', save_best_only=True,), callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=10,), callbacks.TensorBoard(log_dir='stock_tensorboard',histogram_freq=1,embeddings_freq=1,)]
history = network.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=callbacks_list_stocks)
network = load_model('stock_predictor.h5')
network.evaluate(X_test, y_test)
predictions = network.predict(X_test)
num_rows = predictions.shape[0]

del X_train, y_train

predictions = predictions.tolist()
y_test = y_test.tolist()

os.mkdir(os.path.join("Stock Data", "predictions"))

for idx in range(num_rows):
    data = {"prediction": predictions[idx], "actual": y_test[idx]}
    data = pd.DataFrame.from_dict(data)
    filename = str(idx) + ".csv"
    filename = os.path.join("Stock Data", "predictions", filename)
    data.to_csv(filename, index=True, index_label = 'Day')

plt.scatter(x=range(len(history.history['loss'])), y=history.history['loss'])
plt.scatter(x=range(len(history.history['val_loss'])), y=history.history['val_loss'])
plt.show()
print("Done!")