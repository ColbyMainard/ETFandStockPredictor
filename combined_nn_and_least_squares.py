import numpy as np

from keras.models import Model, save_model, load_model
from keras.utils.vis_utils import plot_model
from keras.layers import Conv1D, Input, MaxPool1D, Dense, Dropout, Conv1DTranspose, UpSampling1D, Flatten, LSTM, Bidirectional
from keras import losses
from keras import regularizers
from keras import callbacks
from keras import activations

import datetime

import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

import pandas as pd
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
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
    for list_index in range(len(data_list)):
        try:
            data_list[list_index] = math.log2(data_list[list_index])
        except:
            data_list[list_index] = math.log2(0.0001)
    trendline_fit = LinearRegression().fit(np.array(range(num_points)).reshape(num_points,1), data_list)
    data_list = data_list - ((range(len(data_list)) * trendline_fit.coef_[0]) + trendline_fit.intercept_)
    for idx in range(int(num_points / 500)):
        start_x = idx * 500
        end_x = start_x + 250
        end_y = start_x + 500
        x_data_elem = data_list[start_x:end_x]
        x_data_list.append(x_data_elem)
        y_data_elem = data_list[end_x:end_y]
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

load_file_directory(os.path.join("Combined Securities Data", "train"), in_train_set = True)
load_file_directory(os.path.join("Combined Securities Data", "test"), in_train_set = False)

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

input_layer = Input(shape=(250, 1,))

#conv branch 1
conv_1_1 = Conv1D(16, 5, padding='same', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='conv_1_1')(input_layer)
conv_2_1 = Conv1D(32, 5, padding='same', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='conv_2_1')(conv_1_1)
conv_3_1 = Conv1D(64, 5, padding='same', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='conv_3_1')(conv_2_1)
max_pool_1_1 = MaxPool1D(pool_size=5, padding='same', name='max_pool_1_1')(conv_3_1)
conv_4_1 = Conv1D(128, 5, padding='same', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='conv_4_1')(max_pool_1_1)
conv_5_1 = Conv1D(256, 5, padding='same', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='conv_5_1')(conv_4_1)
conv_6_1 = Conv1D(512, 5, padding='same', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='conv_6_1')(conv_5_1)
lstm_1_1 = LSTM(384, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001), name='lstm_1_1')(conv_6_1)

#conv branch 2
conv_1_2 = Conv1D(32, 10, padding='same', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='conv_1_2')(input_layer)
conv_2_2 = Conv1D(64, 10, padding='same', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='conv_2_2')(conv_1_2)
max_pool_1_2 = MaxPool1D(pool_size=10, padding='same', name='max_pool_1_2')(conv_2_2)
conv_3_2 = Conv1D(128, 10, padding='same', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='conv_3_2')(max_pool_1_2)
conv_4_2 = Conv1D(256, 10, padding='same', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='conv_4_2')(conv_3_2)
lstm_1_2 = LSTM(384, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001), name='lstm_1_2')(conv_4_2)

#conv branch 3
conv_1_3 = Conv1D(32, 15, padding='same', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='conv_1_3')(input_layer)
max_pool_1_3 = MaxPool1D(pool_size=15, padding='same', name='max_pool_1_3')(conv_1_3)
conv_2_3 = Conv1D(64, 15, padding='same', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='conv_2_3')(max_pool_1_3)
lstm_1_3 = LSTM(384, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001), name='lstm_1_3')(conv_2_3)

dense_input_1 = Dense(100, activation=activations.selu, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='dense_input_1')(input_layer)
dropout_input_1 = Dropout(0.15)(dense_input_1)
max_pool_input = MaxPool1D(pool_size=5, padding='same', name='max_pool_input')(dropout_input_1)
dense_input_2 = Dense(512, activation=activations.selu, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='dense_input_2')(max_pool_input)
dropout_input_2 = Dropout(0.15)(dense_input_2)
lstm_input = LSTM(384, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001), name='lstm_input')(dropout_input_2)
dense_input_final_1 = Dense(512, activation=activations.selu, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='dense_input_final_1')(lstm_input)
dropout_final_1 = Dropout(0.15)(dense_input_final_1)
dense_input_final_2 = Dense(512, activation=activations.selu, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='dense_input_final_2')(dense_input_final_1)
dropout_final_2 = Dropout(0.15)(dense_input_final_2)

#concat = concatenate([lstm_1_1, lstm_1_2, lstm_1_3, lstm_input], axis=-1)
concat = concatenate([lstm_1_1, lstm_1_2, lstm_1_3], axis=-1)

dropout_1 = Dropout(0.15)(concat)
hidden_dense_1 = Dense(1024, activation=activations.selu, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='hidden_dense_1')(dropout_1)
dropout_2 = Dropout(0.15)(hidden_dense_1)
hidden_dense_2 = Dense(1024, activation=activations.selu, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='hidden_dense_2')(dropout_2)
dropout_3 = Dropout(0.1)(hidden_dense_2)
hidden_dense_3 = Dense(512, activation=activations.selu, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='hidden_dense_3')(dropout_3)
dropout_4 = Dropout(0.1)(hidden_dense_3)
hidden_dense_4 = Dense(512, activation=activations.selu, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='hidden_dense_4')(dropout_4)
dropout_5 = Dropout(0.1)(hidden_dense_4)
concat_input = concatenate([dropout_final_2, dropout_5], axis=-1)
#output_layer = Dense(250, activation=activations.linear, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001))(dropout_5)
output_layer = Dense(250, activation=activations.linear, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001), name='output')(concat_input)

network = Model(input_layer, output_layer)
network.compile(loss=losses.MAE)
plot_model(network, to_file='combined_securities_network.png', show_shapes=True)
print(network.summary())
callbacks_list_combined_securities = [callbacks.EarlyStopping(monitor='val_loss', patience=5,), callbacks.ModelCheckpoint(filepath='combined_security_predictor.h5', monitor='val_loss', save_best_only=True,), callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=1,), callbacks.TensorBoard(log_dir='combined_security_tensorboard',histogram_freq=1,embeddings_freq=1,)]
history = network.fit(X_train, y_train, epochs=100, batch_size=96, validation_split=0.1, callbacks=callbacks_list_combined_securities)
network = load_model('combined_security_predictor.h5')
network.evaluate(X_test, y_test)
predictions = network.predict(X_test)
num_rows = predictions.shape[0]

del X_train, y_train

predictions = predictions.tolist()
y_test = y_test.tolist()

os.mkdir(os.path.join("Combined Securities Data", "predictions"))

for idx in range(num_rows):
    data = {"prediction": predictions[idx], "actual": y_test[idx]}
    data = pd.DataFrame.from_dict(data)
    filename = str(idx) + ".csv"
    filename = os.path.join("Combined Securities Data", "predictions", filename)
    data.to_csv(filename, index=True, index_label = 'Day')

print("Done!")
