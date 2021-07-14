from keras_visualizer import visualizer

import numpy as np

from keras.models import Model, save_model, load_model
from keras.utils.vis_utils import plot_model
from keras.layers import Conv1D, Input, MaxPool1D, Dense, Dropout, Conv1DTranspose, UpSampling1D, Flatten
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
    for idx in range(int(num_points / 500)):
        start_x = idx * 500
        end_x = start_x + 250
        end_y = start_x + 500
        x_data_list.append(data_list[start_x:end_x])
        y_data_list.append(data_list[end_x:end_y])
    return (x_data_list, y_data_list)

#loads data file
def load_data_file(filename, in_train_set = True):
    errors = 0
    try:
        #print("\tLoading file", filename)
        data_frame = pd.read_csv(filename)
        prices = list(data_frame["Close"])
        price_splits = split_data_into_x_and_y(prices)
        if in_train_set:
            global X_train, y_train
            for element in price_splits[0]:
                if not (len(element) == 0):
                    X_train.append(element)
            for element in price_splits[1]:
                if not (len(element) == 0):
                    y_train.append(element)
        else:
            global X_test, y_test
            for element in price_splits[0]:
                if not (len(element) == 0):
                    X_test.append(element)
            for element in price_splits[1]:
                if not (len(element) == 0):
                    y_test.append(element)
    except:
        errors += 1
    #print("Num errors:", errors)

def load_file_directory(directory_name, in_train_set = True):
    print("Loading directory", directory_name)
    for filename in os.listdir(directory_name):
        load_data_file(os.path.join(directory_name, filename), in_train_set)

load_file_directory(os.path.join("Stock Data", "train"), in_train_set = True)
load_file_directory(os.path.join("Stock Data", "test"), in_train_set = False)

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)
print("X train shape:", X_train.shape)
print("Y train shape", y_train.shape)
print("X test shape:", X_test.shape)
print("Y test shape", y_test.shape)

input_layer = Input(shape=(250,1,))
conv_1 = Conv1D(32, 5, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(input_layer)
conv_2 = Conv1D(32, 5, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(conv_1)
conv_3 = Conv1D(32, 5, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(conv_2)
deconv_1 = Conv1DTranspose(32, 5, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(conv_3)
deconv_2 = Conv1DTranspose(32, 5, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(deconv_1)
deconv_3 = Conv1DTranspose(32, 5, padding='same', kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(deconv_2)
flatten_1 = Flatten()(deconv_3)
dense_1 = Dense(512, activation=activations.relu, kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(flatten_1)
dropout_1 = Dropout(0.2)(dense_1)
dense_2 = Dense(512, activation=activations.relu, kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(dropout_1)
dropout_2 = Dropout(0.2)(dense_2)
output_layer = Dense(250, activation=activations.relu, kernel_regularizer=regularizers.l2(0.0025), bias_regularizer=regularizers.l2(0.0025))(dropout_2)

network = Model(input_layer, output_layer)
network.compile(loss=losses.MAE)
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