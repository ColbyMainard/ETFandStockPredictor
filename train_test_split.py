import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def load_directory(dir_name):
    file_names = os.listdir(dir_name)
    file_name_list = []
    data_frame_list = []
    for file_name in file_names:
        relative_location = os.path.join(dir_name, file_name)
        try:
            data_frame = pd.read_csv(relative_location)
            file_name_list.append(file_name)
            data_frame_list.append(data_frame)
            del data_frame
        except:
            print("\tFile", relative_location, "appears to not be formatted properly.  Ignoring...")
    return (file_name_list, data_frame_list)

def train_test_split_on_files(file_names_and_contents):
    return train_test_split(file_names_and_contents[0], file_names_and_contents[1], shuffle=True, test_size=0.2)

print("Loading stock data...")
stock_file_contents = load_directory("Stocks")
print("Splitting stock data into training and testing sets...")
split_stock = train_test_split_on_files(stock_file_contents)
print("Writing to disk...")
file_name_train = split_stock[0]
file_name_test = split_stock[1]
data_split_train = split_stock[2]
data_split_test = split_stock[3]
del split_stock
os.mkdir("Stock Data")
os.mkdir(os.path.join("Stock Data", "train"))
os.mkdir(os.path.join("Stock Data", "test"))
for idx in range(len(file_name_train)):
    data_split_train[idx].to_csv(os.path.join("Stock Data", "train", file_name_train[idx]), index=False)
for idx in range(len(file_name_test)):
    data_split_test[idx].to_csv(os.path.join("Stock Data", "test", file_name_test[idx]), index=False)
del file_name_test, file_name_train, data_split_test, data_split_train
print("Done!")

print("Loading ETF data...")
stock_file_contents = load_directory("ETFs")
print("Splitting ETF data into training and testing sets...")
split_etf = train_test_split_on_files(stock_file_contents)
print("Writing to disk...")
file_name_train = split_etf[0]
file_name_test = split_etf[1]
data_split_train = split_etf[2]
data_split_test = split_etf[3]
del split_etf
os.mkdir("ETF Data")
os.mkdir(os.path.join("ETF Data", "train"))
os.mkdir(os.path.join("ETF Data", "test"))
for idx in range(len(file_name_train)):
    data_split_train[idx].to_csv(os.path.join("ETF Data", "train", file_name_train[idx]), index=False)
for idx in range(len(file_name_test)):
    data_split_test[idx].to_csv(os.path.join("ETF Data", "test", file_name_test[idx]), index=False)
del file_name_test, file_name_train, data_split_test, data_split_train
print("Done!")