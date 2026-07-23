import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import pathlib as Path
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError ,BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def scalingfortrain(x_inp , scaled_columns , scaler , unimportant_cols_todrop):
    if unimportant_cols_todrop is None:
        x_scaled = scaler.fit_transform( x_inp[scaled_columns] )
        x_nonscaled = x_inp.drop(columns=scaled_columns)
        final_x = np.concatenate((x_scaled ,x_nonscaled) ,axis=1)
    else:
        x_scaled = scaler.fit_transform(x_inp[scaled_columns])
        x_nonscaled = x_inp.drop(columns=scaled_columns + unimportant_cols_todrop)
        final_x = np.concatenate((x_scaled ,x_nonscaled) ,axis=1)
    return final_x
def scalingfortest(x_inp , scaled_columns , scaler , unimportant_cols_todrop):
    if unimportant_cols_todrop is None:
        x_scaled = scaler.transform(x_inp[scaled_columns])
        x_nonscaled = x_inp.drop(columns=scaled_columns)
        final_x = np.concatenate((x_scaled ,x_nonscaled) ,axis=1)
    else:
        x_scaled = scaler.transform(x_inp[scaled_columns])
        x_nonscaled = x_inp.drop(columns=scaled_columns + unimportant_cols_todrop)
        final_x = np.concatenate((x_scaled ,x_nonscaled) ,axis=1)
    return final_x

def data_splitting(X ,Y ,train_size ,test_size ,valid_size ,random_state):
    no_1 = test_size + valid_size
    no_2 = test_size / (test_size + valid_size)
    x_train ,x_temp ,y_train ,y_temp = train_test_split(
        X ,Y ,
        test_size=no_1 ,
        random_state=random_state
    )
    x_valid ,x_test ,y_valid ,y_test = train_test_split(
        x_temp ,y_temp ,
        test_size=no_2 ,
        random_state=random_state
    )
    return x_train ,x_valid ,x_test ,y_train ,y_valid ,y_test

def replace_values_in_csv(df ,column_name ,value_0 ,value_1 ,value_2):
    if value_2 is None:
        df[column_name] = df[column_name].replace({value_0: 0,value_1: 1}).astype(int)
    else:
        df[column_name] = df[column_name].replace({value_0:0 ,value_1: 1,value_2: 2}).astype(int)
    return df[column_name]