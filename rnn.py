import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential as Sequential
from keras.layers import Dense
from keras.layers import LSTM

df = pd.read_csv("coin_Bitcoin.csv",parse_dates=['Date'])
df = df.drop(columns=['SNo','Name','Symbol','Date'])
df = df[df.Volume != 0]

scaler = MinMaxScaler()
df[['Low','Open','Close','Marketcap','High','Volume']] = scaler.fit_transform(df[['Low','Open','Close','Marketcap','High','Volume']])

ds = tf.keras.preprocessing.timeseries_dataset_from_array(
     data=df[['Low','Open','Close','Marketcap','Volume']],
     targets=df['High'],
     batch_size = 1,
     sequence_stride=5,
     shuffle = False,
     sequence_length=25)

def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=1):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

train_ds,val_ds,test_ds = get_dataset_partitions_tf(ds,len(ds))

from keras.layers import Dropout
from keras.layers import RepeatVector
from tensorflow.keras.optimizers import RMSprop
regressor = Sequential()

regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (25,5,),activation='relu'))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 256, activation='relu'))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 256, activation='relu'))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 128, activation='relu'))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 64, activation='relu'))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 32, activation='relu'))
regressor.add(Dense(units = 1, input_shape=(5,1,), activation='relu'))

regressor.compile(RMSprop(learning_rate=0.00001), loss = 'mse')
history = regressor.fit(train_ds, epochs = 10, batch_size = 10,validation_data=val_ds,verbose=1)

import numpy as np
y_test = list(map(lambda x: x[1], test_ds))
y_test = np.asarray(y_test)

from sklearn.metrics import r2_score
y_pred = regressor.predict(test_ds, batch_size=1)
r2score = r2_score(y_test, y_pred)