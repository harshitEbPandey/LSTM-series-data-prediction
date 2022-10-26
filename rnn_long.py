import numpy as np
import pandas as pd
import tensorflow as tf

seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)
df = pd.read_csv('./coin_Bitcoin.csv')
df.head()
def format(row):
  d = row['Date'].split()
  l = d[0].split('-')
  dd, mm, yyyy = l[2], l[1], l[0]
  new_date = dd + '-' + mm + '-' + yyyy
  return new_date + ' ' + d[1]

def data_preprocessing(df):
  df = df.drop(['SNo','Name','Symbol','High','Low','Open','Volume','Marketcap'], axis = 1)
  df['date'] = df.apply(format, axis = 1)
  df = df.drop(['Date'], axis = 1)
  last_column = df.pop('date')
  df.insert(0, 'Date', last_column)
  return df

df = data_preprocessing(df)

from sklearn.preprocessing import MinMaxScaler
data = df.values[:, 1]
trans = MinMaxScaler()
data1 = trans.fit_transform(data.reshape(-1,1))

df1 = pd.DataFrame(df)
df1.columns = df.columns
df1['Close'] = data1




SPLIT = 0.80
n = len(df)
train_df = df1.iloc[0:int(n * SPLIT),:]
test_df = df1.iloc[int(n * SPLIT):, :]

train_df_datetime = train_df.pop('Date')
test_df_datetime = test_df.pop('Date')


train_df_datetime

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, test_df=test_df,
               label_columns=None):
    self.train_df = train_df
    self.test_df = test_df
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift
    self.total_window_size = input_width + shift
    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]
    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

        
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])
  return inputs, labels
WindowGenerator.split_window = split_window


def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size, sequence_stride=1, shuffle=True, batch_size=64,)
  ds = ds.map(self.split_window)
  return ds
WindowGenerator.make_dataset = make_dataset

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size, sequence_stride=1, shuffle=True, batch_size=64,)
  ds = ds.map(self.split_window)
  return ds
WindowGenerator.make_dataset = make_dataset


@property
def train(self):
  return self.make_dataset(self.train_df)
@property
def test(self):
  return self.make_dataset(self.test_df)
@property
def example(self):
  result = getattr(self, '_example', None)
  if result is None:
    result = next(iter(self.train))
    self._example = result
  return result
WindowGenerator.train = train
WindowGenerator.test = test
WindowGenerator.example = example

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(units=1),
])

return_sequences=True

MAX_EPOCHS = 50
def compile_and_fit(model, window, patience=5):
  early_stopping = tf.keras.callbacks.EarlyStopping( monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.test,
                      callbacks=[early_stopping])
  return history

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=['Close'])

history = compile_and_fit(lstm_model, wide_window)

predictions = lstm_model.predict(
    tf.keras.preprocessing.timeseries_dataset_from_array(data= test_df, targets=None, sequence_length=wide_window.input_width, sequence_stride=1, shuffle=False, batch_size=1,)
)
outputs = predictions[:, -1:,:]
new_shape = (outputs.shape[0], outputs.shape[1])
df_preds2_test = pd.DataFrame(
    data=outputs.reshape(new_shape),
    columns=['Close'],
)
df_preds_test = pd.DataFrame(np.nan, index=range(24), columns=['Close'])
df_preds_test = df_preds_test.append(df_preds2_test)
idx_start = test_df.index[0]
idx_end = test_df.index[-1]
df_preds_test['idxs'] = range(idx_start, idx_end + 2)
df_preds_test.set_index(keys='idxs', drop=True, inplace=True)


from sklearn.metrics import mean_squared_error, r2_score

r2score = r2_score(test_df.iloc[24:,:], df_preds_test.iloc[25:,:])
r2score
