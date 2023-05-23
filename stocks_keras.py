import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras import layers
from keras import Input
from keras.optimizers import RMSprop
from keras import utils
from keras import callbacks
import tensorflow as tf
import sys
import functions as func


np.set_printoptions(
    formatter={"float": lambda x: "{0:0.8f}".format(x)}, threshold=sys.maxsize
)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
aaplPath = "csv/AAPL.csv"
ndxPath = "csv/NDX.csv"
tslaPath = "csv/TSLA.csv"
vixPath = "csv/VIX.csv"

dfAapl = pd.read_csv(aaplPath)
dfNdx = pd.read_csv(ndxPath)
dfVix = pd.read_csv(vixPath)

aggData = dfAapl.merge(dfVix, how="inner", on="Date")
aggData = aggData.merge(dfNdx, how="inner", on="Date")
aggData.head()


closeData = {
    ##"aapl": aggData["Close_x"],
    "ndx": aggData["Close"],
    # "vix": aggData["Close_y"]
    ##"tsla": aggData["Close_y"]
}
closeData = pd.DataFrame(data=closeData)
closeData.index = aggData["Date"]
closeData.head()


normedValues = func.normalize(closeData.values)
normed = pd.DataFrame(normedValues[0])
normed.head()
# func.show_heatmap(closeData)


splitFraction = 0.65
splitFractionTest = 0.85
trainSplit = int(splitFraction * int(normed.shape[0]))
testSplit = int(splitFractionTest * int(normed.shape[0]))
past = 3
future = 1
trainData = normed.loc[0 : trainSplit - 1]
valData = normed.loc[trainSplit - past : testSplit]
testData = normed.loc[testSplit - past :]

x_train = trainData[:-future].values
y_train = trainData.iloc[past:][[0]]
y_train = y_train.values

x_val = valData[:-future].values
y_val = valData.iloc[past:][[0]]
y_val = y_val.values

x_test = testData[:-future].values
y_test = testData.iloc[past:][[0]]
y_test = y_test.values


step = 1
learning_rate = 0.0001
batch_size = 32
epochs = 25
sequence_length = int(past / step)

dataset_train = utils.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
    shuffle=False,
)

dataset_val = utils.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
    shuffle=False,
)
dataset_test = utils.timeseries_dataset_from_array(
    x_test,
    y_test,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
    shuffle=False,
)


for batch in dataset_train:
    inputs, targets = batch

##########      MODEL     ############

inputs = layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
x = layers.LSTM(64, dropout=0.02, return_sequences=True)(inputs)
x = layers.LSTM(32)(x)
lstm_out = layers.Dense(16)(x)
outputs = layers.Dense(1)(lstm_out)

model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=RMSprop(learning_rate=learning_rate),
    loss="mean_squared_error",
    metrics=["mae"],
)
model.summary()

#########     TRAINING    ############

path_checkpoint = "checkpt/model_checkpoint.h5"
es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[modelckpt_callback],
)

func.visualize_loss(history, "Training and Validation Loss")

targ = []
predictions = []

for x, y in dataset_val:
    targ.append(y)
    predictions.append(model.predict(x))

targ = func.deNormalize(
    [item[0].numpy() for batch in targ for item in batch],
    normedValues[1],
    normedValues[2],
)
predictions = func.deNormalize(
    [item[0] for batch in predictions for item in batch],
    normedValues[1],
    normedValues[2],
)
y_train = func.deNormalize([i[0] for i in y_train], normedValues[1], normedValues[2])

testTarg = []
testPredictions = []
for x, y in dataset_test:
    testTarg.append(y)
    testPredictions.append(model.predict(x))

testTarg = func.deNormalize(
    [item[0].numpy() for batch in testTarg for item in batch],
    normedValues[1],
    normedValues[2],
)

testPredictions = func.deNormalize(
    [item[0] for batch in testPredictions for item in batch],
    normedValues[1],
    normedValues[2],
)

func.showTrainingSet(y_train, targ, predictions, testTarg, testPredictions)
