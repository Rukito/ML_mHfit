import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

#----- Model name --------------------------
model_name = 'model_50epochs'

#----- prepare data ------------------------
traindata = np.load('../Data/traindata.npy')
valdata = np.load('../Data/valdata.npy')

x_train = np.vstack((traindata.T[0:2], traindata.T[11:15]))
y_train = traindata[:,0]
x_val = np.vstack((valdata.T[0:2], valdata.T[11:15]))
y_val = valdata[:,0]

#----- create model ------------------------
model = Sequential()
model.add(Dense(10,input_dim=6, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))

#----- compile train and save model ---------------
model.compile(loss='mean_squared_error', optimizer='adagrad')
model.fit(x_train.T, y_train.T, epochs=50, shuffle=True, validation_data=(x_val.T, y_val.T))
model.save('../models/'+model_name+'.h5')
