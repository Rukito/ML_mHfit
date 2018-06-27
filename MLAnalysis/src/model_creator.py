import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import analyzer as an
import training_history as my_callbacks
from keras.utils import np_utils

#----- prepare data ------------------------
traindata = np.load('../Data/traindata.npy')
valdata = np.load('../Data/valdata.npy')

x_train = np.vstack((traindata.T[1:3], traindata.T[11:15]))
y_train = traindata[:,0]
x_val = np.vstack((valdata.T[1:3], valdata.T[11:15]))
y_val = valdata[:,0]

x_train = x_train.T
x_val = x_val.T
y_train = y_train.T
y_val = y_val.T

lays = 1
epochs = 1
iters = 5

for nepochs in range (0,epochs):
  for ninits in range (0, iters):
    for nlays in range (0, lays):

      print('\n\nNlayers: ', 4+10*nlays, ', nepochs: ', 1+10*nepochs, ', iniatialized ', ninits+1, '. time.\n\n')

      #----- Model name --------------------------
      model_name = 'lay_'+str(4+10*nlays)+'_epochs_'+str(1+10*nepochs)+'_iter_'+str(ninits+1)

      #----- create model ------------------------
      model = Sequential()
      model.add(Dense(10,input_dim=6, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
      #for i in range (0,4+10*nlays):
      #  model.add(Dense(10, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
      model.add(Dense(1, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros'))

      #----- prepare callback initializer ---------------
      histories = my_callbacks.Histories()

      #----- compile train and save model ---------------
      model.compile(loss='mean_squared_error', optimizer='adagrad')
      model.fit(x_train, y_train, epochs=10+10*nepochs, shuffle=True, validation_data=(x_val, y_val), callbacks=[histories], verbose=1)
      model.save('../models/'+model_name+'.h5')

      #----- save losses --------------------------------
      losses_fname = '../Losses/' + model_name
      np.save(losses_fname, histories.losses)
      print(histories.losses)

      #----- plots --------------------------------------
      an.plots(model_name, False)

