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

x_train = np.vstack((traindata.T[0:2], traindata.T[11:15]))
y_train = traindata[:,0]
x_val = np.vstack((valdata.T[0:2], valdata.T[11:15]))
y_val = valdata[:,0]

nlays = 2
nepochs = 2
#for nepochs in range (0,2):
for ninits in range (0, 5):
    #for nlays in range (0, 5):

      print('\n\nNlayers: ', 4+10*nlays, ', nepochs: ', 10+10*nepochs, ', iniatialized ', ninits+1, '. time.\n\n')

      #----- Model name --------------------------
      model_name = 'lay_'+str(4+10*nlays)+'_epochs_'+str(10+10*nepochs)+'_iter_'+str(ninits)

      #----- create model ------------------------
      model = Sequential()
      model.add(Dense(10,input_dim=6, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
#      model.add(Dense(10, activation='relu'))
#      model.add(Dense(10, activation='relu'))
#      model.add(Dense(10, activation='relu'))
      for i in range (0,4+10*nlays):
        model.add(Dense(10, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
      model.add(Dense(1, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))

      #----- prepare callback initializer ---------------
      histories = my_callbacks.Histories()

      #----- compile train and save model ---------------
      model.compile(loss='mean_squared_error', optimizer='adagrad')
      model.fit(x_train.T, y_train.T, epochs=10+10*nepochs, shuffle=True, validation_data=(x_val.T, y_val.T), callbacks=[histories], verbose=0)
      model.save('../models/final'+model_name+'.h5')

      #----- save losses --------------------------------
      losses_fname = '../Losses/final' + model_name
      np.save(losses_fname, histories.losses)
      print(histories.losses)

      #----- plots --------------------------------------
      an.plots('final'+model_name, False)
