import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
import analyzer as an
import training_history as my_callbacks

#----- load data, x = (METx, METy, covariance matrix, visMass) -------------
traindata = np.load('../Data/traindata.npy')
valdata = np.load('../Data/valdata.npy')

x_train = traindata[:,1:8]
y_train = traindata[:,0:1]
x_val = valdata[:,1:8]
y_val = valdata[:,0:1]

#---- loops over hyperparameters --------------------------
lays = 10		# max number of layers = 4 + 10*lays
epochs = 11		# max number of epochs = 1 + 10*epochs
inits = 5
nepochs = 10

#for nepochs in range (10,epochs):
for ninits in range (0, inits):
  for nlays in range (0, lays):
    loss=45000
    while (loss>22000):
      print('\n\nNlayers: %d, nepochs: %d, iniatialized %d. time.\n\n' %(4+10*nlays, 1+10*nepochs, ninits+1) )

      #----- Model name --------------------------
      model_name = 'lays_'+str(4+10*nlays)+'_epochs_'+str(1+10*nepochs)+'_init_'+str(ninits+1)

      #----- create model ------------------------
      model = Sequential()
      model.add(Dense(10,input_dim=7, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
      for i in range (0,4+10*nlays):
        model.add(Dense(10, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
      model.add(Dense(1, activation='softplus', kernel_initializer='random_uniform', bias_initializer='zeros'))

      #----- prepare callback initializer ---------------
      histories = my_callbacks.Histories()
      earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto', baseline=None)
      callback_list = [earlystop, histories]

      #----- compile train and save model ---------------
      model.compile(loss='mse', optimizer='adagrad')
      model.fit(x_train, y_train, epochs=1+10*nepochs, shuffle=True, validation_data=(x_val, y_val), callbacks=callback_list, verbose=1)
      loss=histories.losses[0]

    model.save('../models/'+model_name+'.h5')

    #----- save epochs validation losses --------------------------------
    losses_fname = '../Losses/' + model_name
    np.save(losses_fname, histories.losses)
    print(histories.losses)

    #----- plots --------------------------------------
    an.plots(model_name, False)


