from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

#------ Name and load model ------------------------------
model_name = 'model_50epochs'
fig_png_path = '../fig_png/Keras_model/' + model_name + '/'
model = load_model('../models/'+model_name+'.h5')
show = 'False'

#----- load & prepare data ---------------------------
traindata = np.load('../Data/traindata.npy')
testdata = np.load('../Data/testdata.npy')
valdata = np.load('../Data/valdata.npy')

x_train = np.vstack((traindata.T[0:2], traindata.T[11:15]))
y_train = traindata[:,0]
x_test = np.vstack((testdata.T[0:2], testdata.T[11:15]))
y_test = testdata[:,0]
x_val = np.vstack((valdata.T[0:2], valdata.T[11:15]))
y_val = valdata[:,0]

#------ Model evaluation -----------------------------
predicted_mass_test = model.predict(x_test.T)
predicted_mass_train = model.predict(x_train.T)
mass_diff_test = y_test.T - predicted_mass_test.T
mass_diff_train = y_train.T - predicted_mass_train.T
mean_sq_test = mass_diff_test**2
mean_sq_train = mass_diff_train**2

#------ Mass histo -------------------
plt.hist(predicted_mass_test, bins = 450, range = (0,450), label = 'predicted', alpha = 0.5, color = 'r')
plt.hist(y_test.T, bins = 450, range = (0,450), label = 'actual', alpha = 0.3, color = 'b')
plt.title('Mass histogram')
plt.xlabel('Mass')
plt.ylabel('Count')
plt.legend()
plt.savefig(fig_png_path + 'keras_mass.png')
if(show=='True'):
  plt.show()
plt.clf()

#------ Mass difference scattered plot (range 0<-->200) -------
plt.scatter(y_train, mass_diff_train, alpha = 0.25, color = 'r', label = 'train', s = 5)
plt.scatter(y_test, mass_diff_test, alpha = 1, color = 'b', label = 'test', s = 5)
plt.xlim(0,200)
plt.title('Mass difference')
plt.xlabel('Actual mass')
plt.ylabel('Mass difference')
plt.legend()
plt.savefig(fig_png_path + 'keras_mass_diff.png')
if(show=='True'):
  plt.show()
plt.clf()

#--------- Mass difference scattered plot -------------
plt.scatter(y_train, mass_diff_train, alpha = 0.25, color = 'r', label = 'train', s = 5)
plt.scatter(y_test, mass_diff_test, alpha = 1, color = 'b', label = 'test', s = 5)
plt.title('Mass difference')
plt.xlabel('Actual mass')
plt.ylabel('Mass difference')
plt.legend()
plt.savefig(fig_png_path + 'keras_mass_diff.png')
if(show=='True'):
  plt.show()
plt.clf()

#-------- Squared mass difference histo ----------------
plt.hist(mean_sq_test.T, bins = 100, range = (0,100), label = 'test', alpha = 0.7, color = 'r', log='True')
plt.hist(mean_sq_train.T, bins = 100, range = (0,100), label = 'train', alpha = 0.3, color = 'b', log='True')
plt.title('Mass difference squared histogram')
plt.xlabel('Mass difference')
plt.ylabel('Count')
plt.legend()
plt.savefig(fig_png_path + 'keras_sq_mass_error.png')
if(show=='True'):
  plt.show()
plt.clf()

