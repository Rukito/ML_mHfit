from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os

def plots(model_name, show=False):

  #------ Name and load model ------------------------------
  fig_png_path = '../fig_png/' + model_name + '/'
  model = load_model('../models/'+model_name+'.h5')
  if not os.path.exists(fig_png_path):
    os.makedirs(fig_png_path)

  #----- load & prepare data ---------------------------
  traindata = np.load('../Data/traindata.npy')
  testdata = np.load('../Data/testdata.npy')
  valdata = np.load('../Data/valdata.npy')

  x_train = traindata[:,1:8]
  y_train = traindata[:,0:1]
  x_test = testdata[:,1:8]
  y_test = testdata[:,0:1]
  x_val = valdata[:,1:8]
  y_val = valdata[:,0:1]

  #------ Model evaluation -----------------------------
  predicted_mass_test = model.predict(x_test)
  predicted_mass_train = model.predict(x_train)

  np.save('../Data/SVfit_mass', y_test)
  np.save('../Data/siec_mass', predicted_mass_test)

  mass_diff_test = y_test - predicted_mass_test
  mass_diff_train = y_train - predicted_mass_train
  mean_sq_test = mass_diff_test**2
  mean_sq_train = mass_diff_train**2
  rel_mass_diff_train = mass_diff_train/y_train
  rel_mass_diff_test = mass_diff_test/y_test

  #------ Mass histo -------------------
  plt.hist(predicted_mass_test, bins = 450, range = (0,450), label = 'predicted', alpha = 0.5, color = 'r')
  plt.hist(y_test, bins = 450, range = (0,450), label = 'actual', alpha = 0.3, color = 'b')
  #plt.title('Mass histogram')
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
  #plt.title('Mass difference')
  plt.xlabel('Actual mass')
  plt.ylabel('Mass difference')
  plt.legend()
  plt.savefig(fig_png_path + 'keras_mass_diff_range.png')
  if(show=='True'):
    plt.show()
  plt.clf()

  #--------- Mass difference scattered plot -------------
  plt.scatter(y_train, mass_diff_train, alpha = 0.25, color = 'r', label = 'train', s = 5)
  plt.scatter(y_test, mass_diff_test, alpha = 1, color = 'b', label = 'test', s = 5)
  #plt.title('Mass difference')
  plt.xlabel('Actual mass')
  plt.ylabel('Mass difference')
  plt.legend()
  plt.savefig(fig_png_path + 'keras_mass_diff.png')
  if(show=='True'):
    plt.show()
  plt.clf()

  #-------- Squared mass difference histo ----------------
  plt.hist(mean_sq_test, bins = 3000, range = (0,3), label = 'test', alpha = 0.7, color = 'r', log='True')
  #plt.hist(mean_sq_train.T, bins = 100, range = (0,10), label = 'train', alpha = 0.3, color = 'b', log='True')
  #plt.title('Mass difference squared')
  plt.xlabel('Mass difference squared')
  plt.ylabel('Count')
  plt.legend()
  plt.savefig(fig_png_path + 'keras_sq_mass_error.png')
  if(show=='True'):
    plt.show()
  plt.clf()

  #--------- Relative mass difference scattered plot -------------
  plt.scatter(y_train, rel_mass_diff_train, alpha = 0.25, color = 'r', label = 'train', s = 5)
  plt.scatter(y_test, rel_mass_diff_test, alpha = 1, color = 'b', label = 'test', s = 5)
  #plt.title('Relative mass difference')
  plt.xlabel('Actual mass')
  plt.ylabel('Relative mass difference')
  plt.legend()
  plt.savefig(fig_png_path + 'keras_relative_mass_diff.png')
  if(show=='True'):
    plt.show()
  plt.clf()

  #--------- Relative mass difference scattered plot (range 0<-->200) -------------
  plt.scatter(y_train, rel_mass_diff_train, alpha = 0.25, color = 'r', label = 'train', s = 5)
  plt.scatter(y_test, rel_mass_diff_test, alpha = 1, color = 'b', label = 'test', s = 5)
  plt.xlim(0,200)
  #plt.title('Relative mass difference')
  plt.xlabel('Actual mass')
  plt.ylabel('Relative mass difference')
  plt.legend()
  plt.savefig(fig_png_path + 'keras_relative_mass_diff.png')
  if(show=='True'):
    plt.show()
  plt.clf()


if __name__ == '__main__':
        plots('lays_4_epochs_101_init_0', True)

