import numpy as np
import tensorflow as tf
import itertools
import time

def get_input_fn(data_set, num_epochs = None, shuffle = True):
	return tf.estimator.inputs.numpy_input_fn(
		x = {
		'METx':  np.array(data_set[:,1]), 
		'METy':  np.array(data_set[:,2]),
		'E1':  np.array(data_set[:,3]),
		'px1':  np.array(data_set[:,4]),
		'py1':  np.array(data_set[:,5]),
		'pz1':  np.array(data_set[:,6]),
		'E2':  np.array(data_set[:,7]),
		'px2':  np.array(data_set[:,8]),
		'py2':  np.array(data_set[:,9]),
		'pz2':  np.array(data_set[:,10]),
                'cov00': np.array(data_set[:,11]),
                'cov01': np.array(data_set[:,12]),
                'cov10': np.array(data_set[:,13]),
                'cov11': np.array(data_set[:,14])
		},
		y = np.array(data_set[:,0]),
		num_epochs = num_epochs,
		shuffle = shuffle 
		)

def predict_mass(layers, steps, name):

	traindata	= np.load('../Data/traindata.npy')
	valdata		= np.load('../Data/valdata.npy')
	testdata	= np.load('../Data/testdata.npy')

	COLUMNS		= ['mH', 'METx', 'METy', 'E1', 'px1', 'py1', 'pz1', 'E2', 'px2', 'py2', 'pz2', 'cov00', 'cov01', 'cov10', 'cov11']
	FEATURES	= ['METx', 'METy', 'E1', 'px1', 'py1', 'pz1', 'E2', 'px2', 'py2', 'pz2', 'cov00', 'cov01', 'cov10', 'cov11']
	LABEL		= 'mH'

#	start = time.perf_counter()

	feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

	regressor	= tf.estimator.DNNRegressor(
	feature_columns = feature_cols, 
	hidden_units = layers, 
	model_dir=name+"/dane", 
	activation_fn=tf.nn.elu,
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, beta1=0.9, beta2=0.999), 
	config=tf.estimator.RunConfig().replace(save_summary_steps=100) )

	regressor.train(input_fn = get_input_fn(traindata, num_epochs = None), steps = steps)

	ev = regressor.evaluate(input_fn = get_input_fn(valdata, num_epochs = 1, shuffle = False) )

	loss_score = ev['loss']
	print('Loss: {0:f}'.format(loss_score)  )

	y = regressor.predict(input_fn = get_input_fn(testdata, num_epochs = 1, shuffle = False)  )


	end = -1 #time.perf_counter() - start


	z = regressor.predict(input_fn = get_input_fn(traindata, num_epochs = 1, shuffle = False)  )

	#predictions = list( np.asscalar(p['predictions']) for p in itertools.islice(y,6) )
	#print('Input: {}'.format(str(testdata[:6] ) ) )
	#print('Predictions: {}'.format(str(predictions ) ) )

	predicted_mass_train 	= list( np.asscalar(p['predictions']) for p in z)
	actual_mass_train	 	= traindata[:,0]
	mass_difference_train	= np.abs(predicted_mass_train - actual_mass_train)
	norm_squares_train		= np.sum(mass_difference_train**2)/np.shape(mass_difference_train)[0]


	predicted_mass_test 	= list( np.asscalar(p['predictions']) for p in y)
	actual_mass_test	 	= testdata[:,0]
	mass_difference_test	= np.abs(predicted_mass_test - actual_mass_test)
	norm_squares_test		= np.sum(mass_difference_test**2)/np.shape(mass_difference_test)[0]

	np.save(name + '/predicted_mass_test', predicted_mass_test)
	np.save(name + '/actual_mass_test', actual_mass_test)	
	np.save(name + '/predicted_mass_train', predicted_mass_train)
	np.save(name + '/actual_mass_train', actual_mass_train)	

	return predicted_mass_test, actual_mass_test, predicted_mass_train, actual_mass_train, end, loss_score

if __name__ == '__main__':
	
	predicted_mass_test, actual_mass_test, predicted_mass_train, actual_mass_train, end = predict_mass([10,10], 10000, '10lay_10nodes1')

