import numpy as np
import matplotlib.pyplot as plt

def plot_partial(predicted_mass_test, actual_mass_test, predicted_mass_train, actual_mass_train, name, show = False):

	#traindata	= np.load('traindata.npy')
	#valdata		= np.load('valdata.npy')
	#testdata	= np.load('testdata.npy')

	#predicted_mass	= np.load('predicted_mass.npy')
	#actual_mass		= np.load('actual_mass.npy')
	
	mass_difference_train		= predicted_mass_train - actual_mass_train
	mass_difference_sq_train	= mass_difference_train**2
	RMSD_train					= np.sqrt(np.sum(mass_difference_sq_train)/np.shape(mass_difference_sq_train)[0])

	mass_difference_test		= predicted_mass_test - actual_mass_test
	mass_difference_sq_test		= mass_difference_test**2
	RMSD_test					= np.sqrt(np.sum(mass_difference_sq_test)/np.shape(mass_difference_sq_test)[0])

	#plt.hist(predicted_mass, bins = 20, range = (0,100), label = 'predicted', alpha = 0.5, color = 'r')
	#plt.hist(actual_mass, bins = 20, range = (0,100), label = 'actual', alpha = 0.5, color = 'b')
	#plt.title('Mass histogram')
	#plt.xlabel('Mass')
	#plt.ylabel('Count')
	#plt.legend()
	#plt.savefig(name + '/mass.png')
	#if show == True:
	#	plt.show()
	#plt.clf()

	#plt.hist(mass_difference, bins = 20, color = 'r')
	#plt.title('Mass difference histogram')
	#plt.xlabel('Mass difference')
	#plt.ylabel('Count')
	#plt.savefig(name + '/mass_diff.png')
	#if show == True:
	#	plt.show()
	#plt.clf()

	#plt.hist(mass_difference_sq, bins = 20, color = 'r')
	#plt.title('Mass difference histogram')
	#plt.xlabel('Squared mass difference')
	#plt.ylabel('Count')
	#plt.savefig(name + '/mass_diff_sq.png')
	#if show == True:
	#	plt.show()
	#plt.clf()
	
	plt.scatter(actual_mass_train, mass_difference_train, alpha = 0.25, color = 'r', label = 'train', s = 5)
	plt.scatter(actual_mass_test, mass_difference_test, alpha = 1, color = 'b', label = 'test', s = 5)
	plt.title(name+'\nMass difference scatter plot')
	plt.xlabel('Actual mass')
	plt.ylabel('Mass difference')
	plt.legend()
	plt.savefig(name + '/mass_diff.png')
	if show == True:
		plt.show()
	plt.clf()

	plt.scatter(actual_mass_train, mass_difference_train/actual_mass_train, alpha = 0.25, color = 'r', label = 'train', s = 5)
	plt.scatter(actual_mass_test, mass_difference_test/actual_mass_test, alpha = 1, color = 'b', label = 'test', s = 5)
	plt.title(name+'\nMass difference scatter plot (normalized)')
	plt.xlabel('Actual mass')
	plt.ylabel('Relative mass difference')
	plt.legend()
	plt.savefig(name + '/mass_diff_relative.png')
	if show == True:
		plt.show()
	plt.clf()
	
	plt.scatter(actual_mass_train, mass_difference_train/actual_mass_train, alpha = 0.25, color = 'r', label = 'train', s = 5)
	plt.scatter(actual_mass_test, mass_difference_test/actual_mass_test, alpha = 1, color = 'b', label = 'test', s = 5)
	plt.title(name+'\nMass difference scatter plot (normalized)')
	plt.xlabel('Actual mass')
	plt.ylabel('Relative mass difference')
	plt.xlim( (0,10) )
	plt.ylim( (-1,10,) )
	plt.legend()
	plt.savefig(name + '/mass_diff_relative_zoom.png')
	if show == True:
		plt.show()
	plt.clf()

	
	print('RMSD_test: ', RMSD_test, '\nRMSD_train: ', RMSD_train)
	return RMSD_test, RMSD_train


if __name__ == '__main__':
	main()
