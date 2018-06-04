import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def plot_partial(folder,lwarstw,nr,show = False):

	predicted_mass_train	= np.load(folder+'/predicted_mass_train.npy')
	predicted_mass_test		= np.load(folder+'/predicted_mass_test.npy')
	actual_mass_train		= np.load(folder+'/actual_mass_train.npy')
	actual_mass_test		= np.load(folder+'/actual_mass_test.npy')
	
	mass_difference_train		= predicted_mass_train - actual_mass_train
	mass_difference_sq_train	= mass_difference_train**2
	RMSD_train					= np.sqrt(np.sum(mass_difference_sq_train)/np.shape(mass_difference_sq_train)[0])

	mass_difference_test		= predicted_mass_test - actual_mass_test
	mass_difference_sq_test		= mass_difference_test**2
	RMSD_test					= np.sqrt(np.sum(mass_difference_sq_test)/np.shape(mass_difference_sq_test)[0])

	plt.scatter(actual_mass_train, mass_difference_train, alpha = 0.4, facecolors='none', edgecolors='r', label = 'dane treningowe', s = 20)
	plt.scatter(actual_mass_test, mass_difference_test, alpha = 1, c='b', marker='x', label = 'dane testowe', s = 20)
	plt.title(lwarstw+' warstw nr '+nr+'.', fontsize = 14)
	plt.xlabel('Rzeczywista masa', fontsize = 14)
	plt.ylabel('Różnica mas', fontsize = 14)
	plt.legend(fontsize = 10)
	plt.savefig('massdiff_'+lwarstw+'_'+nr+'.png')
	if show == True:
		plt.show()
	plt.clf()
	
	plt.scatter(actual_mass_train, mass_difference_train/actual_mass_train, alpha = 0.6, facecolors='none', edgecolors='r', label = 'dane treningowe', s = 30)
	plt.scatter(actual_mass_test, mass_difference_test/actual_mass_test, alpha = 1, c='b', marker='x', label = 'dane testowe', s = 30)
	plt.title(lwarstw+' warstw nr '+nr+'.', fontsize = 14)
	plt.xlabel('Rzeczywista masa', fontsize = 14)
	plt.ylabel('Znormalizowana różnica mas', fontsize = 14)
	plt.xlim( (0,2) )
#	plt.ylim( (-1,20,) )
	plt.legend(fontsize = 10)
	plt.savefig('zoom_'+lwarstw+'_'+nr+'.png')
	if show == True:
		plt.show()
	plt.clf()

if __name__ == '__main__':
	tab1 = ['20lay_10nodes1', '20lay_10nodes2', '20lay_10nodes3', '20lay_10nodes4', '20lay_10nodes5', '20lay_10nodes6', '20lay_10nodes7', '20lay_10nodes8', '20lay_10nodes9'] 
	tab2 = ['30lay_10nodes1', '30lay_10nodes2', '30lay_10nodes3', '30lay_10nodes4', '30lay_10nodes5', '30lay_10nodes6', '30lay_10nodes7', '30lay_10nodes8', '30lay_10nodes9']
	nr = ['1','2','3','4','5','6','7','8','9']

	for i in range(len(tab1)):
		plot_partial(tab1[i], '20', nr[i])
	for i in range(len(tab2)):
		plot_partial(tab2[i], '30', nr[i])
	
	
	
