import predict
import analize
import matplotlib.pyplot as plt

'''
sets = {
'20lay_10nodes1': 	[10]*20,
'20lay_10nodes2': 	[10]*20,
'20lay_10nodes3': 	[10]*20,
'20lay_10nodes4': 	[10]*20,
'20lay_10nodes5': 	[10]*20,
'20lay_10nodes6': 	[10]*20,
'20lay_10nodes7': 	[10]*20,
'20lay_10nodes8': 	[10]*20,
'20lay_10nodes9': 	[10]*20
}'''
'''
sets = {
'30lay_10nodes1': 	[10]*30,
'30lay_10nodes2': 	[10]*30,
'30lay_10nodes3': 	[10]*30,
'30lay_10nodes4': 	[10]*30,
'30lay_10nodes5': 	[10]*30,
'30lay_10nodes6': 	[10]*30,
'30lay_10nodes7': 	[10]*30,
'30lay_10nodes8': 	[10]*30,
'30lay_10nodes9': 	[10]*30,
'20lay_15nodes1': 	[15]*20,
'20lay_15nodes2': 	[15]*20,
'20lay_15nodes3': 	[15]*20,
'20lay_15nodes4': 	[15]*20,
'20lay_15nodes5': 	[15]*20,
'20lay_15nodes6': 	[15]*20,
'20lay_15nodes7': 	[15]*20,
'20lay_15nodes8': 	[15]*20,
'20lay_15nodes9': 	[15]*20
}'''

sets = {
'20lay_10nodes1': 	[10]*20,
'20lay_10nodes2': 	[10]*20,
'20lay_10nodes3': 	[10]*20,
'20lay_10nodes4': 	[10]*20,
'20lay_10nodes5': 	[10]*20,
'20lay_10nodes6': 	[10]*20,
'20lay_10nodes7': 	[10]*20,
'20lay_10nodes8': 	[10]*20,
'20lay_10nodes9': 	[10]*20

}

list_label = []
list_sq_test = []
list_sq_train = []
steps = 100


for name, set_ in sets.items():
	nodes  = set_[0]
	layers = len(set_)
	print(nodes)

	print('\nname: ', name, ' set: ', set_,'\n')
	predicted_mass_test, actual_mass_test, predicted_mass_train, actual_mass_train, time, loss_score = 	predict.predict_mass(set_, steps, name)
	
	RMSD_test, RMSD_train = analize.plot_partial(
	predicted_mass_test, actual_mass_test, predicted_mass_train, actual_mass_train, name, show = False
	)
	
	with open ('results.txt', 'a') as f:
		f.write('{:14} | layers: {:<2d} | nodes: {:<2d} | steps: {:<7d} | loss: {:<10.3f} | RMSD_test: {:.3f} | RMSD_train: {:.3f} | time: {:.0f} \n'.format(name, layers, nodes, steps, loss_score, RMSD_test, RMSD_train, time) )
		
	
	

