import predict
import analize
import matplotlib.pyplot as plt

sets = {
'20lay_10nodes1': 	[10]*20,
'20lay_10nodes2': 	[10]*20,
}

list_label = []
list_sq_test = []
list_sq_train = []
steps = 1000


for name, set_ in sets.items():
	nodes  = set_[0]
	layers = len(set_)

	print('\nname: ', name)
	predicted_mass_test, actual_mass_test, predicted_mass_train, actual_mass_train, time, loss_score = 	predict.predict_mass(set_, steps, name)

	RMSD_test, RMSD_train = analize.plot_partial(
	predicted_mass_test, actual_mass_test, predicted_mass_train, actual_mass_train, name, show = False
	)

	with open ('results.txt', 'a') as f:
		f.write('{:14} | layers: {:<2d} | nodes: {:<2d} | steps: {:<7d} | loss: {:<10.3f} | RMSD_test: {:.3f} | RMSD_train: {:.3f} | time: {:.3f} \n'.format(name, layers, nodes, steps, loss_score, RMSD_test, RMSD_train, time) )
