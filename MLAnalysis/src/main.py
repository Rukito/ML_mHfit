import predict
import analize
import matplotlib.pyplot as plt

sets1 = {
'20lay_10nodes1': 	[10]*20,
'20lay_10nodes2': 	[10]*20,
'20lay_10nodes3':	[10]*20,
}

sets2 = {
'30lay_10nodes1':       [10]*30,
'30lay_10nodes2':       [10]*30,
'30lay_10nodes3':       [10]*30,
}

'''
sets3 = {
'40lay_10nodes1':       [10]*40,
'40lay_10nodes2':       [10]*40,
'40lay_10nodes3':       [10]*40,
}

sets4 = {
'50lay_10nodes1':       [10]*50,
'50lay_10nodes2':       [10]*50,
'50lay_10nodes3':       [10]*50,
}

sets5 = {
'60lay_10nodes1':       [10]*60,
'60lay_10nodes2':       [10]*60,
'60lay_10nodes3':       [10]*60,
}
'''

models_list = []
models_list.append(sets1)
models_list.append(sets2)

for sets in models_list:
    for i in range (2,4):
        steps = 10**i
        for name, set_ in sets.items():
        	nodes  = set_[0]
        	layers = len(set_)

        	print('\nname: ', name, '\tsteps: ', steps)
        	predicted_mass_test, actual_mass_test, predicted_mass_train, actual_mass_train, time, loss_score = 	predict.predict_mass(set_, steps, name)

        	RMSD_test, RMSD_train = analize.plot_partial(predicted_mass_test, actual_mass_test, predicted_mass_train, actual_mass_train, name, steps, show = False)

        	with open ('results.txt', 'a') as f:
        		f.write('{:14} | layers: {:<2d} | nodes: {:<2d} | steps: {:<7d} | loss: {:<10.3f} | RMSD_test: {:.3f} | RMSD_train: {:.3f} | time: {:.3f} \n'.format(name, layers, nodes, steps, loss_score, RMSD_test, RMSD_train, time) )
