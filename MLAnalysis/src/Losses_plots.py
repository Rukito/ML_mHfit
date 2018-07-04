import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

markers = ["o",">"]
colors = cm.rainbow(np.linspace(0, 7, 3*5))

for iter in range (0, 4):
  for nlay in range (0, 3):

    #----- load & prepare data ---------------------------
    losses = np.load('../Losses/lays_'+str(4+10*nlay)+'_epochs_101_init_'+str(iter)+'.npy')
    print('../Losses/lays_'+str(4+10*nlay)+'_epochs_101_init_'+str(iter)+'.npy')
    print(losses, '\n\n')
    epoch = np.array([])
    for i in range (0, losses.size):
      epoch = np.insert(epoch, len(epoch), i)

    #------ Losses plot -------------------
    if(iter%2==0):
      plt.scatter(epoch, losses, label = str(4+10*nlay)+' lays, relu', alpha = 0.8, color = colors[nlay], marker=markers[0])
    else:
      plt.scatter(epoch, losses, label = str(4+10*nlay)+' lays, softplus', alpha = 0.8, color = colors[nlay], marker=markers[1])
    plt.xlabel('Epoch number')
    plt.ylabel('Loss score')
    #plt.yscale('log')
    #plt.ylim(700,800)

#---- Create custom legend --------------------------
recs = []
labels = ['4 layers', '14 layers', '24 layers']
#labels = ['10epochs', '20epochs', '30 epochs', '40 epochs', '50 epochs', '60 epochs', '70 epochs']
for i in range (0,6):
  recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
plt.legend(recs, labels)

plt.savefig('../fig_png/Losses.png')
if(True):
  plt.show()
plt.clf()


#print(losses)

