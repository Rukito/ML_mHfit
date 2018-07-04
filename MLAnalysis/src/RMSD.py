import ROOT
import numpy as np

root_file = ROOT.TFile("../Losses/RMSD.root", "RECREATE")
tree = ROOT.TTree("tree", "tutorial")
x = np.empty((1), dtype="float32")
y = np.empty((1), dtype="float32")
tree.Branch("SVfit", x, "x/F")
tree.Branch("siec", y, "y/F")

svfit = np.load('../Data/SVfit_mass.npy')
siec = np.load('../Data/siec_mass.npy')

for i in range (0, svfit.size):
  x[0]=svfit[i]
  y[0]=siec[i]
  print(x[0], y[0])
  tree.Fill()

root_file.Write()

