import numpy as np
import pickle

def generate():

    with open("../Data/raw_data.txt", "rb") as file:
        raw_data = pickle.load(file)

    datasize = len(raw_data[0])
    trainsize = int( datasize*6/10 )
    valsize = int( datasize*3/10 )
    testsize = datasize - trainsize - valsize

    METx = []
    METy = []
    #cov00 = []
    #cov11 =[]
    mH = []

    for j in range (0, datasize):
        if(j%1000==0):
            print(j," / ", datasize)
        METx = np.insert(METx, len(METx), raw_data[1][j])
        METy = np.insert(METy, len(METy), raw_data[2][j])
        #cov00 = np.insert(cov00, len(cov00), raw_data[3][j])
        #cov11 = np.insert(cov11, len(cov11), raw_data[4][j])
        mH = np.insert(mH, len(mH), raw_data[0][j])

    data        = np.stack([mH,METx,METy], axis = -1)
    traindata   = data[:trainsize, :]
    valdata     = data[trainsize:trainsize+valsize, :]
    testdata    = data[trainsize+valsize:, :]
    return traindata, valdata, testdata

if __name__ == '__main__':
    traindata, valdata, testdata = generate()
    np.save('../Data/traindata', traindata)
    np.save('../Data/valdata',   valdata)
    np.save('../Data/testdata',  testdata)

