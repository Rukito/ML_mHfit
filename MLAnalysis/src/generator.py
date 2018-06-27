import numpy as np
import pickle

def generate():

    with open("../Data/raw_data.txt", "rb") as file:
        raw_data = pickle.load(file)

    with open("../Data/legs_list.txt", "rb") as file2:
        legs = pickle.load(file2)

    datasize = len(raw_data[0])
    trainsize = int( datasize*7/10 )
    valsize = int( datasize*2/10 )
    testsize = datasize - trainsize - valsize
    print(trainsize, valsize, testsize)

    mH = []
    METx = []
    METy = []
    E1 = []
    px1 = []
    py1 = []
    pz1 = []
    E2 = []
    px2 = []
    py2 = []
    pz2 = []
    cov00 = []
    cov01 = []
    cov10 = []
    cov11 = []


    for j in range (0, datasize):
        if(j%1000==0):
            print(j," / ", datasize)
        mH =    np.insert(mH, len(mH), raw_data[0][j])
        METx =	np.insert(METx, len(METx), raw_data[1][j])
        METy =	np.insert(METy, len(METy), raw_data[2][j])
        E1 = 	np.insert(E1, len(E1), legs[0][0][j])
        px1 =	np.insert(px1, len(px1), legs[0][1][j])
        py1 =	np.insert(py1, len(py1), legs[0][2][j])
        pz1 =	np.insert(pz1, len(pz1), legs[0][3][j])
        E2 =	np.insert(E2, len(E2), legs[1][0][j])
        px2 =	np.insert(px2, len(px2), legs[1][1][j])
        py2 =	np.insert(py2, len(py2), legs[1][2][j])
        pz2 =	np.insert(pz2, len(pz2), legs[1][3][j])
        cov00 =np.insert(cov00, len(cov00), raw_data[3][j])
        cov01 =np.insert(cov11, len(cov01), raw_data[4][j])
       	cov10 =np.insert(cov11, len(cov10), raw_data[5][j])
        cov11 =np.insert(cov11, len(cov11), raw_data[6][j])

    data        = np.stack([mH,METx,METy, E1, px1, py1, pz1, E2, px2, py2, pz2, cov00, cov01, cov10, cov11], axis = -1)
    traindata   = data[:trainsize, :]
    valdata     = data[trainsize:trainsize+valsize, :]
    testdata    = data[trainsize+valsize:, :]
    return traindata, valdata, testdata

if __name__ == '__main__':
    traindata, valdata, testdata = generate()
    np.save('../Data/traindata', traindata)
    np.save('../Data/valdata',   valdata)
    np.save('../Data/testdata',  testdata)

