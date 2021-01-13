import torch
from torch import nn as nn
import sys
sys.path.extend(['E:\\zhiyangwang\\common\\NGCF-pytorch-master'])
from toyDataset.loaddata import load100KRatingstrain
from toyDataset.loaddata import load100KRatingstest
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from numpy import diag
from GraphNCF.GCFmodel import GCF
from torch.utils.data import DataLoader
from GraphNCF.dataPreprosessing import ML1K
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import MSELoss
from GraphNCF.GCFmodel import SVD
from GraphNCF.GCFmodel import NCF


#
# rtIt = rt['itemId'] + userNum
# uiMat = coo_matrix((rt['rating'],(rt['userId'],rt['itemId'])))
# uiMat_upperPart = coo_matrix((rt['rating'],(rt['userId'],rtIt)))
# uiMat = uiMat.transpose()
# uiMat.resize((itemNum,userNum+itemNum))
# uiMat = uiMat.todense()
# uiMat_t = uiMat.transpose()
# zeros1 = np.zeros((userNum,userNum))
# zeros2 = np.zeros((itemNum,itemNum))
#
# p1 = np.concatenate([zeros1,uiMat],axis=1)
# p2 = np.concatenate([uiMat_t,zeros2],axis=1)
# mat = np.concatenate([p1,p2])
#
# count = (mat > 0)+0
# diagval = np.array(count.sum(axis=0))[0]
# diagval = np.power(diagval,(-1/2))
# D_ = diag(diagval)
#
# L = np.dot(np.dot(D_,mat),D_)
#
para = {
    'epoch':60,
    'lr':0.01,
    'batch_size':2048,
    'train':0.8
}
sum_mse=0
for idx in range(5):
    rt = load100KRatingstrain(idx+1)

    userNum = rt['userId'].max()
    itemNum = rt['itemId'].max()

    rt['userId'] = rt['userId'] - 1
    rt['itemId'] = rt['itemId'] - 1
    train = ML1K(rt)

    dl = DataLoader(train,batch_size=para['batch_size'],shuffle=True,pin_memory=True)
    model = GCF(userNum, itemNum, rt, 80, layers=[80,80,]).cuda()
    
    #print(model)

    # model = SVD(userNum,itemNum,50).cuda()
    # model = NCF(userNum,itemNum,64,layers=[128,64,32,16,8]).cuda()
    optim = Adam(model.parameters(), lr=para['lr'],weight_decay=0.001)
    lossfn = MSELoss()
    for i in range(para['epoch']):

        for id,batch in enumerate(dl):
            #print('epoch:',i,' batch:',id)
            optim.zero_grad()
            prediction = model(batch[0].cuda(), batch[1].cuda())
            loss = lossfn(batch[2].float().cuda(),prediction)
            loss.backward()
            optim.step()   
        print('epoch:',i,' loss:',loss)
    
    rt_test = load100KRatingstest(idx+1)

    userNum = rt_test['userId'].max()
    itemNum = rt_test['itemId'].max()

    rt_test['userId'] = rt_test['userId'] - 1
    rt_test['itemId'] = rt_test['itemId'] - 1
    test = ML1K(rt_test)

    testdl = DataLoader(test,batch_size=len(test),)
    for data in testdl:
        prediction = model(data[0].cuda(),data[1].cuda())

    lossfn = torch.nn.MSELoss(reduction='sum')
    rmse = lossfn(data[2].float().cuda(),prediction)/prediction.size(0)
    sum_mse=sum_mse+rmse
    print('MSE on cv_{0}_df:{1}'.format(idx+1,rmse))

print('Average MSE:{0}'.format(sum_mse/5))
