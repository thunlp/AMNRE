from __future__ import print_function
import os
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from src.trainer import Trainer
from src.dataloader import *
from src.utils import *
from src.constant import *
from sklearn.metrics import average_precision_score

torch.manual_seed(2333333)
np.random.seed(2333333)
#os.environ['CUDA_VISIBLE_DEVICES']='6'
trainer=Trainer()

trainSet=dataset(dataName+"Train_")
validSet=dataset(dataName+"Valid_")
testSet=dataset(dataName+"Test_")
Store=True
StoreTime=1
OutRes=True
Nick="AMNRE_RNN"
def train_AE():
    print("Start Training AutoEncoder")
    for e in range(0,epoch_AE):
        print("EPOCH %d:"%(e))
        trainer.set_train()
        batch_cnt=0
        for wordVec_en,NwordVec_en,pos1_en,pos2_en,ep_en,r_en,l_en,wordVec_zh,NwordVec_zh,pos1_zh,pos2_zh,ep_zh,r_zh,l_zh,re,e1,e2 in trainSet.batchs():
            batch_cnt+=1
            AE_loss=trainer.train_AE(NwordVec_en,pos1_en,pos2_en,NwordVec_zh,pos1_zh,pos2_zh)
            print("%s Batch %d: %.3f"%(currentTime(),batch_cnt,AE_loss))
        test(testSet,"Test",e)
    print("Stop Train")
def train_all(bias=0):
    print("Start Training All")
    for e in range(0,epoch_All):
        print("EPOCH %d:"%(e))
        trainer.set_train()
        batch_cnt=0
        for wordVec_en,NwordVec_en,pos1_en,pos2_en,ep_en,r_en,l_en,wordVec_zh,NwordVec_zh,pos1_zh,pos2_zh,ep_zh,r_zh,l_zh,re,e1,e2 in trainSet.batchs():
            batch_cnt+=1
            re_mask=torch.cuda.ByteTensor(len(re),1,dimR)
            for i,Re in enumerate(re):
                re_sel=[(0,Re)]
                re_mask[i]=get_mask(re_sel)
            for i in range(D_Times):
                dis_loss=trainer.train_dis(NwordVec_en,pos1_en,pos2_en,NwordVec_zh,pos1_zh,pos2_zh)
            for i in range(G_Times):
                fool_loss=trainer.fool_dis(NwordVec_en,pos1_en,pos2_en,NwordVec_zh,pos1_zh,pos2_zh)
            AE_loss,train_loss=trainer.train_RE(wordVec_en,NwordVec_en,pos1_en,pos2_en,r_en,l_en,wordVec_zh,NwordVec_zh,pos1_zh,pos2_zh,r_zh,l_zh,re_mask)
            print("%s Batch %d: %.3f %.3f %.3f %.3f"%(currentTime(),batch_cnt,AE_loss,dis_loss,fool_loss,train_loss))
        acc=test(testSet,"Test",e+bias)
        trainer.decay(acc)
    print("Stop Train")
def test(dataSet,name,Time):
    trainer.set_eval()
    fout=open("AUC.txt","a")
    print("Testing %s"%(currentTime()))
    if Store and Time%StoreTime==0:
        trainer.save(str(Time))
    stack_label=[]
    stack_pred=[]
    for wordVec_en,NwordVec_en,pos1_en,pos2_en,ep_en,r_en,l_en,wordVec_zh,NwordVec_zh,pos1_zh,pos2_zh,ep_zh,r_zh,l_zh,re,e1,e2\
        in dataSet.batchs():
        r_en=np.array([[r for i in range(0,pos1_en.size(0))] for r in range(0,dimR)])
        r_en=Variable(torch.cuda.LongTensor(r_en))
        r_zh=np.array([[r for i in range(0,pos1_zh.size(0))] for r in range(0,dimR)])
        r_zh=Variable(torch.cuda.LongTensor(r_zh))
        re_sel=[(i,i) for i in range(0,dimR)]
        re_mask=torch.cuda.ByteTensor(len(re),dimR,dimR)
        label=[]
        for i in range(0,len(re)):
            re_mask[i]=get_mask(re_sel)
            label.append(re[i])
        label_=np.zeros((len(re),dimR))
        label_[np.arange(len(re)), label]=1
        pred=trainer.eval(wordVec_en,NwordVec_en,pos1_en,pos2_en,r_en,l_en,wordVec_zh,NwordVec_zh,pos1_zh,pos2_zh,r_zh,l_zh,re_mask)
        stack_label.append(label_)
        stack_pred.append(pred.data.cpu().numpy())
    stack_label=np.concatenate(stack_label,axis=0)
    stack_pred=np.concatenate(stack_pred,axis=0)
    exclude_na_flatten_label=np.reshape(stack_label[:,1:],(-1))
    exclude_na_flatten_pred=np.reshape(stack_pred[:,1:],(-1))
    average_precision=average_precision_score(exclude_na_flatten_label,exclude_na_flatten_pred)
    if OutRes:
        np.save("./"+Nick+"_label_"+str(Time)+".npy",exclude_na_flatten_label)
        np.save("./"+Nick+"_pred_"+str(Time)+".npy",exclude_na_flatten_pred)
    print("%s  %s result: %f"%(currentTime(),name,average_precision))
    fout.write("%f\n"%(average_precision))
    fout.close()
    return average_precision
if __name__=='__main__':
    train_all(0)
