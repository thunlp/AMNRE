from __future__ import print_function
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from constant import *
from torch import optim
from models import *
class Trainer(object):
    def __init__(self):
        wordVec_en=np.load(dataPath+dataName+"newVec_en.npy")
        wordVec_zh=np.load(dataPath+dataName+"newVec_zh.npy")
        self.model=MARE(wordVec_en,wordVec_zh)
        self.D_optim=optim.SGD(self.model.D.parameters(),lr=dis_lr)
        self.encoder_optim=optim.SGD(self.model.share_encoder.parameters(),lr=encoder_lr)
        self.optim=optim.SGD(self.model.parameters(),lr=RE_lr)
        self.mono_optim=optim.SGD(self.model.monoRE.parameters(),lr=RE_lr)
    def get_dis_xy(self,wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh):
        encoded_en,encoded_zh=self.model.share_encoder(wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh)
        Len_en=encoded_en.size(0)
        Len_zh=encoded_zh.size(0)
        x=torch.cat((encoded_en,encoded_zh),0)
        y=torch.FloatTensor(Len_en+Len_zh).zero_()
        y[:Len_en]=dis_smooth
        y[Len_en:]=1.0-dis_smooth
        y=Variable(y.cuda())
        return x,y
    def train_dis(self,wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh):
        if dis_lambda==0:
            return
        self.model.D.train()
        self.model.share_encoder.eval()
        x,y=self.get_dis_xy(wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh)
        preds=self.model.D(x)
        loss=F.binary_cross_entropy(preds,y)
        if (loss!=loss).data.any():
            print("NaN Loss (discriminator)")
            exit()
        self.D_optim.zero_grad()
        loss.backward()
        self.D_optim.step()
        return loss.data[0]
    def fool_dis(self,wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh):
        if dis_lambda==0:
            return
        self.model.D.eval()
        self.model.share_encoder.train()
        x,y=self.get_dis_xy(wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh)
        pred=self.model.D(x)
        loss=F.binary_cross_entropy(pred,1-y)
        loss=dis_lambda*loss
        if (loss!=loss).data.any():
            print("NaN Loss (fooling discriminator)")
            exit()
        self.encoder_optim.zero_grad()
        loss.backward()
        self.encoder_optim.step()
        return loss.data[0]
    def set_train(self):
        self.model.train()
    def set_mono_train(self):
        self.model.train()
        self.model.multiRE.eval()
    def set_eval(self):
        self.model.eval()
    def train_RE(self,wordsEn,pos1En,pos2En,rEn,lEn,wordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask):
        self.model.train()
        pred=self.model(wordsEn,pos1En,pos2En,rEn,lEn,wordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask)
        loss=-torch.sum(pred.view(lEn.size(0)))+Orth_Coef*self.model.Orth_con(wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh)
        if (loss!=loss).data.any():
            print("NaN Loss (training RE)")
            exit()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.data[0]
    def train_RE_mono(self,wordsEn,pos1En,pos2En,rEn,lEn,wordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask):
        pred=self.model(wordsEn,pos1En,pos2En,rEn,lEn,wordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask)
        loss=-torch.sum(pred.view(lEn.size(0)))
        if (loss!=loss).data.any():
            print("NaN Loss (training RE)")
            exit()
        self.mono_optim.zero_grad()
        loss.backward()
        self.mono_optim.step()
        return loss.data[0]
    def eval(self,wordsEn,pos1En,pos2En,rEn,lEn,wordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask):
        pred=self.model(wordsEn,pos1En,pos2En,rEn,lEn,wordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask)
        return pred
    def save(self,Tag):
        torch.save(self.model.state_dict(),"model%s.tar"%(Tag))
    def load(self,Tag):
        self.model.load_state_dict(torch.load("model%s.tar"%(Tag)))
