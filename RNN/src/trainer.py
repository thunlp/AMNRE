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
        NwordVec_en=np.load(dataPath+dataName+"newVec_en.npy")
        NwordVec_zh=np.load(dataPath+dataName+"newVec_zh.npy")
        self.enSz=len(NwordVec_en)
        self.zhSz=len(NwordVec_zh)
        print(self.enSz)
        print(self.zhSz)
        weight_mask=torch.ones(self.enSz).cuda()
        weight_mask[0]=0#UNK?PAD?
        self.en_criterion=nn.CrossEntropyLoss(weight=weight_mask).cuda()
        weight_mask=torch.ones(self.zhSz).cuda()
        weight_mask[0]=0
        self.zh_criterion=nn.CrossEntropyLoss(weight=weight_mask).cuda()
        self.model=MARE(wordVec_en,wordVec_zh,NwordVec_en,NwordVec_zh).cuda()
        self.D_optim=optim.SGD(self.model.D.parameters(),lr=dis_lr)
        self.encoder_optim=optim.SGD(self.model.share_encoder.parameters(),lr=encoder_lr)
        self.optim=optim.SGD(self.model.parameters(),lr=RE_lr)
        self.mono_optim=optim.SGD(self.model.monoRE.parameters(),lr=RE_lr)
        self.AE_optim=optim.SGD(self.model.share_encoder.parameters(),lr=AE_lr)
        self.D_sch=optim.lr_scheduler.ReduceLROnPlateau(self.D_optim,mode='max',factor=0.4,patience=2,verbose=True)
        self.encoder_sch=optim.lr_scheduler.ReduceLROnPlateau(self.encoder_optim,mode='max',factor=0.4,patience=2,verbose=True)
        self.sch=optim.lr_scheduler.ReduceLROnPlateau(self.optim,mode='max',factor=0.4,patience=2,verbose=True)
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
    def add_noise(self,words_,pos1_,pos2_):
        words=words_.data
        pos1=pos1_.data
        pos2=pos2_.data
        idx=(np.arange(words.size(1))+(Distance+1)*np.random.rand(words.size(1))).argsort()
        words=words[:,idx]
        pos1=pos1[:,idx]
        pos2=pos2[:,idx]
        idx=[]
        for i in range(SenLen):
            if np.random.rand()<ProbDrop:
                idx.append(i)
        if idx:
            words[:,idx]=0
            pos1[:,idx]=0
            pos2[:,idx]=0
        return Variable(words).cuda().contiguous(),Variable(pos1).cuda().contiguous(),Variable(pos2).cuda().contiguous()

    def AE_loss(self,wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh):
        noised_wordsEn,noised_pos1En,noised_pos2En=self.add_noise(wordsEn,pos1En,pos2En)
        logit_en=self.model.share_encoder.enEncoder(noised_wordsEn,noised_pos1En,noised_pos2En)
        noised_wordsZh,noised_pos1Zh,noised_pos2Zh=self.add_noise(wordsZh,pos1Zh,pos2Zh)
        logit_zh=self.model.share_encoder.zhEncoder(noised_wordsZh,noised_pos1Zh,noised_pos2Zh)
        loss_en=self.en_criterion(logit_en.view(-1,self.enSz),wordsEn.view(-1))
        loss_zh=self.zh_criterion(logit_zh.view(-1,self.zhSz),wordsZh.view(-1))
        loss=loss_en+loss_zh
        return loss
    def train_AE(self,wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh):
        self.AE_optim.zero_grad()
        loss=self.AE_loss(wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh)
        loss.backward()
        self.AE_optim.step()
        return loss.data[0]
    def train_RE(self,wordsEn,NwordsEn,pos1En,pos2En,rEn,lEn,wordsZh,NwordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask):
        self.model.train()
        pred=self.model(wordsEn,NwordsEn,pos1En,pos2En,rEn,lEn,wordsZh,NwordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask)
        loss=-torch.sum(pred.view(lEn.size(0)))+Orth_Coef*self.model.Orth_con(wordsEn,NwordsEn,pos1En,pos2En,wordsZh,NwordsZh,pos1Zh,pos2Zh)
        if AE_lambda>0.0:
            AE_loss=self.AE_loss(NwordsEn,pos1En,pos2En,NwordsZh,pos1Zh,pos2Zh)
            loss+=AE_lambda*AE_loss
            AE_loss=AE_loss.data[0]
        else:
            AE_loss=0.0
        if (loss!=loss).data.any():
            print("NaN Loss (training RE)")
            exit()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return AE_loss,loss.data[0]
    def train_RE_mono(self,wordsEn,NwordsEn,pos1En,pos2En,rEn,lEn,wordsZh,NwordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask):
        pred=self.model(wordsEn,NwordsEn,pos1En,pos2En,rEn,lEn,wordsZh,NwordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask)
        loss=-torch.sum(pred.view(lEn.size(0)))
        if (loss!=loss).data.any():
            print("NaN Loss (training RE)")
            exit()
        self.mono_optim.zero_grad()
        loss.backward()
        self.mono_optim.step()
        return loss.data[0]
    def eval(self,wordsEn,NwordsEn,pos1En,pos2En,rEn,lEn,wordsZh,NwordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask):
        pred=self.model(wordsEn,NwordsEn,pos1En,pos2En,rEn,lEn,wordsZh,NwordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask)
        return pred
    def decay(self,acc):
        self.D_sch.step(acc)
        self.encoder_sch.step(acc)
        self.sch.step(acc)
    def save(self,Tag):
        torch.save(self.model.state_dict(),"model%s.tar"%(Tag))
    def load(self,Tag):
        self.model.load_state_dict(torch.load("model%s.tar"%(Tag)))
