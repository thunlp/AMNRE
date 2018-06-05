from __future__ import print_function
import numpy as np
import random
import os
import re
import sys
import torch
from torch.autograd import Variable
from constant import *
import pickle
wordVec_en=np.array([])
wordVec_zh=np.array([])
def load_init():
    wordVec_en=np.load(dataPath+dataName+"newVec_en.npy")
    wordVec_zh=np.load(dataPath+dataName+"newVec_zh.npy")
class dataset:
    def __init__(self,Tag):
        print(Tag)
        self.wordsEn=np.load(dataPath+Tag+"NwordsEn.npy")
        self.NwordsEn=np.load(dataPath+Tag+"NwordsEn.npy")
        self.pos1En=np.load(dataPath+Tag+"pos1En.npy")
        self.pos2En=np.load(dataPath+Tag+"pos2En.npy")
        self.epEn=np.load(dataPath+Tag+"epEn.npy")
        self.rEn=np.load(dataPath+Tag+"rEn.npy")
        self.lEn=np.load(dataPath+Tag+"lEn.npy")
        self.wordsZh=np.load(dataPath+Tag+"NwordsZh.npy")
        self.NwordsZh=np.load(dataPath+Tag+"NwordsZh.npy")
        self.pos1Zh=np.load(dataPath+Tag+"pos1Zh.npy")
        self.pos2Zh=np.load(dataPath+Tag+"pos2Zh.npy")
        self.epZh=np.load(dataPath+Tag+"epZh.npy")
        self.rZh=np.load(dataPath+Tag+"rZh.npy")
        self.lZh=np.load(dataPath+Tag+"lZh.npy")
        self.relation=np.load(dataPath+Tag+"relation.npy")
        self.e1=np.load(dataPath+Tag+"e1.npy")
        self.e2=np.load(dataPath+Tag+"e2.npy")
    def batchs(self):
        idx=np.random.permutation(np.arange(len(self.wordsEn)))
        for i in idx:
            wordsEn=Variable(torch.cuda.LongTensor(self.wordsEn[i]))
            NwordsEn=Variable(torch.cuda.LongTensor(self.NwordsEn[i]))
            pos1En=Variable(torch.cuda.LongTensor(self.pos1En[i]))
            pos2En=Variable(torch.cuda.LongTensor(self.pos2En[i]))
            epEn=Variable(torch.cuda.LongTensor(self.epEn[i]))
            rEn=Variable(torch.cuda.LongTensor(self.rEn[i]))
            rEn=rEn.view(1,-1)
            lEn=Variable(torch.cuda.LongTensor(self.lEn[i]))
            wordsZh=Variable(torch.cuda.LongTensor(self.wordsZh[i]))
            NwordsZh=Variable(torch.cuda.LongTensor(self.NwordsZh[i]))
            pos1Zh=Variable(torch.cuda.LongTensor(self.pos1Zh[i]))
            pos2Zh=Variable(torch.cuda.LongTensor(self.pos2Zh[i]))
            epZh=Variable(torch.cuda.LongTensor(self.epZh[i]))
            rZh=Variable(torch.cuda.LongTensor(self.rZh[i]))
            rZh=rZh.view(1,-1)
            lZh=Variable(torch.cuda.LongTensor(self.lZh[i]))
            yield wordsEn,NwordsEn,pos1En,pos2En,epEn,rEn,lEn,wordsZh,NwordsZh,pos1Zh,pos2Zh,epZh,rZh,lZh,self.relation[i],self.e1[i],self.e2[i]
