from __future__ import print_function
import numpy as np
import random
import os
import re
import sys
from constant import *
import pickle
entityId={}
relationId={}
wordTable=[{} for i in range(0,LangNum)]
wordVec=[[] for i in range(0,LangNum)]
wordTableSize=[0 for i in range(0,LangNum)]
wordVecDim=[0 for i in range(0,LangNum)]
neId={}
pun={' ','(',')'
     ,"'",'"','&','~','`','<','>'
     ,'*','%','^','#','$','@','!',' ',',','.','-','[',']','{','}',':',';'}
def judge(s):
    flag=False
    for i in range(0,len(s)):
        if not (s[i] in pun):
            flag=True
    return flag
def get_entityId(x):
    if not x in neId:
        neId[x]=len(neId)
    if x in entityId:
        return entityId[x]
    return 0
patR=[".","-",",","'","&","%","!","`","(",")","/"]
def Rep(sen,key):
    newkey=key
    for x in patR:
        newkey=newkey.replace(x," "+x+" ")
    return sen.replace(newkey,key)
class item:
    def __init__(self,sentence,language):
        self.Raw=sentence
        #self.raw=[re.sub("\s*","",x) for x in sentence.split()]
        self.raw=sentence.split()
        self.e1=get_entityId(self.raw[0])
        self.e2=get_entityId(self.raw[1])
        #self.rawe1=re.sub("\s*","",self.raw[2])
        #self.rawe2=re.sub("\s*","",self.raw[3])
        self.rawe1=self.raw[2]
        self.rawe2=self.raw[3]
        self.relation=relationId[self.raw[4]]
        sentence=Rep(sentence,self.rawe1)
        sentence=Rep(sentence,self.rawe2)
        #self.raw=[re.sub("\s*","",x) for x in sentence.split()]
        self.raw=sentence.split()
        self.sentence=["<\s>"]
        self.lang=language
        self.ep=[0,0]
        for i in range(5,len(self.raw)):
            if judge(self.raw[i]):
                self.sentence.append(self.raw[i])
        self.pos1=self.gen_position(1)
        self.pos2=self.gen_position(2)
        if self.ep[0]>self.ep[1]:
            self.ep[0],self.ep[1]=self.ep[1],self.ep[0]
        if self.ep[0]==self.ep[1]:
            if self.ep[0]==1:
                self.ep[1]+=1
            else:
                self.ep[0]-=1
    def gen_words(self):
        resVec=[]
        for x in self.sentence:
            if x in wordTable[self.lang]:
                resVec.append(wordTable[self.lang][x])
            else:
                resVec.append(0)
        if len(resVec)>SenLen[self.lang]:
            return resVec[0:SenLen[self.lang]]
        for i in range(0,SenLen[self.lang]-len(resVec)):
            resVec.append(0)
        return resVec
    def gen_position(self,enum):
        #print("Gen Pos")
        baseP=0
        if enum==1:
            for i in range(0,len(self.sentence)):
                if self.sentence[i].find(self.rawe1)!=-1:
                    baseP=i
            if baseP==0:
                return None
        else:
            for i in list(range(0,len(self.sentence)))[::-1]:
                if self.sentence[i].find(self.rawe2)!=-1:
                    baseP=i
            if baseP==0:
                return None
        self.ep[enum-1]=min(baseP,SenLen[self.lang]-3)
        baseP=min(baseP,SenLen)
        resVec=[]
        for i in range(0,min(SenLen[self.lang],len(self.sentence))):
            resVec.append(i-baseP+SenLen[self.lang])
        if len(resVec)>SenLen[self.lang]:
            print("???")
            print(len(resVec))
            return None #to be discussed
        for i in range(0,SenLen[self.lang]-len(resVec)):
            resVec.append(resVec[len(resVec)-1])
        return resVec
class instance:
    def __init__(self,e1,e2,r):
        self.e1=e1
        self.e2=e2
        self.r=r
        self.items=[[],[]]
    def add(self,language,it):
        self.items[language].append(it)
    def dump(self):
        for i in range(0,LangNum):
            fout=open("dumped"+str(i)+".txt","w")
            for x in self.items[i]:
                fout.write(x.Raw)
            fout.close()
    def out(self):
        wordVec_en=[]
        wordVec_zh=[]
        pos1_en=[]
        pos1_zh=[]
        pos2_en=[]
        pos2_zh=[]
        ep_en=[]
        ep_zh=[]
        for x in self.items[0]:
            wordVec_en.append(x.gen_words())
            pos1_en.append(x.pos1)
            pos2_en.append(x.pos2)
            ep_en.append(x.ep)
        for x in self.items[1]:
            wordVec_zh.append(x.gen_words())
            pos1_zh.append(x.pos1)
            pos2_zh.append(x.pos2)
            ep_zh.append(x.ep)
        return np.array(wordVec_en),np.array(pos1_en),np.array(pos2_en),np.array(ep_en),np.array(wordVec_zh),np.array(pos1_zh),np.array(pos2_zh),np.array(ep_zh),self.r,self.e1,self.e2
class dataset:
    def __init__(self,filenames,forTest=False):
        print("Start load %s %s dataset"%(filenames[0],filenames[1]))
        self.sentences=[[] for x in filenames]
        TmpCnt=0
        #ferr=open("err.txt","w")
        for i,filename in enumerate(filenames):
            print("Start load %s"%(dataPath+filename))
            with open(dataPath+filename,"r") as f:
                for x in f:
                    t=item(x,i)
                    if t.pos1 and t.pos2:
                        self.sentences[i].append(t)
        self.get_instances()
        self.datas=np.array(self.instances)
        print("load ended")
    def get_instances(self):
        self.instances=[]
        mp=dict()
        for i in range(0,LangNum):
            for x in self.sentences[i]:
                if not ((x.e1,x.e2,x.relation) in mp):
                    mp[(x.e1,x.e2,x.relation)]=len(self.instances)
                    self.instances.append(instance(x.e1,x.e2,x.relation))
                self.instances[mp[(x.e1,x.e2,x.relation)]].add(i,x)
    def dump(self):
        mx=0
        cho=instance(1,1,1)
        for x in self.instances:
            if len(x.items[0]) and len(x.items[1]):
                if len(x.items[0])+len(x.items[1])>mx:
                    mx=len(x.items[0])+len(x.items[1])
                    cho=x
        cho.dump()
    def pre_batchs(self):
        datas=self.datas
        #np.array(self.instances)
        shuffle_indices=np.random.permutation(np.arange(len(self.instances)))
        datas=datas[shuffle_indices]
        print("gen batch %d %d"%(len(self.instances),batch_size))
        for i in range(0,max(len(self.instances)//batch_size,1)):
            s,t=i*batch_size,min((i+1)*batch_size,len(self.instances))
            yield datas[s:t].tolist()
        s,t=(len(self.instances)//batch_size)*batch_size,len(self.instances)
        yield datas[s:t].tolist()
    def batchs(self):
        for b in self.pre_batchs():
            wordVec_en=[]
            pos1_en=[]
            pos2_en=[]
            ep_en=[]
            wordVec_zh=[]
            pos1_zh=[]
            pos2_zh=[]
            ep_zh=[]
            r_en=[]
            r_zh=[]
            l_en=[]
            l_zh=[]
            e1=[]
            e2=[]
            relation=[]
            for x in b:
                wordVec_e,pos1_e,pos2_e,ep_e,wordVec_z,pos1_z,pos2_z,ep_z,r,E1,E2=x.out()
                if wordVec_e.shape[0]:
                    wordVec_en.append(wordVec_e)
                    pos1_en.append(pos1_e)
                    pos2_en.append(pos2_e)
                    ep_en.append(ep_e)
                if wordVec_z.shape[0]:
                    wordVec_zh.append(wordVec_z)
                    pos1_zh.append(pos1_z)
                    pos2_zh.append(pos2_z)
                    ep_zh.append(ep_z)
                for i in range(0,wordVec_e.shape[0]):
                    r_en.append(r)
                for i in range(0,wordVec_z.shape[0]):
                    r_zh.append(r)
                l_en.append(wordVec_e.shape[0])
                l_zh.append(wordVec_z.shape[0])
                e1.append(E1)
                e2.append(E2)
                relation.append(r)
            wordVec_en=np.concatenate(wordVec_en,axis=0)
            pos1_en=np.concatenate(pos1_en,axis=0)
            pos2_en=np.concatenate(pos2_en,axis=0)
            ep_en=np.concatenate(ep_en,axis=0)
            wordVec_zh=np.concatenate(wordVec_zh,axis=0)
            pos1_zh=np.concatenate(pos1_zh,axis=0)
            pos2_zh=np.concatenate(pos2_zh,axis=0)
            ep_zh=np.concatenate(ep_zh,axis=0)
            r_en=np.array(r_en)
            r_zh=np.array(r_zh)
            l_en=np.array(l_en)
            l_zh=np.array(l_zh)
            relation=np.array(relation)
            e1=np.array(e1)
            e2=np.array(e2)
            yield wordVec_en,pos1_en,pos2_en,ep_en,r_en,l_en,wordVec_zh,pos1_zh,pos2_zh,ep_zh,r_zh,l_zh,relation,e1,e2
    def save(self,Tag):
        wordVec_en=[]
        pos1_en=[]
        pos2_en=[]
        ep_en=[]
        wordVec_zh=[]
        pos1_zh=[]
        pos2_zh=[]
        ep_zh=[]
        r_en=[]
        r_zh=[]
        l_en=[]
        l_zh=[]
        e1=[]
        e2=[]
        relation=[]
        for wordVec_e,pos1_e,pos2_e,ep_e,r_e,l_e,wordVec_z,pos1_z,pos2_z,ep_z,r_z,l_z,re,E1,E2 in self.batchs():
            wordVec_en.append(wordVec_e)
            pos1_en.append(pos1_e)
            pos2_en.append(pos2_e)
            ep_en.append(ep_en)
            r_en.append(r_en)
            l_en.append(l_en)
            wordVec_zh.append(wordVec_z)
            pos1_zh.append(pos1_z)
            pos2_zh.append(pos2_z)
            ep_zh.append(ep_z)
            r_zh.append(r_z)
            l_zh.append(l_z)
            relation.append(re)
            e1.append(E1)
            e2.append(E2)
        np.save(dataPath+Tag+"wordsEn.npy",np.array(wordVec_en))
        np.save(dataPath+Tag+"pos1En.npy",np.array(pos1_en))
        np.save(dataPath+Tag+"pos2En.npy",np.array(pos2_en))
        np.save(dataPath+Tag+"epEn.npy",np.array(ep_en))
        np.save(dataPath+Tag+"rEn.npy",np.array(r_en))
        np.save(dataPath+Tag+"lEn.npy",np.array(l_en))
        np.save(dataPath+Tag+"wordsZh.npy",np.array(wordVec_zh))
        np.save(dataPath+Tag+"pos1Zh.npy",np.array(pos1_zh))
        np.save(dataPath+Tag+"pos2Zh.npy",np.array(pos2_zh))
        np.save(dataPath+Tag+"epZh.npy",np.array(ep_zh))
        np.save(dataPath+Tag+"rZh.npy",np.array(r_zh))
        np.save(dataPath+Tag+"lZh.npy",np.array(l_zh))
        np.save(dataPath+Tag+"relation.npy",np.array(relation))
        np.save(dataPath+Tag+"e1.npy",np.array(e1))
        np.save(dataPath+Tag+"e2.npy",np.array(e2))
def load_vocab(filename,idx):
    vec=open(dataPath+filename,"r")
    print("Start Reading "+filename)
    t=vec.readlines()
    wordTableSize[idx]=int(t[0])
    wordVecDim[idx]=int(t[1])
    wordTable[idx]={"<UNK>":0}
    wordVec[idx].append([0.0 for i in range(0,wordVecDim[idx])])
    for i in list(range(2,len(t)))[::2]:
        wd=re.sub("\s*","",t[i])
        wordTable[idx][wd]=len(wordTable[idx])
        wordVec[idx].append([float(x) for x in t[i+1].split()])
    wordVec[idx]=np.array(wordVec[idx])
    vec.close()
    saveF=open(dataPath+"Vec"+str(idx)+".data","wb")
    pickle.dump(wordVec[idx],saveF)
    saveF.close()
    saveF=open(dataPath+"Table"+str(idx)+".data","wb")
    pickle.dump(wordTable[idx],saveF)
    saveF.close()
def load_vo2():
    for i in range(0,LangNum):
        saveF=open(dataPath+"Vec"+str(i)+".data","rb")
        print("loading "+dataPath+"Vec"+str(i)+".data")
        wordVec[i]=pickle.load(saveF)
        wordVec[i]=wordVec[i]/np.sqrt((wordVec[i]**2).sum(1))[:,None]
        saveF.close()
    for i in range(0,LangNum):
        saveF=open(dataPath+"Table"+str(i)+".data","rb")
        print("loading "+dataPath+"Table"+str(i)+".data")
        wordTable[i]=pickle.load(saveF)
        saveF.close()
def load_init():
    with open(dataPath+"entity2id.txt","r") as e2I:
        for x in e2I:
            x=x.split()
            entityId[x[0]]=int(x[1])
    with open(dataPath+"relation2id.txt","r") as r2I:
        for x in r2I:
            x=x.split()
            relationId[x[0]]=int(x[1])
    load_vocab("vec_en.out",0)
    load_vocab("vec_zh.out",1)
    #load_vo2()
def outneid():
    f=open(dataPath+"neId.txt","w")
    for x in neId:
        f.write("%s %s\n"%(x,neId[x]))
    f.close()
if __name__=="__main__":
    load_init()
    np.save(dataPath+"wordVec_en.npy",np.array(wordVec[0]))
    np.save(dataPath+"wordVec_zh.npy",np.array(wordVec[1]))
    train=dataset(["train_en.txt","train_zh.txt"])
    train.save("Train_")
    valid=dataset(["valid_en.txt","valid_zh.txt"])
    valid.save("Valid_")
    test=dataset(["test_en.txt","test_zh.txt"])
    test.save("Test_")

