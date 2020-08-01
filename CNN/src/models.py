from __future__ import print_function
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from constant import *
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderGRU(nn.Module):
    def __init__(self,
               vocab_size,emb_dim,emb,
               hidden_dim,
               nlayers,
               pad_token,
               bidir=False):
        #emb---np wordVec vocab_size=len(emb)
        super(EncoderGRU,self).__init__()
        #self.word_emb=nn.Embedding(vocab_size,emb_dim,pad_token)
        #self.word_emb.weight.data.copy_(torch.from_numpy(emb))
        #self.pos1_emb=nn.Embedding(MaxPos,dimWPE)
        #self.pos2_emb=nn.Embedding(MaxPos,dimWPE)
        self.hidden_dim=hidden_dim
        self.emb_dim=emb_dim+dimWPE*2
        self.nlayers=nlayers
        self.bidir=bidir
        #using gru
        self.gru=nn.GRU(
            self.emb_dim//2 if bidir else self.emb_dim,
            self.hidden_dim,
            self.nlayers,
            bidirectional=bidir,
            batch_first=True
        )
    def forward(self,input_,pos1,pos2):
        embd=self.word_emb(input_)
        pos1=self.pos1_emb(pos1)
        pos2=self.pos2_emb(pos2)
        embd=torch.cat((embd,pos1,pos2),2)
        #using gru
        _,h_t_=self.encoder(embed)
        h_t=torch.cat((h_t_[-1],h_t_[-2]),1)if self.bidir else h_t_[-1]
        return h_t
class EncoderCNN(nn.Module):
    def __init__(self,
               vocab_size,emb,emb_dim=dimWE,
               hidden_dim=dimC,lang=0):
        #emb---np wordVec vocab_size=len(emb)
        super(EncoderCNN,self).__init__()
        self.lang=lang
        self.word_emb=nn.Embedding(vocab_size,emb_dim)
        self.word_emb.weight.data.copy_(torch.from_numpy(emb))
        self.pos1_emb=nn.Embedding(MaxPos,dimWPE)
        self.pos2_emb=nn.Embedding(MaxPos,dimWPE)
        self.maxPooling=nn.MaxPool1d(SenLen[self.lang]-2)
        self.emb_dim=emb_dim+dimWPE*2
        self.hidden_dim=hidden_dim
        #using CNN
        self.tanh=nn.Tanh()
        self.conv=nn.Conv1d(self.emb_dim,hidden_dim,filter_size)
        self.dropout=nn.Dropout(p=CNNDropout)
    def forward(self,inp,pos1,pos2):
        Len=inp.size(0)
        embd=self.word_emb(inp)
        pos1=self.pos1_emb(pos1)
        pos2=self.pos2_emb(pos2)
        embd=torch.cat((embd,pos1,pos2),2).transpose(1,2)
        conved=self.conv(embd)
        pooled=self.maxPooling(conved).view(Len,dimC)
        out=self.tanh(pooled)
        return self.dropout(out)
class CNNEncoder(nn.Module):
    def __init__(self,vocab_en,emb_en,vocab_zh,emb_zh):
        super(CNNEncoder,self).__init__()
        self.encoder_en=EncoderCNN(vocab_en,emb_en,dimWE,dimC,0)
        self.encoder_zh=EncoderCNN(vocab_zh,emb_zh,dimWE,dimC,1)
    def forward(self,wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh):
        return self.encoder_en(wordsEn,pos1En,pos2En),self.encoder_zh(wordsZh,pos1Zh,pos2Zh)
class Discriminator(nn.Module):
    def __init__(self,
                 dis_input_dim=Encodered_dim,
                 nlayers=dis_layers,
                 hidden_dim=dis_hidden_dim,
                 input_dropout=dis_input_dropout,
                 dropout=dis_dropout):
        super(Discriminator,self).__init__()
        self.dis_input=dis_input_dim
        layers=[nn.Dropout(input_dropout)]
        for i in range(0,nlayers+1):
            input_dim=self.dis_input if i==0 else hidden_dim
            output_dim=1 if i==nlayers else hidden_dim
            layers.append(nn.Linear(input_dim,output_dim))
            if i<nlayers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Sigmoid())
        self.layers=nn.Sequential(*layers)
    def forward(self,inp):
        assert inp.dim()==2 and inp.size(1)==self.dis_input
        return self.layers(inp).view(-1)
class MultiRE(nn.Module):
    def __init__(self):
        super(MultiRE,self).__init__()
        self.relation_emb=nn.Embedding(dimR,Encodered_dim)
        self.dropout=nn.Dropout(p=Att_dropout)
        #self.softmax=nn.Softmax()
        #self.logsoftmax=nn.LogSoftmax()
        self.M=nn.Linear(Encodered_dim,dimR)
    def forward(self,inp_en,r_en,l_en,inp_zh,r_zh,l_zh,re_mask):
        NumRe=r_en.size(0)
        NumIn=l_zh.size(0)
        relation_en=self.relation_emb(r_en)
        relation_zh=self.relation_emb(r_zh)
        attn_en=torch.sum(relation_en*inp_en,2)
        attn_zh=torch.sum(relation_zh*inp_zh,2)
        p=Variable(torch.cuda.FloatTensor(NumIn,NumRe).fill_(0.0))
        L_en=0
        L_zh=0
        R_vec=Variable(torch.cuda.FloatTensor(NumIn,NumRe,Encodered_dim).fill_(0.0))
        S=Variable(torch.cuda.FloatTensor(NumIn,NumRe,Encodered_dim).fill_(0.0))
        for i in range(0,NumIn):
            R_en=L_en+l_en[i].data[0]
            R_zh=L_zh+l_zh[i].data[0]
            if R_en>L_en and R_zh>L_zh:
                Att=F.softmax(torch.cat((attn_en[:,L_en:R_en],attn_zh[:,L_zh:R_zh]),1),1)
                S[i]=self.dropout(torch.matmul(Att,torch.cat((inp_en[L_en:R_en],inp_zh[L_zh:R_zh]),0)))
                R_vec[i]=relation_en[:,L_en,:]
            elif R_en>L_en:
                Att=F.softmax(attn_en[:,L_en:R_en],1)
                S[i]=self.dropout(torch.matmul(Att,inp_en[L_en:R_en]))
                R_vec[i]=relation_en[:,L_en,:]
            elif R_zh>L_zh:
                Att=F.softmax(attn_zh[:,L_zh:R_zh],1)
                S[i]=self.dropout(torch.matmul(Att,inp_zh[L_zh:R_zh]))
                R_vec[i]=relation_zh[:,L_zh,:]
            else:
                print("ERR NO sentences")
                exit()
            L_en=R_en
            L_zh=R_zh
        p_n=F.log_softmax(self.M(S)+torch.sum(R_vec*S,2).view(NumIn,NumRe,1),2).view(NumIn,NumRe,dimR)
        return p_n[re_mask].view(NumIn,NumRe)
class MonoRE(nn.Module):
    def __init__(self):
        super(MonoRE,self).__init__()
        self.relation_emb=nn.Embedding(dimR,Encodered_dim)
        self.dropout=nn.Dropout(p=Att_dropout)
        #self.softmax=nn.Softmax()
        #self.logsoftmax=nn.LogSoftmax()
        self.M=nn.Linear(Encodered_dim,dimR)
    def forward(self,inp,r,l,re_mask):
        NumRe=r.size(0)
        NumIn=l.size(0)
        relation=self.relation_emb(r)
        attn=torch.sum(relation*inp,2)
        p=Variable(torch.cuda.FloatTensor(NumIn,NumRe).fill_(0.0))
        L=0
        R_vec=Variable(torch.cuda.FloatTensor(NumIn,NumRe,Encodered_dim).fill_(0.0))
        S=Variable(torch.cuda.FloatTensor(NumIn,NumRe,Encodered_dim).fill_(0.0))
        for i in range(0,NumIn):
            R=L+l[i].data[0]
            if R>L:
                Att=F.softmax(attn[:,L:R],1)
                S[i]=self.dropout(torch.matmul(Att,inp[L:R]))
                R_vec[i]=relation[:,L,:]
            L=R
        p_n=F.log_softmax((self.M(S)+torch.sum(R_vec*S,2).view(NumIn,NumRe,1)),2).view(NumIn,NumRe,dimR)
        return p_n[re_mask].view(NumIn,NumRe)

class AMRE(nn.Module):
    def __init__(self,emb_en,emb_zh):
        super(AMRE,self).__init__()
        self.encoder=CNNEncoder(len(emb_en),emb_en,len(emb_zh),emb_zh).cuda()
        self.enRE=MonoRE().cuda()
        self.zhRE=MonoRE().cuda()
    def forward(self,wordsEn,pos1En,pos2En,rEn,lEn,wordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask):
        inp_en,inp_zh=self.encoder(wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh)
        return self.enRE(inp_en,rEn,lEn,re_mask)+self.zhRE(inp_zh,rZh,lZh,re_mask)
class MARE(nn.Module):
    def __init__(self,emb_en,emb_zh):
        super(MARE,self).__init__()
        self.D=Discriminator().cuda()
        self.share_encoder=CNNEncoder(len(emb_en),emb_en,len(emb_zh),emb_zh).cuda()
        self.multiRE=MultiRE().cuda()
        self.monoRE=AMRE(emb_en,emb_zh)
    def Orth_con(self,wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh):
        share_en,share_zh=self.share_encoder(wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh)
        mono_en,mono_zh=self.monoRE.encoder(wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh)
        share=torch.cat((share_en,share_zh),0)
        mono=torch.cat((mono_en,mono_zh),0)
        share-=torch.mean(share,0)
        mono-=torch.mean(mono,0)
        share=F.normalize(share,2,1)
        mono=F.normalize(mono,2,1)
        correlation_mat=torch.matmul(share.transpose(0,1),mono)
        cost=torch.mean(correlation_mat*correlation_mat)
        return cost
    def forward(self,wordsEn,pos1En,pos2En,rEn,lEn,wordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask):
        share_en,share_zh=self.share_encoder(wordsEn,pos1En,pos2En,wordsZh,pos1Zh,pos2Zh)
        return self.monoRE(wordsEn,pos1En,pos2En,rEn,lEn,wordsZh,pos1Zh,pos2Zh,rZh,lZh,re_mask)+self.multiRE(share_en,rEn,lEn,share_zh,rZh,lZh,re_mask)
