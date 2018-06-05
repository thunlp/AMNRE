from __future__ import print_function
import os
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime
from constant import *

def currentTime():
    return datetime.datetime.now().isoformat()
def get_mask(re_sel):
    rm=torch.cuda.ByteTensor(len(re_sel),dimR)
    rm.zero_()
    for xy in re_sel:
        rm[xy[0]][xy[1]]=1
    return rm

