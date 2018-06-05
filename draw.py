import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
#matplotlib.use('Agg')
plt.switch_backend('agg')
from matplotlib.pyplot import plot,savefig
flag=['mD','g+','bs','cx','yh','rH']
markers=["D","+","s","x","h","H"]
linestys=["solid","dashed","dashdot","dotted"]
colors=["magenta","green","blue","cyan","yellow","red"]
if __name__=='__main__':
    print("Start")
    label=np.load(sys.argv[1])
    pred=np.load(sys.argv[2])
    p,r,th=precision_recall_curve(label,pred)
    label=np.reshape(label,(-1,175))
    pred=np.reshape(pred,(-1,175))
    index2=[]
    for i in range(0,label.shape[1]):
        if np.any(label[:,i]):
            index2.append(i)
    label=label[:,index2]
    pred=pred[:,index2]
    micro_auc=average_precision_score(label,pred,average="micro")
    macro_auc=average_precision_score(label,pred,average="macro")
    Tag="Micro: %.5f Macro: %.5f"%(micro_auc,macro_auc)
    print(Tag)
    plt.plot(r[:],p[:],linewidth="1",markevery=0.1)
    print("Start printing")
    plt.xlim(0.0,0.3)
    plt.ylim(0.5,1.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    savefig(sys.argv[3]+".eps")
    print("END")
