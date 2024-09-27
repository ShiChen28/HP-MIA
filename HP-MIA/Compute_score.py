import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import random
import scipy.stats as st


def Com_Loss(model, ref_model, data_loader, lossf):
    sf = nn.Softmax(dim=1)
    Loss = []
    C_Loss = []

    for i, data in enumerate(tqdm(data_loader)):
        x,y = data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        x, y = x.to(device),y.to(device)
        output = sf(model(x))
        o_Loss = lossf(model(x), y)

        los = []
        for s in range(len(ref_model)):
            refmodel = ref_model[s]
            refmodel.to(device)
            los.append(lossf(refmodel(x), y))

        for j in range(len(y)):
            o_Loss0 = -1*o_Loss[j].item()
            eloss = torch.zeros([len(ref_model)])
            for i in range(len(ref_model)):
                eloss[i] = -1*los[i][j].item()

            c_Loss = o_Loss0 - torch.mean(eloss)
            Loss.append(o_Loss0)
            C_Loss.append(c_Loss.item())

    return Loss, C_Loss 

def Com_AUC(tpr,fpr):
    sort_i = np.argsort(fpr)
    tpr = tpr[sort_i]
    fpr = fpr[sort_i]
    AUC = 0
    for i in range(len(fpr)-1):
        AUC = AUC + 0.5*(tpr[i]+tpr[i+1])*(fpr[i+1]-fpr[i])
    return AUC


def Mentro(output, j, y, num):
    M = -(1 - output[j][y[j]].item())*torch.log(output[j][y[j]]);
    for i in range(num):
        if i != y[j]:
            M = M  - output[j][i].item()*torch.log(1 - output[j][i])
    return -M
        

def Com_Mentr(model, data_loader, num):
    sf = nn.Softmax(dim=1)
    Mentr = []
    for i, data in enumerate(tqdm(data_loader)):
        x,y = data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        x, y = x.to(device),y.to(device)
        output = sf(model(x))

        for j in range(len(y)):
            M = Mentro(output, j, y, num)
            Mentr.append(M.item())


    return Mentr



def Com_Conf(model, ref_model, data_loader):
    sf = nn.Softmax(dim=1)
    Conf = []
    C_Conf = []
    for i, data in enumerate(tqdm(data_loader)):
        x,y = data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        x, y = x.to(device),y.to(device)
        output = sf(model(x))
        los = []
        for s in range(len(ref_model)):
            refmodel = ref_model[s]
            refmodel.to(device)
            outp = sf(refmodel(x))
            los.append(outp)

        for j in range(len(y)):
            o_conf0 = output[j][y[j]].item()
            econf = torch.zeros([len(ref_model)])
            for i in range(len(ref_model)):
                econf[i] = los[i][j][y[j]].item()

            c_conf = o_conf0 - torch.mean(econf)
            Conf.append(o_conf0)
            C_Conf.append(c_conf.item())


    return Conf, C_Conf 


def findthre_in(a_list,label,mu = 0.95):
    sort_i = np.argsort(a_list)
    
    a_list = a_list[sort_i]
    #print(np.sum(label))
    label = label[sort_i]
    #print(np.sum(label))
    num = a_list.size
    Pr = np.zeros([num - 1])
    Pr_num = np.zeros([num - 1])
    flag = np.zeros([num - 1])
    for i in range(num - 1):
        Pr_num[i] = np.sum(label[i:num])
        Pr[i] = Pr_num[i]/(num - i)
       
        
    mu_i = np.argwhere(Pr > mu)
    if len(mu_i) == 0:
        if len(Pr) == 0:
            return 0,0,0,0
        else:
            Max_i = np.argmax(Pr)
            thre = (a_list[Max_i] + a_list[Max_i + 1])*0.5
            return thre,0,0,0
    Max_i = mu_i[0]
    for i in range(len(mu_i)):
        if Pr_num[mu_i[i]] > Pr_num[Max_i]:
            Max_i = mu_i[i]
    
    thre = (a_list[Max_i] + a_list[Max_i + 1])*0.5
    other_i = np.nonzero(a_list < thre)
 
    return thre,other_i,Pr_num[Max_i],1

def findthre_out(a_list,label,mu = 0.95):
    sort_i = np.argsort(a_list)
    a_list = a_list[sort_i]
    label = label[sort_i]
    num = a_list.size
    Pr = np.zeros([num - 1])
    Pr_num = np.zeros([num - 1])

    for i in range(num - 1):
        Pr_num[i] = i + 1 - np.sum(label[0:i + 1])
        Pr[i] = Pr_num[i]/(i + 1)
    mu_i = np.argwhere(Pr > mu)
    if len(mu_i) == 0:
        return 0,0,0,0
    Max_i = mu_i[0]
    for i in range(len(mu_i)):
        if Pr_num[mu_i[i]] > Pr_num[Max_i]:
            Max_i = mu_i[i]
            
    thre = (a_list[Max_i] + a_list[Max_i + 1])*0.5
    other_i = np.nonzero(a_list >= thre)
    
    return thre,other_i,Pr_num[Max_i],1


def Twostage_attack(score0,score1,t0,t1):
    if score0 < t0:
        return 0
    elif score1 > t1:
        return 1
    else:
        return 0

def Twostage_attack_eval(Score0_in,Score1_in,Score0_out,Score1_out,t0,t1,radio = 1):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    c = [i for i in range(len(Score0_in))]
    random.shuffle(c)
    num = int(len(Score0_out)/radio)
    for i in range(len(Score0_in)):
        if Twostage_attack(Score0_in[c[i]],Score1_in[c[i]],t0,t1) == 1:
            TP = TP + 1
        else:
            FN = FN + 1
    c = [i for i in range(len(Score0_out))]
    random.shuffle(c)
    num = int(len(Score0_out)/radio)
    for i in range(len(Score0_out)):
        if Twostage_attack(Score0_out[c[i]],Score1_out[c[i]],t0,t1) == 1:
            FP = FP + 1
        else:
            TN = TN + 1
    return TP,FP,TN,FN

def Twostage_attack_eval_in(Score0_in,Score1_in,t0,t1,radio = 1):
    T_score = []
    F_score = []
    c = [i for i in range(len(Score0_in))]
    random.shuffle(c)
    num = int(len(Score0_in)/radio)
    for i in range(num):
        if Twostage_attack(Score0_in[c[i]],Score1_in[c[i]],t0,t1) == 1:
            T_score.append(Score1_in[c[i]] - Score0_in[c[i]])
        else:
            F_score.append(Score1_in[c[i]] - Score0_in[c[i]])
    return T_score,F_score


def Onestage_attack(score,t):
    if score < t:
        return 0
    else:
        return 1
def Onestage_attack_eval(Score_in,Score_out,t):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(Score_in)):
        if Onestage_attack(Score_in[i],t) == 1:
            TP = TP + 1
        else:
            FN = FN + 1
    for i in range(len(Score_out)):
        if Onestage_attack(Score_out[i],t) == 1:
            FP = FP + 1
        else:
            TN = TN + 1
    return TP,FP,TN,FN

def Twostage_create(shadow0_in,shadow1_in,shadow0_out,shadow1_out,mu):
    score0 = np.array(shadow0_in + shadow0_out)
    score1 = np.array(shadow1_in + shadow1_out)
    label = np.array([1 for i in range(len(shadow0_in))] + [0 for i in range(len(shadow0_out))])
    sort_i = np.argsort(score0)
    score0 = score0[sort_i]
    score1 = score1[sort_i]
    label = label[sort_i]
    best_t0 = 0
    best_t1 = 0
    best_Num = 0
    for i in range(0,10000,50):
        t0,other_i,_,flag0  = findthre_out(score0,label,i*1e-4)
        if flag0 == 1:
            score1_t = score1[other_i]
            label_t = label[other_i]
            t1,other_j,Num,flag1 = findthre_in(score1_t,label_t,mu)
         
            if Num > best_Num and flag0 + flag1 == 2:
                best_Num = Num
                best_t0 = t0
                best_t1 = t1
            
    return best_t0,best_t1

def Twostage_create_beta(shadow0_in,shadow1_in,shadow0_out,shadow1_out,mu,beta):
    score0 = np.array(shadow0_in + shadow0_out)
    score1 = np.array(shadow1_in + shadow1_out)
    label = np.array([1 for i in range(len(shadow0_in))] + [0 for i in range(len(shadow0_out))])
    sort_i = np.argsort(score0)
    score0 = score0[sort_i]
    score1 = score1[sort_i]
    label = label[sort_i]
    best_t0 = 0
    best_t1 = 0
    best_Num = 0
    t0,other_i,_,flag0  = findthre_out(score0,label,beta)
    if flag0 == 1:
        score1_t = score1[other_i]
        label_t = label[other_i]
        t1,other_j,Num,flag1 = findthre_in(score1_t,label_t,mu)

    return t0,t1


def Onestage_create(shadow_in,shadow_out,mu):
    score = np.array(shadow_in + shadow_out)
    label = np.array([1 for i in range(len(shadow_in))] + [0 for i in range(len(shadow_out))])
    sort_i = np.argsort(score)
    score = score[sort_i]
    label = label[sort_i]
    t,_,_,flag  = findthre_in(score,label,mu)            
    return t,flag