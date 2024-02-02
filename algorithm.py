import numpy as np
from munkres import Munkres
import time

K=10
LH=22
LG=LH+1

def Kmer(sequence, K): 
    # kmer.cpp provides a faster implement
    m=4**K
    na_vect=[0]*(3*m)
    pos_sum=[0]*m
    squa_sum=[0]*m
    n=len(sequence)-(K-1)
    index_map = {  'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3  }
    for i in range(0, n):
        flag=1
        for l in range(0,K):
            if sequence[i+l] not in index_map.keys():
                flag=0
        if flag == 0:
            continue
        tem=index_map[sequence[i]]
        for l in range(1,K):
            tem=4*tem+index_map[sequence[i+l]]
        na_vect[tem] += 1
        pos_sum[tem] += i+1
    for k in range(0,m):
        if na_vect[k] != 0:
            na_vect[k+m] = pos_sum[k] / na_vect[k]
        else:
            na_vect[k+m]=0
    for i in range(0, n):
        flag=1
        for l in range(0,K):
            if sequence[i+l] not in index_map.keys():
                flag=0
        if flag == 0:
            continue
        tem=index_map[sequence[i]]
        for l in range(1,K):
            tem=4*tem+index_map[sequence[i+l]]
        squa_sum[tem] += ( i + 1 - na_vect[tem+m] ) ** 2
    for k in range(0,m):
        if na_vect[k] != 0:
            na_vect[k+2*m] = squa_sum[k] / (n * na_vect[k])
        else:
            na_vect[k+2*m]=0
    return na_vect

def add(nv1, nv2):
    K_power_4 = 4 ** K
    nvtem = np.zeros(3 * K_power_4)
    nvtem[:K_power_4] = nv1[:K_power_4] + nv2[:K_power_4]
    N1 = nv1[:K_power_4].sum()
    N2 = nv2[:K_power_4].sum()
    non_zero_indices = nvtem[:K_power_4] != 0
    denominator = nv1[:K_power_4][non_zero_indices] + nv2[:K_power_4][non_zero_indices]
    
    nvtem[K_power_4:2 * K_power_4][non_zero_indices] = (nv1[:K_power_4][non_zero_indices] * nv1[K_power_4:2 * K_power_4][non_zero_indices] 
                                                         + nv2[:K_power_4][non_zero_indices] * (nv2[K_power_4:2 * K_power_4][non_zero_indices] + N1)) / denominator
    term1 = N1 * nv1[:K_power_4][non_zero_indices] * nv1[2 * K_power_4:3 * K_power_4][non_zero_indices]
    term2 = N2 * nv2[:K_power_4][non_zero_indices] * nv2[2 * K_power_4:3 * K_power_4][non_zero_indices]
    term3 = nv1[:K_power_4][non_zero_indices] * (nv1[K_power_4:2*K_power_4][non_zero_indices] - nvtem[K_power_4:2*K_power_4][non_zero_indices]) ** 2
    term4 = nv2[:K_power_4][non_zero_indices] * (nv2[K_power_4:2*K_power_4][non_zero_indices] + N1 - nvtem[K_power_4:2*K_power_4][non_zero_indices]) ** 2
    
    nvtem[2 * K_power_4:3 * K_power_4][non_zero_indices] = (term1 + term2 + term3 + term4) / (N1 + N2) / denominator
    
    return nvtem

def cos(nv1,nv2):
    return np.dot(nv1,nv2)/np.linalg.norm(nv1, ord=2)/np.linalg.norm(nv2, ord=2)

def rank(v):
    ranked_indices = np.argsort(v)
    ranked_vector = np.zeros(len(v))
    ranked_vector[ranked_indices] = np.arange(0, len(v))
    return ranked_vector

def search(D):
    mk=Munkres()
    indexes=mk.compute(D.copy())
    ans=0
    for i in range(len(indexes)):
        ans+=D[indexes[i]]
    return ans,indexes

def main():
    T1=time.time()
    g=list(np.load("nv_g_10.npy",allow_pickle=True))
    gr=list(np.load("nv_g_reverse.npy",allow_pickle=True))
    h=list(np.load("nv_h_10.npy",allow_pickle=True))
    gl=list(np.load("nv_g_len.npy",allow_pickle=True))
    hl=list(np.load("nv_h_len.npy",allow_pickle=True))
    h=h[0:LH]
    hl=hl[0:LH]

    Mat_nv_gh=np.zeros((LH,LG))
    Mat_L_gh=np.zeros((LH,LG))
    Mat_combined_nv=np.zeros((LG,LG,LH))
    Mat_combined_L=np.zeros((LG,LG,LH))

    for i in range(LH):
        for j in range(LG):
            Mat_nv_gh[i][j]=-max(cos(h[i],g[j]),cos(h[i],gr[j]))
            Mat_L_gh[i][j]=abs(hl[i]-gl[j])
    
    for i1 in range(LG):
        print(i1)
        for i2 in range(LG):
            if i1==i2:
                continue
            v1=add(g[i1],g[i2])
            v2=add(gr[i1],g[i2])
            v3=add(g[i1],gr[i2])
            v4=add(gr[i1],gr[i2])
            for j in range(LH):
                a1=cos(h[j],v1)
                a2=cos(h[j],v2)
                a3=cos(h[j],v3)
                a4=cos(h[j],v4)
                Mat_combined_nv[i1][i2][j]=-max(a1,a2,a3,a4)
                Mat_combined_L[i1][i2][j]=abs(gl[i1]+gl[i2]-hl[j])

    minn=100000
    P=[]
    I1=0
    I2=0

    for i1 in range(LG):
        print(i1)
        for i2 in range(LG):
            if i1==i2:
                continue
            Num=LH
            Dnv=np.zeros((Num,Num))
            DL=np.zeros((Num,Num))
            for j in range(LH):
                Dnv[j][0]=Mat_combined_nv[i1][i2][j]
                DL[j][0]=Mat_combined_L[i1][i2][j]
            ilist=[]
            for i in range(LG):
                if i!=i1 and i!=i2:
                    ilist.append(i)
            for j in range(LH):
                for i in range(len(ilist)):
                    Dnv[j][i+1]=Mat_nv_gh[j][ilist[i]]
                    DL[j][i+1]=Mat_L_gh[j][ilist[i]]
            for i in range(Num):
                Dnv[i,:]=(Dnv[i,:]-np.min(Dnv[i,:]))/(np.max(Dnv[i,:])-np.min(Dnv[i,:]))
                DL[i,:]=(DL[i,:]-np.min(DL[i,:]))/(np.max(DL[i,:])-np.min(DL[i,:]))
            D=Dnv+DL
            score,plan=search(D)
            if score<minn:
                minn=score
                P=plan
                I1=i1
                I2=i2
    print(I1,I2,minn,P)

    T2=time.time()
    print((T2-T1),"s")

if __name__=='__main__':
    main()
