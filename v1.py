import numpy as np
from numpy.random import random as rng
from constants import D_horizon
from baseFunctions import SNR
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#bez kosmologii, nadaje sie do rozkladow zaleznych tylko do Dl i do rysowania rozkładu katowego TH

    
def randNSv1(r=-1,Rfun=-1):
    cosTh=rng()*2 -1
    Theta=np.arccos(cosTh)
    Phi=2*np.pi*rng()
    cosIo=rng()*2 -1
    Iota=np.arccos(cosIo)
    Psi=2*np.pi*rng()
    if(r==-1):
        if(Rfun==-1):
            R=D_horizon*rng()**(1/3) 
        else:
            R=Rfun() 
    else:
        R=r
    return (SNR(R,Theta,Phi,Psi,Iota),R,Theta,Phi,Psi,Iota)
    

def generateSample(name,lowerLimit,sampleSize,D=-1,xrange=-1,LogLog=False):
    draws=0
    print("draws= "+str(draws))
    k=0
    SNRs=[]
    while(k<=sampleSize):
        draws+=1
        temp=randNSv1(D)[0]
        if(temp>lowerLimit): ####
#            print(temp)
            SNRs.append(temp)
            k+=1
            if(k%1000==0):
                print(str(k/sampleSize*100)+"%")
    print("draws= "+str(draws))
    print("draws/samplesize= "+str(int(draws/k))) 
    SNRs=np.array(SNRs)
    
    binsNr=200
    file="pics/v1/"+name+"_bins_"+str(binsNr)+"_sample_"+str(sampleSize)
    if(xrange!=-1):
        if(not LogLog):
            plt.xlim(xrange)
            SNRs=SNRs[SNRs<xrange[1]]
    if(LogLog):
#        print(max(SNRs))
        plt.xlim
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        data=plt.hist(SNRs,bins=np.logspace(start=np.log10(lowerLimit),stop=np.log10(max(SNRs)),num=binsNr))
        file+="_log"
    else:
       data=plt.hist(SNRs,bins=binsNr) 

    plt.savefig(file,dpi=300)
    plt.show()
    return data
    
    
### test w postaci fitu do x^-4
#data=generateSample("SNR_test",8,10000,xrange=(8,50))
#ys=data[0]
#xs=data[1][:-1]
#plt.plot(xs,ys)
#fitFun =lambda x,a: a*x**(-4)
#popt, pcov = curve_fit(f=fitFun, xdata=xs, ydata=ys,p0=(1))
#print(popt)
#print(np.sqrt(np.diag(pcov)))
#print(np.sqrt(np.diag(pcov))/popt)
#plt.plot(xs,fitFun(xs,popt[0]))
#plt.xlim((8,50))
#plt.show()
    
data=generateSample("SNR_v1",8,100000,xrange=(5,50),LogLog=True)
data=generateSample("SNR_v1",8,100000,xrange=(5,50))





