import numpy as np
from numpy.random import random as rng
from baseFunctions import mergerRate,SNR,MonteCarloProb,D_L,E,z_max,DNSDistribution_z,D_c,D_H
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from constants import 
from scipy import integrate


DNSDistribution_z_Norm=integrate.quad(DNSDistribution_z,0,z_max())[0] 
DNSDistribution_z_vector = np.vectorize(DNSDistribution_z)


def DNSProb_z(z): ### dN/dz/Norm = dP/dz
    return mergerRate(z)/(1+z)*4*np.pi*D_c(z)**2*D_H/E(z)/DNSDistribution_z_Norm
    
ProbMax_z=DNSProb_z(z_max())

def randNS():
    cosTh=rng()*2 -1
    Theta=np.arccos(cosTh)
    Phi=2*np.pi*rng()
    cosIo=rng()*2 -1
    Iota=np.arccos(cosIo)
    Psi=2*np.pi*rng()
    randomZ=MonteCarloProb(DNSProb_z,(0,z_max()),(0,ProbMax_z))
    M=1
    Dl=D_L(randomZ)  
    return (SNR(Dl,Theta,Phi,Psi,Iota,m=1+randomZ),Dl,M,randomZ,M*(1+randomZ),Theta,Phi,Psi,Iota)
    
    
def generateSample(name,lowerLimit,sampleSize,LogLog=False,xrange=-1,parameterToReturn=0,norm=False):
    draws=0
#    print("draws= "+str(draws))
    k=0
    current=0 #progess print
    last=0 #progess print
    SNRs=[]
    sampleData=[]
    while(k<sampleSize):
        draws+=1
        randomSample=randNS()
        temp=randomSample[parameterToReturn]
        if(temp>lowerLimit): ####
            SNRs.append(temp)
            sampleData.append(randomSample)
            k+=1
            if(current!=last):
                print(str(int(k/sampleSize*100))+"%")
                last=current
            current=int(k/sampleSize*100)
    print("draws= "+str(draws))
    print("draws/samplesize= "+str(int(draws/k))) 
    SNRs=np.array(SNRs)
    sampleData=np.array(sampleData)
    binsNr=200
    file="pics/v2/"+name+"_bins_"+str(binsNr)+"_sample_"+str(sampleSize)
    plt.xlabel("SNR") ####
    plt.ylabel("dN/dSNR") ####
    plt.tick_params(direction= "inout",which="both")
    if(xrange!=-1):
        if(not LogLog):
            plt.xlim(xrange)
            SNRs=SNRs[SNRs<xrange[1]]
    if(LogLog):
#        print(max(SNRs))
        plt.xlim((10,1000)) #######
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.title(name+"_LogLog")
        plotData=plt.hist(SNRs,bins=np.logspace(start=np.log10(lowerLimit),stop=np.log10(max(SNRs)),num=binsNr),normed=norm)
#        print(max(SNRs))
#        print(binsNr)        
#        print(np.logspace(start=np.log10(lowerLimit),stop=np.log10(max(SNRs)),num=binsNr).size)
        file+="_LogLog"
    else:
        plt.title(name)
        plotData=plt.hist(SNRs,bins=binsNr,normed =norm) 
    plt.savefig(file,dpi=300)
    plt.show()
    np.savetxt("data/v2/"+name+"_sample_"+str(sampleSize)+".gz",sampleData)
    np.savetxt("data/v2/"+name+"_sample_"+str(sampleSize)+".txt",sampleData)
    return plotData

#data=generateSample("SNR_v2_1_massNormal_Dhorizon=10Gpc",8,100000,xrange=(5,50),LogLog=True,norm=False)
data=generateSample("SNR_v2_K800",8,100000,xrange=(5,100),LogLog=True,norm=False)
#data=generateSample("Dl_Dhorizon=10Gpc",0.1,100000,LogLog=True,parameterToReturn=1)




#
#ys=data[0]
#xs=data[1][:-1]
#print(xs.size)
#plt.plot(xs,ys)
#fitFun =lambda x,a: a*x**(-4)
#popt, pcov = curve_fit(f=fitFun, xdata=xs, ydata=ys,p0=(1))
#print(popt)
#print(np.sqrt(np.diag(pcov)))
#print(np.sqrt(np.diag(pcov))[0]/popt[0])
#plt.plot(xs,fitFun(xs,popt[0]))
##plt.xlim((8,50))
#plt.savefig("SNR_porownanieDhorizon=10Gpc",dpi=300)
#plt.show()