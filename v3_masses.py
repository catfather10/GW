##dodano masy z syntheticuniverse
import numpy as np
from numpy.random import random as rng
from baseFunctions import SNR,MonteCarloProb,D_L,z_max,DNSDistribution_z,mergerRate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
from constants import A

#ProbMax_z=DNSProb_z(z_max())

#masses=np.loadtxt("data/ABHBH02_masses.gz")
#masses=np.loadtxt("data/oneMass.txt")
#def randMass():
#    return(np.random.choice(masses))
   

def randNS():
    cosTh=rng()*2 -1
    Theta=np.arccos(cosTh)
    Phi=2*np.pi*rng()
    cosIo=rng()*2 -1
    Iota=np.arccos(cosIo)
    Psi=2*np.pi*rng()
    M=1
#    M=randMass()
    DNSDistribution_z_Norm=integrate.quad(DNSDistribution_z,0,z_max(M))[0] ###uwzglednia rozne zmax dla roznych mas
    probZ=lambda z: DNSDistribution_z(z)/DNSDistribution_z_Norm 
    ProbMax_z=probZ(z_max(M))
    randomZ=MonteCarloProb(probZ,(0,z_max(M)),(0,ProbMax_z))
    Dl=D_L(randomZ)  
#    print(z_max(M))
#    print(SNR(Dl,Theta,Phi,Psi,Iota,m=M*(1+randomZ)))
    return (SNR(Dl,Theta,Phi,Psi,Iota,m=M*(1+randomZ)),Dl,M,randomZ,M*(1+randomZ),Theta,Phi,Psi,Iota)
    
    
def generateSample(name,lowerLimit,sampleSize,LogLog=False,xrange=-1,parameterToReturn=0,norm=False):
    binsNr,draws,k,current,last,SNRs,sampleData=200,0,0,0,0,[],[]
    while(k<sampleSize):
        draws+=1
        randomSample=randNS()
        temp=randomSample[parameterToReturn]
        if(temp>lowerLimit):
            SNRs.append(temp)
            sampleData.append(randomSample)
            k+=1
            if(current!=last):
                print(str(int(k/sampleSize*100))+"%")
                last=current
            current=int(k/sampleSize*100)
    print("draws= "+str(draws)+" draws/samplesize= "+str((draws/k))) 
    SNRs,sampleData=np.array(SNRs),np.array(sampleData)
    file="pics/v3/"+name+"A"+str(A)+"_sample"+str(sampleSize)
    plt.xlabel("SNR")
    if(norm):
        plt.ylabel("dP/dSNR")
    else:
        plt.ylabel("dN/dSNR")
    plt.tick_params(direction= "inout",which="both")
    if(xrange!=-1):
        if(not LogLog):
            plt.xlim(xrange)
            SNRs=SNRs[SNRs<xrange[1]]
    if(LogLog):

        plt.xlim((10,1000)) 
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.title(name+"A"+str(A)+"_LogLog")
        plotData=plt.hist(SNRs,bins=np.logspace(start=np.log10(lowerLimit),stop=np.log10(max(SNRs)),num=binsNr),normed=norm)
        file+="_LogLog"
    else:
        plt.title(name+"A"+str(A))
        plotData=plt.hist(SNRs,bins=binsNr,normed =norm) 
    plt.savefig(file,dpi=300)
    plt.show()
    np.savetxt("data/v3/"+name+"A"+str(A)+"_sample"+str(sampleSize)+".gz",sampleData)
#    np.savetxt("data/v3/"+name+"_sample"+str(sampleSize)+".txt",sampleData)
    return plotData


#data=generateSample("SNRv3_M1_NNorm",8,100000,xrange=(5,100),LogLog=True,norm=True)
#data=generateSample("SNRv3_M1_Norm",8, 100000,xrange=(5,100),LogLog=True,norm=False)

#tralalalal

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