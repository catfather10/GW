##dodano masy z syntheticuniverse
#import subprocess
import numpy as np
from numpy.random import random as rng
from baseFunctions import SNR,MonteCarloProb,D_L,DNSDistribution_z#,z_max
import matplotlib.pyplot as plt
from prepMasses import preComputeMasses
#from scipy.optimize import curve_fit
#from scipy import integrate
from pathlib import Path
import time

def randMassData():
    return massesData[np.random.randint(0,massesData.shape[0])]


def randNS(A,SU,mergerRateFunCoef):
    cosTh=rng()*2 -1
    Theta=np.arccos(cosTh)
    Phi=2*np.pi*rng()
    cosIo=rng()*2 -1
    Iota=np.arccos(cosIo)
    Psi=2*np.pi*rng()
    M=1
    if(SU):
        M,zMaxFromFile,probMaxFromFile,NormFromFile=randMassData()
        dnsd1=lambda z: DNSDistribution_z(z,lambda z:(1+z)**(mergerRateFunCoef),zMaxFromFile) #### DNSDistribution_z z zadanym mergerRateFun
        probZ=lambda z: dnsd1(z)/NormFromFile 
    randomZ=MonteCarloProb(probZ,(0,zMaxFromFile),(0,probMaxFromFile))
#    print("przed liczeniem dl")
    Dl=D_L(randomZ)  
#    print("po liczeniem dl")
    return (SNR(Dl,Theta,Phi,Psi,Iota,M*(1+randomZ),A),Dl,M,randomZ,M*(1+randomZ),Theta,Phi,Psi,Iota)
          
def generateSample(name,sampleSize,A,mergerRateFunCoef,SU=False,LogLog=True,parameterToReturn=0,norm=False,forceToPrecalc=False):
    startTime = time.time()    
    if(SU):
        file1='ABHBH02'
        fToCheckName="data/mass/"+file1+"_mass_A_"+str(A)+"_a"+str(float("%.3f"%(mergerRateFunCoef)))+".txt"
        my_file = Path(fToCheckName)
        if ((not my_file.is_file()) or forceToPrecalc):
            print("precomputing masses data for: "+fToCheckName)
            preComputeMasses(file1,A,mergerRateFunCoef)
            
        else:
            print('znaleziono plik')
        global massesData
        massesData=np.loadtxt(fToCheckName)
    
    
    binsNr,draws,k,current,last,SNRs,sampleData=200,0,0,0,0,[],[]
    while(k<sampleSize):
        draws+=1
        randomSample=randNS(A,SU,mergerRateFunCoef)
        temp=randomSample[parameterToReturn]
        if(temp>8):
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
    if(LogLog):
        plt.xlim((10,1000))
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.title(name+"A"+str(A)+"_LogLog")
        plotData=plt.hist(SNRs,bins=np.logspace(start=np.log10(8),stop=np.log10(max(SNRs)),num=binsNr),normed=norm)
        file+="_LogLog"
    else:
        plt.title(name+"A"+str(A))
        plotData=plt.hist(SNRs,bins=binsNr,normed =norm) 
    plt.savefig(file,dpi=300)
    plt.show()
    plt.clf()
    np.savetxt("data/v3/"+name+"A"+str(A)+"_sample"+str(sampleSize)+".gz",sampleData)
#    np.savetxt("data/v3/"+name+"_sample"+str(sampleSize)+".txt",sampleData)
    doneTime = time.time()
    print('Done in this many minutes: '+str((doneTime-startTime)/60))
    return plotData

ile=100000
normFlag=True
SUFlag=False
forceFlag=False
startTime = time.time()   
for a in range(0,4):
#    SUFlag=False
#    print('a='+str(a)+' A='+str(8000)+' m1')
#    data=generateSample("SNRv3_m1_a"+str(a)+'_',ile,A=8000,mergerRateFunCoef=a,SU=SUFlag,norm=normFlag)
#    print('a='+str(a)+' A='+str(800)+' m1')
#    data=generateSample("SNRv3_m1_a"+str(a)+'_',ile,A=800,mergerRateFunCoef=a,SU=SUFlag,norm=normFlag)
    SUFlag=True
    print('a='+str(a)+' A='+str(800)+' mSU')
    data=generateSample("SNRv3_mSU02_a"+str(a)+'_',ile,A=800,mergerRateFunCoef=a,SU=SUFlag,norm=normFlag,forceToPrecalc=forceFlag)
    print('a='+str(a)+' A='+str(8000)+' mSU')
    data=generateSample("SNRv3_mSU02_a"+str(a)+'_',ile,A=8000,mergerRateFunCoef=a,SU=SUFlag,norm=normFlag,forceToPrecalc=forceFlag)
doneTime = time.time()
print('In total done in this many minutes: '+str((doneTime-startTime)/60))
#turn off PC:
#subprocess.call(["shutdown", "/s"])
