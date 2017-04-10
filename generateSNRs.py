import numpy as np
from scipy import integrate
from scipy.optimize import minimize
from numpy.random import random as rng
from baseFunctions import SNR,MonteCarloSampling,D_L,DNSDistribution,zMaxGet
from pathlib import Path
import time



def randMass():
    return massesData[np.random.randint(0,massesData.size)]

def randNS(A,mergerRateFunCoef,zmax,norm,probmax):
    cosTh=rng()*2 -1
    Theta=np.arccos(cosTh)
    Phi=2*np.pi*rng()
    cosIo=rng()*2 -1
    Iota=np.arccos(cosIo)
    Psi=2*np.pi*rng()
    M=randMass()
    probFun=lambda z:DNSDistribution(z,lambda x:(1+x)**mergerRateFunCoef,zmax)
    randomZ=MonteCarloSampling(probFun,(0,zmax),(0,probmax))
    Dl=D_L(randomZ)  
    return (SNR(Dl,Theta,Phi,Psi,Iota,M*(1+randomZ),A),Dl,M,randomZ,M*(1+randomZ),Theta,Phi,Psi,Iota)
          
def generateSample(name,sampleSize,A,mergerRateFunCoef,paramToReturn=0,saveFlag=True):
    startTime = time.time()  
    print('A: ',A)
    file1='ABHBH02'
    fToCheckName="data/mass/"+file1+"_mass.gz"
    my_file = Path(fToCheckName)
    if (not my_file.is_file()):
        print("nie znaleziono pliku z masami")
        return -1
    else:
        print('znaleziono plik')
        global massesData
        massesData=np.loadtxt(fToCheckName)
        
    zMaxGeneral=zMaxGet(A,max(np.loadtxt('data/mass/'+file1+'_mass.gz')))
    print('zMaxGeneral',zMaxGeneral)
    DNSonlyZ=lambda z: DNSDistribution(z,lambda x:(1+x)**mergerRateFunCoef,zMaxGeneral)
    normGeneral=integrate.quad(DNSonlyZ,0,zMaxGeneral)
    t=minimize(lambda z:-1*DNSonlyZ(z)/normGeneral,0.01,bounds=((0,zMaxGeneral),))
    probMaxGeneral=-1*t.fun[0]
    
    draws,k,current,last,SNRs,sampleData=0,0,0,0,[],[]
    while(k<sampleSize):
        draws+=1
        randomSample=randNS(A,mergerRateFunCoef,zMaxGeneral,normGeneral,probMaxGeneral)
        temp=randomSample[paramToReturn]
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
    file+="_LogLog"
    if(saveFlag):
        np.savetxt("data/v3/"+name+"A"+str(A)+"_sample"+str(sampleSize)+".gz",sampleData)
    doneTime = time.time()
    print('Done in this many minutes: '+str((doneTime-startTime)/60))
    return sampleData

ile=100000
startTime = time.time()   
for a in range(0,2):
    A=800
    print('a='+str(a)+' A='+str(A)+' mSU')
    data=generateSample("SNRv3_mSU02_a"+str(a)+'_',ile,A=A,mergerRateFunCoef=a)
#    print('a='+str(a)+' A='+str(8000)+' mSU')
#    data=generateSample("SNRv3_mSU02_a"+str(a)+'_',ile,A=8000,mergerRateFunCoef=a,SU=SUFlag,norm=normFlag,forceToPrecalc=forceFlag)
doneTime = time.time()
print('In total done in this many minutes: '+str((doneTime-startTime)/60))

"""
##turn off PC:
#subprocess.call(["shutdown", "/s"])
"""
