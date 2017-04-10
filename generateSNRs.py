import numpy as np
from numpy.random import random as rng
from baseFunctions import SNR,MonteCarloProb,D_L,DNSDistribution_z
from prepMasses import preComputeMasses
from pathlib import Path
import time
#from numba import jit

#@jit(nopython=True)
def randMassData():
    return massesData[np.random.randint(0,massesData.shape[0])]

#@jit(nopython=True)
def randNS(A,SU,mergerRateFunCoef):
    cosTh=rng()*2 -1
    Theta=np.arccos(cosTh)
    Phi=2*np.pi*rng()
    cosIo=rng()*2 -1
    Iota=np.arccos(cosIo)
    Psi=2*np.pi*rng()
    M,zMaxFromFile,probMaxFromFile,NormFromFile=randMassData()
    dnsd1=lambda z: DNSDistribution_z(z,lambda z:(1+z)**(mergerRateFunCoef),zMaxFromFile) #### DNSDistribution_z z zadanym mergerRateFun
    probZ=lambda z: dnsd1(z)/NormFromFile 
    randomZ=MonteCarloProb(probZ,(0,zMaxFromFile),(0,probMaxFromFile))
    Dl=D_L(randomZ)  
    return (SNR(Dl,Theta,Phi,Psi,Iota,M*(1+randomZ),A),Dl,M,randomZ,M*(1+randomZ),Theta,Phi,Psi,Iota)
          
def generateSample(name,size,A,mergerRateFunCoef,SU=True,\
    parameterToReturn=0,forceToPrecalc=False,saveFlag=True):
#    startTime = time.time()  
    print('A: ',A)
    sampleSize=size
    if(SU):
        file1='ABHBH02'
        fToCheckName="data/mass/"+file1+"_mass_A_"+str(A)+"_a"+str(float("%.3f"%(mergerRateFunCoef)))+".gz"
        my_file = Path(fToCheckName)
        if ((not my_file.is_file()) or forceToPrecalc):
            print("precomputing masses data for: "+fToCheckName)
            preComputeMasses(file1,A,mergerRateFunCoef)
        else:
            print('znaleziono plik')
        global massesData
        massesData=np.loadtxt(fToCheckName)
    
  
    draws,k,current,last,SNRs,sampleData=0,0,0,0,[],[]
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
    file+="_LogLog"
    if(saveFlag):
        np.savetxt("data/v3/"+name+"A"+str(A)+"_sample"+str(sampleSize)+".gz",sampleData)
#    doneTime = time.time()
#    print('Done in this many minutes: '+str((doneTime-startTime)/60))
    return sampleData

#ile=100000
#forceFlag=False
#startTime = time.time()   
#for a in range(0,2):
#    A=6400
#    print('a='+str(a)+' A='+str(A)+' mSU')
#    data=generateSample("SNRv3_mSU02_a"+str(a)+'_',ile,A=A,mergerRateFunCoef=a,forceToPrecalc=forceFlag)
##    print('a='+str(a)+' A='+str(8000)+' mSU')
##    data=generateSample("SNRv3_mSU02_a"+str(a)+'_',ile,A=8000,mergerRateFunCoef=a,SU=SUFlag,norm=normFlag,forceToPrecalc=forceFlag)
#doneTime = time.time()
#print('In total done in this many minutes: '+str((doneTime-startTime)/60))
###turn off PC:
##subprocess.call(["shutdown", "/s"])
