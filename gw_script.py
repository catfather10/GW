#!/home/mossowski/python-virtualenvs/myenv3.5/bin/python3

####
from scipy import integrate
import os
import numpy as np
from numpy.random import random as rng
from scipy.optimize import fsolve,brentq,minimize
import scipy.io
from numba import jit
from scipy.interpolate import interp1d

##### moje importy
from InterpPdf import InterpPdf
from logInterp import logInterp
from pathlib import Path
import time

#### SAVE ORDER
## SNR,Dl,M,z,Th,Ph,Ps,Io


##### stałe
H_0=70.4 # km/s  Mpc^-1
Omega_m=0.27 
Omega_lambda=1-Omega_m
c=299792 # km/s
D_H=c/H_0

print(os.getcwd())

@jit(nopython=True) 
def E(z):
    return np.sqrt(Omega_m*(1+z)**3+(1-Omega_m))
    
def D_c(z):
    z2=lambda z: 1/E(z)
    return D_H*integrate.quad(z2,0,z)[0] 

@jit
def D_L(z):
    return (1+z)*D_c(z)

def zMaxGet(detector,m):
  #  print('det',detector)
    if(detector=='AdLigoDesign'):
        detSen=AdLigoDesign
    elif(detector=='AdLigoH1'):
        detSen=AdLigoH1
        #print('wybrano detector ADL')
    elif(detector=='ET'):
        #print('wybrano detector ET')
        detSen=ET
    else:
        
        print('zly detector from zMaxGet')
    toSolve = lambda z:detSen(m*(1+z))/D_L(z)-2
    return fsolve(toSolve,.1)[0]
        
def binaryDistribution(z,mergerRateFun,zMax): ### dN/dz
    if(zMax>10):
        zMax=10
    if(z<=zMax):
        return mergerRateFun(z)*4*np.pi*D_H*(D_c(z))**2/(E(z)*(1+z))
    else:
        return 0
                
@jit(nopython=True)
def SNR(Dl,theta,phi,psi,iota,A):
    FP=0.5*(1+(np.cos(theta))**2)*np.cos(2*phi)*np.cos(2*psi)-np.cos(theta)*np.sin(2*phi)*np.sin(2*psi)
    FX=0.5*(1+(np.cos(theta))**2)*np.cos(2*phi)*np.sin(2*psi)+np.cos(theta)*np.sin(2*phi)*np.cos(2*psi)
    TH=2*np.sqrt(((FP*(1+(np.cos(iota))**2))**2) +4*(FX*np.cos(iota))**2) 
    return A*TH/Dl
   

def MonteCarloSampling(probFun,xRange,yRange):
    k=0
    while (True):
        k+=1
        x=rng()*(xRange[1]-xRange[0])+xRange[0]
        y=rng()*(yRange[1]-yRange[0])+yRange[0]
        if(probFun(x)>y):
            return x
            
    
def inverseSampling(invCDF):
    u=rng()
    return invCDF(u)

def rejectionEnvelopeSampling(probFun,intpdf):
    k=0
    while (True):
        k+=1
        u=rng()
        x=inverseSampling(intpdf)
        if(u<probFun(x)/(intpdf.pdf(x)*intpdf.norm)):
            return x  
            
def prepCustomSamplerEnvelope(massFile,points,detector,MRD,modelName=''):
    mmax=max(np.loadtxt('mass/'+massFile+'_mass.txt'))
    zm=zMaxGet(detector,mmax)

    if(zm>10):
        zm=10
    xs=np.logspace(np.log10(0.001),np.log10(zm),num=1000)

    xspoints=points[0]
    yspoints=points[1]

    fint=InterpPdf(xspoints,yspoints)
    ysInterp=[]
    for x in xs:
        ysInterp.append(fint(x))
        

    vDNSDistribution=np.vectorize(binaryDistribution)

    norm=integrate.quad(lambda z:binaryDistribution(z,MRD,zm),0,zm)[0]
    f=lambda x:vDNSDistribution(x,MRD,zm)/norm
    for i in xs:
        if(fint(i)<f(i)):
            print('zla obwiadnia',' tu sie psuje',i)
            return -1
    return fint

def prepAdLigoDesign():
    data = np.genfromtxt('data/AdLigoSensitivityDesign.csv', delimiter=',')
    AdLigo=logInterp(data[:,0],data[:,1])
    return lambda z:AdLigo(z)*2 

def prepAdLigoH1():
    data = np.genfromtxt('data/AdLigoSensitivityH1.csv', delimiter=',')
    AdLigo=logInterp(data[:,0],data[:,1])
    return lambda z:AdLigo(z)*2




def prepET():
    data = np.genfromtxt('data/ET_M.csv', delimiter=',')
    xs=data[:,0]*.44
    ys=data[:,1]*4/2.26
    ET=logInterp(xs,ys)
    return lambda z:ET(z)*2 

def makeGCmergerRateDensity():
    data=np.loadtxt('data/GCrdensity.dat')
    xs=data[0:1002,0]
    ys=data[0:1002,2]
    f=interp1d(xs,ys)
    ys=0
    return f

def randMass():
    return massesData[np.random.randint(0,massesData.size)]

def randNS(detector,redshiftSampler):
    cosTh=rng()*2 -1
    Theta=np.arccos(cosTh)
    Phi=2*np.pi*rng()
    cosIo=rng()*2 -1
    Iota=np.arccos(cosIo)
    Psi=2*np.pi*rng()
    M=randMass()
    randomZ=redshiftSampler()
    Dl=D_L(randomZ)  
    if(detector=='AdLigoDesign'):
        A=AdLigoDesign(M*(1+randomZ))
    elif(detector=='AdLigoH1'):
        A=AdLigoH1(M*(1+randomZ))
    elif(detector=='ET'):
        A=ET(M*(1+randomZ))
    else:
        print('zly detector:', detector)
        return 0
    return (SNR(Dl,Theta,Phi,Psi,Iota,A) ,Dl,M,randomZ,Theta,Phi,Psi,Iota)
          
def generateSample(name,sampleSize,detector,mergerRateFun,massFName,\
                    saveFlag=True,savePath=''):
    print('########## generateSample ##########')
    print(detector)
    startTime = time.time()  
    print('massFName',massFName)
    fNameToCheck="mass/"+massFName+".txt"
    my_file = Path(fNameToCheck)
    if (not my_file.is_file()):
        print(os.getcwd())
        print("nie znaleziono pliku z masami")
        return -1
    else:
        print('znaleziono plik z masami')
        global massesData
        massesData=np.loadtxt(fNameToCheck)

        
    zMaxGeneral=zMaxGet(detector,max(np.loadtxt('mass/'+massFName+'.txt')))
    if (zMaxGeneral>10):
        zMaxGeneral=10
    print('zMaxGeneral',zMaxGeneral)
    
    DNSonlyZ=lambda z: binaryDistribution(z,mergerRateFun,zMaxGeneral)
    normGeneral=integrate.quad(DNSonlyZ,0,zMaxGeneral)[0]
    toMinimize=minimize(lambda z:-1*DNSonlyZ(z)/normGeneral,0.01,bounds=((0,zMaxGeneral),))
    probMaxGeneral=-1*toMinimize.fun[0]
    zPDF=lambda z:binaryDistribution(z,mergerRateFun,zMaxGeneral)/normGeneral
    vectzPDF=np.vectorize(zPDF)


    ## nowy invserse sampling
    zToInv=np.linspace(0,zMaxGeneral,num=1000,endpoint=True)
    yToInv=vectzPDF(zToInv)
    integs, yy=[], []
    for i in range(len(zToInv)):
        integs.append(integrate.simps(yToInv[:i+1],zToInv[:i+1])) ## tablicowanie CDF
    zCDF=interp1d(zToInv,integs)
     
    zzz=np.linspace(0,zMaxGeneral,1e5,endpoint=True)
    zz=np.linspace(0,zMaxGeneral,num=1e3)
    ts=np.linspace(0,1,num=1e3,endpoint=False)

    for t in ts:
        toSolve =lambda z: zCDF(z)-t
        yy.append(brentq(toSolve,a=0,b=zMaxGeneral,disp=True)) ### tablicowanie invCDF
    yy.append(zMaxGeneral)
    ts=np.append(ts,1)
    invCDF=interp1d(ts,yy)
    sampler = lambda :invCDF(rng())

    tohist=[]
    for i in range(10**5):
        tohist.append(sampler())
        """
    import matplotlib.pyplot as plt
    plt.hist(tohist,bins=100,normed=True)
    plt.plot(zz,vectzPDF(zz)/integrate.simps(vectzPDF(zz),zz))
    plt.xlabel('z')
    plt.savefig(name+'.png',dpi=300)
    plt.show()
   """
    #if(customSamplerEnvelope==False):
    #    print('MonteCarloSampler')
    #    sampler=lambda pdf: MonteCarloSampling(pdf,(0,zMaxGeneral+.1),(0,probMaxGeneral+.1))
    #else:
    #    print("using envelope rejection sampling")
    #    sampler= lambda pdf:rejectionEnvelopeSampling(pdf,customSamplerEnvelope)
    
    draws,k,current,last,SNRs,sampleData=0,0,0,0,[],[]
    while(k<sampleSize):
        draws+=1
        randomSample=randNS(detector,sampler)
        tempSNR=randomSample[0]
        if(tempSNR>8):
            SNRs.append(tempSNR)
            sampleData.append(randomSample)
            k+=1
            if(current!=last):
                print('\rComplete: '+str(int(k/sampleSize*100))+"%",end="")
                last=current
            current=int(k/sampleSize*100)
            
    print("draws= "+str(draws)+" draws/samplesize= "+str((draws/k))) 
    SNRs,sampleData=np.array(SNRs),np.array(sampleData)
    if(saveFlag):
        if(savePath!=''):
                oldPath=os.getcwd()
                os.chdir(savePath)
        if(not os.path.isdir('SNR')):
                os.mkdir('SNR')
        if(not os.path.isdir('SNR/'+detector)):
                os.mkdir('SNR/'+detector)
        np.savetxt("SNR/"+detector+"/"+name+detector+"_sample"+str(sampleSize)+".gz",sampleData)
        os.chdir(oldPath)
        #np.savetxt("SNR/"+detector+"/"+name+detector+"_sample"+str(sampleSize)+".txt",SNRs)
    doneTime = time.time()
    print('Done in this many minutes: '+str((doneTime-startTime)/60))
    massesData=0
    #return SNRs

def makeCHEmergerRate():
    data = scipy.io.loadmat('data/DefaultBinaries.mat')
    zs=data['zvec'][0]
    qq=data['MergerRate']
    s=np.sum(qq,axis=0)

    f=interp1d(zs,s)
    return f

def makeSFR():
    data=np.loadtxt('data/SFR.dat')
    xs=data[:,0]
    ys=data[:,1]
    f=interp1d(xs,ys)
    ys=0
    return f


def sklejka(model,part,path,indices,totalFiles=116):
        files=[]
        for i in range(part*totalFiles,(part+1)*totalFiles):
            print(i)
            data=np.loadtxt(path+'SNR/AdLigoH1/snrH1-'+str(i)+model+'AdLigo_sample500000.gz')[:,indices]
            files.append(data)
        masterFile=np.concatenate(files)
        np.savetxt(path+'SNR/AdLigoH1/SNR_'+model+'master'+str(part)+str(indices)+'.gz',masterFile)
        

###################### SETUP
CHEMRD=makeCHEmergerRate()
GCMRD=makeGCmergerRateDensity()
SFR=makeSFR()
AdLigoDesign=prepAdLigoDesign()
AdLigoH1=prepAdLigoH1()
ET=prepET()

######## envelopes
#envelopeSFRAdLigoDesign=prepCustomSamplerEnvelope('SU02',([0,.2,.4,0.69],[0.1,.5,2,4.5]),'AdLigoDesign',SFR,"SFR")
#envelopeGCAdLigoDesign=prepCustomSamplerEnvelope('GC',([0,1.25,1.65,1.98],[0,.7,1.3,1.1]),'AdLigoDesign',GCMRD,"GC")
#envelopea0AdLigoDesign=prepCustomSamplerEnvelope('SU02',([0,.12,0.69],[0,.4,3.5]),'AdLigoDesign',lambda z:1,"a0")
#envelopeCHEAdLigoDesign=prepCustomSamplerEnvelope('CHE',([0,.5,.7,1.2,1.87],[0,1.7,1.6,.2,.05]),'AdLigoDesign',CHEMRD,"CHE")

#envelopea0ET=prepCustomSamplerEnvelope('SU02',([0,.9,2,6,10],[0,.22,.23,.072,.03]),'ET',lambda z:1,"a0")
#envelopeSFRET=prepCustomSamplerEnvelope('SU02',([0,1.6,2.2,4,6,10],[0,.39,.38,.125,.03,0.01]),'ET',SFR,"SFR")
#envelopeGCET=prepCustomSamplerEnvelope('GC',([0,1,1.7,2.7,6,10],[0,0.2,.55,0.25,.034,0]),'ET',GCMRD,"GC")
#envelopeCHEET=prepCustomSamplerEnvelope('CHE',([0,.5,.65,1.2,1.7,10],[0,1.7,1.55,.3,0,0]),'ET',CHEMRD,"CHE")


##### tescik

print('########## koniec baseFun ##########')
