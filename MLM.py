import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from generateSNRs import generateSample
from histograms import histAndSaveLogLog
from scipy import integrate
import time

def round_down(num, divisor):
    return num - (num%divisor)
    
A=800  
def interpolateSNR(modelCoef):
    dataModel=np.loadtxt("SNR/"+str(A)+"/SNRv3_mSU02_a"+str(modelCoef)+"_A"+str(A)+"_sample100000.txt")[:,0]
    binsNr=201
    histData=plt.hist(dataModel,bins=np.logspace(start=np.log10(min(dataModel)),stop=np.log10(max(dataModel)),\
        num=binsNr),normed=True)
    plt.show()
    plt.clf()
    ys=histData[0]
    xs=histData[1][:-1]
    #ys=[i for i in ys if i !=0]
    ys2=[]
    xs2=[]
    #print(len(xs),len(ys))
    for i in range (len(xs)):
        if(ys[i]!=0):
            ys2.append(ys[i])
            xs2.append(xs[i])
    ys=ys2#/np.sum(ys2)
    print('sum ', np.sum(ys2))
    xs=xs2
    #print(len(xs),len(ys))
    #print(min(ys))
    xInterpolatedMax=max(xs)
    xInterpolatedMin=min(xs)
    f1=interp1d(xs,ys,kind='linear')
    return f1,xInterpolatedMin,xInterpolatedMax
    
f0,xInterpolatedMin0,xInterpolatedMax0=interpolateSNR(modelCoef=0)
xs1=np.linspace(xInterpolatedMin0,xInterpolatedMax0,num=6000)
ys1=f0(xs1)
print('simps int: ',integrate.simps(ys1,xs1))
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.plot(xs1,ys1,'g')
plt.savefig('model0.png')

#binsNr=600
#data=np.loadtxt('SNR/1600/SNRv3_mSU02_a0_A1600_sample100000.gz')[:,0]
#histData=plt.hist(data,bins=np.logspace(start=np.log10(min(data)),stop=np.log10(max(data)),\
#        num=binsNr),normed=True)


f1,xInterpolatedMin1,xInterpolatedMax1=interpolateSNR(modelCoef=1)
xs1=np.linspace(xInterpolatedMin1,xInterpolatedMax1,num=6000)
ys1=f1(xs1)
print('simps int: ',integrate.simps(ys1,xs1))
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.plot(xs1,ys1,'r')
plt.savefig('model1.png')
plt.show()
plt.clf()


xInterpolatedMin=max(xInterpolatedMin1,xInterpolatedMin0)
xInterpolatedMax=min(xInterpolatedMax1,xInterpolatedMax0)


def MLM(size,samples):
    print('size ',size,' samples ',samples)
    startTime = time.time()
    Os=[]
    dataSample1=generateSample("test",size*samples,A=A,mergerRateFunCoef=1,saveFlag=False)[:,0]
    print('a1',[i for i in dataSample1 if i <=0])
    #usuwanie SNRow z poza zakresu interpolacji
    dataSample1=[i for i in dataSample1 if (i <= xInterpolatedMax and i>=xInterpolatedMin)]
    newSamples=len(dataSample1)
    dataSample1=dataSample1[0:round_down(newSamples,size)]
    print('newsize',newSamples)
    print('b1',[i for i in dataSample1 if i <=0])
    for t in range(int(newSamples/size)):
#        print(t)
        slice1=dataSample1[t*size :(t+1)*size]
        Ps1=f1(slice1) 
        Ps0=f0(slice1) 
        L1=np.prod(Ps1)
        L0=np.prod(Ps0)
#        print(L0,L1)
        if(L1/L0<=0):
            print(L1,L0)
        Os.append(L1/L0)
    print('done')
    print([i for i in Os if i <=0])
#    print(Os)
    histAndSaveLogLog(Os,'Bayes Factor',r'$\frac{dN}{dO}$','Bayes Factors A: '+str(A)+' sampleSize: '+str(size)+' samples: '+str(samples),\
    'MLM/'+str(A)+'/BayesFactors_'+str(A)+'_'+str(size)+'_'+str(samples)+'.png',yLog=False)
    np.savetxt('MLM/'+str(A)+'/BayesFactors_'+str(A)+'_'+str(size)+'_'+str(samples)+'.txt',Os)
    doneTime = time.time()
    print('In total done in this many minutes: '+str((doneTime-startTime)/60))
    title='Bayes Factors A: '+str(A)+' sampleSize: '+str(size)+' samples: '+str(samples)
    histAndSaveLogLog(Os,'Bayes Factor',r'$\frac{dN}{dO}$',title,\
        'MLM/'+str(A)+'/BayesFactors_'+str(A)+'_'+str(size)+'_'+str(samples)+'.png',yLog=False)
    return Os
    

print('loaded')
samples=10000
for t in [60,100]:
    MLM(size=t,samples=samples)
A=1600    
for t in [3,6,10,30,60,100]:
    MLM(size=t,samples=samples)
import subprocess
subprocess.call(["shutdown", "/s"])


