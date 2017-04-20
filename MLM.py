import matplotlib.pyplot as plt
import numpy as np
from baseFunctions import SFR
from generateSNRs import generateSample
from histograms import histAndSaveLogLog
from scipy import integrate
import bisect
import time
print('########## MLM ##########')
A=16000
size=100000
binsNr=200
#numberOfObservations=400
print('binsNr',binsNr)
def round_down(num, divisor):
    return num - (num%divisor)
#### losowanie probek z modelu 1 !
###model 0
mFun0= lambda z: 1
name0='a0'
###model 1 SFR
mFun1=SFR
name1='SFR'
###
names=name0+'_'+name1



def interpolateSNR(model):
    if(model==0):
        modelName=name0
    elif(model ==1):
        modelName=name1
    else:
        print('zly model')
        return -1
    dataModel=np.loadtxt("SNR/"+str(A)+"/SNR_mSU02_"+modelName+"_A"+str(A)+"_sample100000.txt")[:,0]
#    dataModel=dataModel[0:numberOfObservations]
    histData=np.histogram(dataModel,bins=np.logspace(start=np.log10(min(dataModel)),stop=np.log10(max(dataModel)),\
        num=binsNr),normed=False)
    ysInt=histData[0]
    xsInt=histData[1]
    print('poczatek histogramu',xsInt[0])
    print(modelName, sum(ysInt))
    for i in range (len(ysInt)):
        if(ysInt[i]==0):
            ysInt[i]=ysInt[i-1]

    xInterpolatedMax=max(xsInt)
    xInterpolatedMin=min(xsInt)
    
    def prob(x):
        size=100000
        b=bisect.bisect_right(xsInt,x)-1
#        if(b==0):
#            print('b=0')
        if(b>=ysInt.size):
#            print('b=maxSize')
            b=ysInt.size-1
        binWidth=xsInt[b+1]-xsInt[b]
        toReturn=ysInt[b]/size/binWidth
        return  toReturn 
        
    f1=np.vectorize(prob)
    print('norm inside interpolate',integrate.simps(f1(xsInt[:-1]),xsInt[:-1]))
    return f1,xInterpolatedMin,xInterpolatedMax
    
f0,xIntMin0,xIntMax0=interpolateSNR(0)
f1,xIntMin1,xIntMax1=interpolateSNR(1)

xIntMax=min(xIntMax0,xIntMax1)
xIntMin=max(xIntMin0,xIntMin1)
xs=np.linspace(xIntMin,xIntMax,num=10000)
ys00=f0(xs)
ys01=f1(xs)

norm00=integrate.simps(ys00,xs)
norm01=integrate.simps(ys01,xs)
print('przed normalizacja')
print(norm00)
print(norm01)

f0n=lambda x:f0(x)/norm00
f1n=lambda x:f1(x)/norm01

ys00=f0n(xs)
ys01=f1n(xs)

print('po normalizacji')
print(integrate.simps(ys00,xs))
print(integrate.simps(ys01,xs))

plt.rc('text', usetex=True)
plt.plot(xs,ys00,'r',label=r'model $\alpha$ =0')
plt.plot(xs,ys01,'c',label=r'SFR')
plt.title('Probability density function for detector A='+str(A))
plt.loglog()
plt.legend()
plt.savefig('Probability density function for detector A'+str(A)+names,dpi=500)
plt.clf()
plt.rc('text', usetex=False)





def MLM(size,samples):
    print('size ',size,' samples ',samples)
    startTime = time.time()
    Os=[]
    dataSample1=generateSample("test",size*samples,A=A,mergerRateFun=mFun1,saveFlag=False,customSampler=True)[:,0]
    print('a1',[i for i in dataSample1 if i <=0])
    #usuwanie SNRow z poza zakresu interpolacji
    dataSample1=[i for i in dataSample1 if (i <= xIntMax and i>=xIntMin)]
    newSamples=len(dataSample1)
    dataSample1=dataSample1[0:round_down(newSamples,size)]
    print('newsize',newSamples)
    print('b1',[i for i in dataSample1 if i <=0])
    for sampleIndex in range(int(newSamples/size)):
#        print(sampleIndex)
        O=1
        part=dataSample1[sampleIndex*size :(sampleIndex+1)*size]
        for p in part:
            O*=f1n(p)
            O/=f0n(p)
        if(O<=0):
            print('O <= 0',O)
        Os.append(O)
    print('done')
    print([i for i in Os if i <=0])
    histAndSaveLogLog(Os,'Bayes Factor',r'$\frac{dN}{dO}$','Bayes Factors A: '+str(A)+' sampleSize: '+str(size)+' samples: '+str(samples)+'/'+names,\
    'MLM/'+str(A)+'/'+names+'/BayesFactors_'+str(A)+'_'+str(size)+'_'+str(samples)+'.png',yLog=False)
    np.savetxt('MLM/'+str(A)+'/'+names+'/BayesFactors_'+str(A)+'_'+str(size)+'_'+str(samples)+'.txt',Os)
    doneTime = time.time()
    print('In total done in this many minutes: '+str((doneTime-startTime)/60))
    title='Bayes Factors A: '+str(A)+' sampleSize: '+str(size)+' samples: '+str(samples)
    histAndSaveLogLog(Os,'Bayes Factor',r'$\frac{dN}{dO}$',title,\
        'MLM/'+str(A)+'/'+names+'/BayesFactors_'+str(A)+'_'+str(size)+'_'+str(samples)+'.png',yLog=False)
    return Os
    
##MLM
totalStartTime=time.time()
print('loaded')
samples=1000
import itertools
t=0
for a,b in itertools.product([1,2,3],range(1,10)):
    size=b*10**a
    t+=size*samples
    if(size>3000):
        break
    print (size)
    MLM(size,samples)
print(t/100000.0*3.4,'minuts',t/100000.0*3.4/60,'godzin')
totalDoneTime=time.time()
print('Cały MLM zrobiony w tyle minut: '+str((totalDoneTime-totalStartTime)/60))
print('Cały MLM zrobiony w tyle godzin: '+str((totalDoneTime-totalStartTime)/60/60))
#    
#    
#import time
#time.sleep(60*5)
#import subprocess
#subprocess.call(["shutdown", "/s"])
