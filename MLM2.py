import matplotlib.pyplot as plt
import numpy as np
#from scipy.interpolate import interp1d
from generateSNRs import generateSample
from histograms import histAndSaveLogLog
#from scipy import integrate
import time
import bisect

A=1600
#size=100000
binsNr=201
def round_down(num, divisor):
    return num - (num%divisor)

def interpolateSNR(modelCoef):
    dataModel=np.loadtxt("SNR/"+str(A)+"/SNRv3_mSU02_a"+str(modelCoef)+"_A"+str(A)+"_sample100000.txt")[:,0]
    binsNr=201
    histData=np.histogram(dataModel,bins=np.logspace(start=np.log10(min(dataModel)),stop=np.log10(max(dataModel)),\
        num=binsNr),normed=True)
    ys=histData[0]
    xs=histData[1]
    for i in range (len(ys)):
        if(ys[i]==0):
            ys[i]=ys[i-1]

    xInterpolatedMax=max(xs)
    xInterpolatedMin=min(xs)
    
    def prob(x):
        size=100000
        b=bisect.bisect_right(xs,x)
#        if(b>=ys.size-1):
#            return 0
        binWidth=xs[b+1]-xs[b]
        toReturn=ys[b]/size/binWidth
        return  toReturn 
        
    f1=np.vectorize(prob)
    return f1,xInterpolatedMin,xInterpolatedMax
     
f0,xInterpolatedMin0,xInterpolatedMax0=interpolateSNR(modelCoef=0)
f1,xInterpolatedMin1,xInterpolatedMax1=interpolateSNR(modelCoef=1)
xInterpolatedMin=max(xInterpolatedMin1,xInterpolatedMin0)
xInterpolatedMax=min(xInterpolatedMax1,xInterpolatedMax0)-15

pointToPlot=6000
xs=np.linspace(xInterpolatedMin,xInterpolatedMax,num=pointToPlot)
ys0=f0(xs)
ys1=f1(xs)
print('minys0 ',min(ys0))
print('minys1 ',min(ys1))
plt.plot(xs,ys0,'g',label='model0')
plt.plot(xs,ys1,'r',label='model1')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.legend()
plt.savefig(str(A)+'.png',dpi=300)
plt.show()
plt.clf()




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
    for sampleIndex in range(int(newSamples/size)):
#        print(sampleIndex)
        O=1
        part=dataSample1[sampleIndex*size :(sampleIndex+1)*size]
        for p in part:
            O*=f1(p)
            O/=f0(p)
        if(O<=0):
            print('O ponizej zera',O)
        Os.append(O)
#        Ps1=f1(slice) 
#        Ps0=f0(slice) 
#        L1=np.prod(Ps1)
#        L0=np.prod(Ps0)
#        print(L0,L1)
#        if(L1/L0<=0):
#            print(L1,L0)
#        Os.append(L1/L0)
    print('done')
    print([i for i in Os if i <=0])
    histAndSaveLogLog(Os,'Bayes Factor',r'$\frac{dN}{dO}$','Bayes Factors A: '+str(A)+' sampleSize: '+str(size)+' samples: '+str(samples),\
    'MLM2/'+str(A)+'/BayesFactors_'+str(A)+'_'+str(size)+'_'+str(samples)+'.png',yLog=False)
    np.savetxt('MLM2/'+str(A)+'/BayesFactors_'+str(A)+'_'+str(size)+'_'+str(samples)+'.txt',Os)
    doneTime = time.time()
    print('In total done in this many minutes: '+str((doneTime-startTime)/60))
    title='Bayes Factors A: '+str(A)+' sampleSize: '+str(size)+' samples: '+str(samples)
    histAndSaveLogLog(Os,'Bayes Factor',r'$\frac{dN}{dO}$',title,\
        'MLM2/'+str(A)+'/BayesFactors_'+str(A)+'_'+str(size)+'_'+str(samples)+'.png',yLog=False)
    return Os
    
###MLM
print('loaded')
samples=1000
for t in [3,6,10,30,60,100]:
    MLM(size=t,samples=samples)
    
import winsound
winsound.PlaySound('sound.wav', winsound.SND_FILENAME)
    
""""
import subprocess
subprocess.call(["shutdown", "/s"])
"""

