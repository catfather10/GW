import matplotlib.pyplot as plt
import numpy as np
#from scipy.interpolate import interp1d
from generateSNRs import generateSample
from histograms import histAndSaveLogLog
from scipy import integrate
import bisect
import time
A=1600
size=100000
binsNr=201
def round_down(num, divisor):
    return num - (num%divisor)

###model 0

###model 1

def interpolateSNR(modelCoef):
    dataModel=np.loadtxt("SNR/"+str(A)+"/SNRv3_mSU02_a"+str(modelCoef)+"_A"+str(A)+"_sample100000.txt")[:,0]
    binsNr=201
    histData=np.histogram(dataModel,bins=np.logspace(start=np.log10(min(dataModel)),stop=np.log10(max(dataModel)),\
        num=binsNr),normed=False)
    ysInt=histData[0]
    xsInt=histData[1]
    
#    print(integrate.simps(ysInt,xsInt[:-1]))
    for i in range (len(ysInt)):
        if(ysInt[i]==0):
            ysInt[i]=ysInt[i-1]

    xInterpolatedMax=max(xsInt)
    xInterpolatedMin=min(xsInt)
    
    def prob(x):
#        print(xsInt[5])
        size=100000
        b=bisect.bisect_right(xsInt,x)
        if(b>=ysInt.size):
#            return 0
            b=ysInt.size-1
        binWidth=xsInt[b+1]-xsInt[b]
#        binWidth=1
        toReturn=ysInt[b]/size/binWidth
        return  toReturn 
        
    f1=np.vectorize(prob)
#    print(integrate.simps(f1(xsInt[:-1]),xsInt[:-1]))
    return f1,xInterpolatedMin,xInterpolatedMax
    
f0,xIntMin0,xIntMax0=interpolateSNR(0)
f1,xIntMin1,xIntMax1=interpolateSNR(1)

xs0=np.linspace(xIntMin0,xIntMax0,num=5000)
ys0=f0(xs0)
#plt.loglog()
#plt.plot(xs0,ys0)
#print(integrate.simps(ys0,xs0))

xs1=np.linspace(xIntMin1,xIntMax1,num=5000)
ys1=f1(xs1)
#plt.loglog()
#plt.plot(xs1,ys1)
#print(integrate.simps(ys1,xs1))


xIntMax=min(xIntMax0,xIntMax1)
xIntMin=max(xIntMin0,xIntMin1)
xs=np.linspace(xIntMin,xIntMax,num=5000)
ys00=f0(xs)
ys01=f1(xs)

norm00=integrate.simps(ys00,xs)
norm01=integrate.simps(ys01,xs)
#print(norm00)
#print(norm01)


#plt.plot(xs,ys00,'r')
#plt.plot(xs,ys01,'c')

plt.show()
f0n=lambda x:f0(x)/norm00
f1n=lambda x:f1(x)/norm01

ys00=f0n(xs)
ys01=f1n(xs)
#print(min(ys00),min(ys01))
print('po normalizacji')
print(integrate.simps(ys00,xs))
print(integrate.simps(ys01,xs))

plt.rc('text', usetex=True)
plt.title('Probability density function for detector A='+str(A))
plt.plot(xs,ys00,'r',label=r'model $\alpha$ =0')
plt.plot(xs,ys01,'c',label=r'model $\alpha$ =1')

plt.loglog()
plt.legend()
plt.savefig(str(A),dpi=300)
plt.clf()
plt.rc('text', usetex=False)





def MLM(size,samples):
    print('size ',size,' samples ',samples)
    startTime = time.time()
    Os=[]
    dataSample1=generateSample("test",size*samples,A=A,mergerRateFunCoef=1,saveFlag=False)[:,0]
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
    histAndSaveLogLog(Os,'Bayes Factor',r'$\frac{dN}{dO}$','Bayes Factors A: '+str(A)+' sampleSize: '+str(size)+' samples: '+str(samples),\
    'MLM/'+str(A)+'/BayesFactors_'+str(A)+'_'+str(size)+'_'+str(samples)+'.png',yLog=False)
    np.savetxt('MLM/'+str(A)+'/BayesFactors_'+str(A)+'_'+str(size)+'_'+str(samples)+'.txt',Os)
    doneTime = time.time()
    print('In total done in this many minutes: '+str((doneTime-startTime)/60))
    title='Bayes Factors A: '+str(A)+' sampleSize: '+str(size)+' samples: '+str(samples)
    histAndSaveLogLog(Os,'Bayes Factor',r'$\frac{dN}{dO}$',title,\
        'MLM/'+str(A)+'/BayesFactors_'+str(A)+'_'+str(size)+'_'+str(samples)+'.png',yLog=False)
    return Os
    
###MLM
#print('loaded')
#samples=10000
#for t in [3,6,10,30,60,100]:
#    MLM(size=t,samples=samples)
#    
#import winsound
#winsound.PlaySound('sound.wav', winsound.SND_FILENAME)
#    
#import time
#time.sleep(60*5)
#import subprocess
#subprocess.call(["shutdown", "/s"])
