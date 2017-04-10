#import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from generateSNRs import generateSample
from histograms import histAndSaveLogLog
import time

def round_down(num, divisor):
    return num - (num%divisor)
    
AtoSim=6400
def interpolate(model):
    dataModel1=np.loadtxt("data/v3/SNRv3_mSU02_a1_A"+str(AtoSim)+"_sample100000.gz")[:,0]
    histData=np.histogram(dataModel1,bins=600)
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
    ys=ys2/np.sum(ys2)
    xs=xs2
    #print(len(xs),len(ys))
    #print(min(ys))
    xInterpolatedMax=max(xs)
    xInterpolatedMin=min(xs)
    f1=interp1d(xs,ys,kind='linear')
    return f1,(xInterpolatedMin,xInterpolatedMax)
    

#xs1=np.linspace(min(xs),max(xs),num=6000)
#ys1=f(xs1)
#print(integrate.simps(ys,xs))
#print(integrate.simps(ys1,xs1))
#print(np.sum(ys))
#plt.gca().set_xscale("log")
#plt.gca().set_yscale("log")
#plt.plot(xs,ys,'.')
#plt.plot(xs1,ys1)
#plt.show()

def MLM(size,samples):
    print('size ',size,' samples ',samples)
    startTime = time.time()
    Os=[]
    dataSample1=generateSample("test",size*samples,A=AtoSim,mergerRateFunCoef=1,saveFlag=False)[:,0]
    dataSample0=generateSample("test",size*samples,A=AtoSim,mergerRateFunCoef=0,saveFlag=False)[:,0]
    print('a1',[i for i in dataSample1 if i <=0])
    print('a0',[i for i in dataSample0 if i <=0])
    #usuwanie SNRow z poza zakresu interpolacji
    dataSample1=[i for i in dataSample1 if (i <= xInterpolatedMax and i>=xInterpolatedMin)]
    dataSample0=[i for i in dataSample0 if (i <= xInterpolatedMax and i>=xInterpolatedMin)]
    newSize=min([len(dataSample1),len(dataSample0)])
    dataSample1=dataSample1[0:round_down(newSize,size)]
    dataSample0=dataSample0[0:round_down(newSize,size)]


    print('newsize',newSize)
    print('b1',[i for i in dataSample1 if i <=0])
    print('b0',[i for i in dataSample0 if i <=0])
    for t in range(int(newSize/size)):
#        print(t)
        slice1=dataSample1[t*size :(t+1)*size]
        slice0=dataSample0[t*size :(t+1)*size]
        Ps1=f(slice1) 
        Ps0=f(slice0) 
        L1=np.prod(Ps1)
        L0=np.prod(Ps0)
#        print(L0,L1)
        if(L1/L0<=0):
            print(L1,L0)
        Os.append(L1/L0)
    print('done')
    print([i for i in Os if i <=0])
#    print(Os)
    histAndSaveLogLog(Os,'Bayes Factor',r'$\frac{dN}{dO}$','Bayes Factors A: '+str(AtoSim)+' sampleSize: '+str(size)+' samples: '+str(samples),\
    'MLM/'+str(AtoSim)+'/BayesFactors_'+str(AtoSim)+'_'+str(size)+'_'+str(samples)+'.png',yLog=False)
    np.savetxt('MLM/'+str(AtoSim)+'/BayesFactors_'+str(AtoSim)+'_'+str(size)+'_'+str(samples)+'.txt',Os)
    doneTime = time.time()
    print('In total done in this many minutes: '+str((doneTime-startTime)/60))
    return Os
def MLM_stats(sampleSize):
    s=size
    text_file = open('MLM/'+str(AtoSim)+'/BayesFactors_stats_'+str(AtoSim)+'_'+str(s)+'_10000.txt', "w")
    data=np.loadtxt('MLM/'+str(AtoSim)+'/BayesFactors_'+str(AtoSim)+'_'+str(s)+'_10000.txt')
    moreThanOne=np.array([i for i in data if i >=1])
    print('median: \t\t',np.median(data),file=text_file)
    print('log median: \t',np.log10(np.median(data)),file=text_file)
    print('mean log:\t\t',np.mean(np.log(data)),file=text_file)
    print('% of >1 :\t\t',moreThanOne.size/data.size*100    ,file=text_file)
    print('variance :\t',np.var(data),file=text_file)
    print('var/mean :\t',np.var(data)/np.mean(data),file=text_file)
    text_file.close()
def MLMPlots(): 
    import matplotlib.pyplot as plt
    plt.clf()
    for s in [3,6,10,30,60,100]:
        data=np.loadtxt('MLM/'+str(AtoSim)+'/BayesFactors_'+str(AtoSim)+'_'+str(s)+'_10000.txt')
        title='Bayes Factors A: '+str(AtoSim)+' sampleSize: '+str(size)+' samples: '+str(s)
        histAndSaveLogLog(data,'Bayes Factor',r'$\frac{dN}{dO}$',title,\
        'MLM/'+str(AtoSim)+'/BayesFactors_'+str(AtoSim)+'_'+str(size)+'_'+str(s)+'.png',yLog=False)


print('loaded')
size=10000
for t in [3,6,10,30,60,100]:
    MLM(size=t,samples=size)
for s in [3,6,10,30,60,100]:
    MLM_stats(s)

