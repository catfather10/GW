import matplotlib.pyplot as plt
import numpy as np
from v3_masses import generateSample
from scipy import stats
import time


AtoSim=6400
n=100000 
dataModel0=np.loadtxt("data/v3/SNRv3_mSU02_a0_A"+str(AtoSim)+"_sample100000.gz")[:,0]
ile=1000
sample=3
pvalues=[]
Ds=[]
startTime = time.time()

for t in range(ile):
    print(t)
    dataModel1=generateSample("test",sample,A=AtoSim,mergerRateFunCoef=1,saveFlag=False)[:,0]
    t=stats.ks_2samp(dataModel1,dataModel0)
    Ds.append(t[0])
    pvalues.append(t[1])
pvalues,Ds=np.array(pvalues),np.array(Ds)
name='KStest/'+str(AtoSim)+'/pvalues_'+str(sample)
plt.hist(pvalues,bins=100)
plt.savefig(name+'.png')
np.savetxt(name+'.txt',pvalues)
np.savetxt('KStest/'+str(AtoSim)+'/Ds_'+str(sample)+'.txt',pvalues)
pe=[]
pe.append(np.percentile(pvalues,50))
#pe=np.array(pe)
np.savetxt('KStest/median_'+str(sample)+'.txt',pe)
print(pe)
doneTime = time.time()
print('In total done in this many minutes: '+str((doneTime-startTime)/60))
med=np.median(pvalues)
print('mediana:' , med)

#sampleSize=5000
#pvalues=np.loadtxt('KStest/pvalues_'+str(sampleSize)+'.txt')
##print(pvalues)
#med=np.median(pvalues)
#print('mediana:' , med)