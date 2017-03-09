from scipy import integrate
import numpy as np
from scipy.optimize import fsolve,minimize
from baseFunctions import z_max,DNSDistribution_z
import matplotlib.pyplot as plt
from baseFunctions import MonteCarloProb
import time
from tests import randFromDistribution

fname="ABHBH02"
#fname="oneMass"

def loadFromSU(file):
    f=open("data/mass/"+file+".dat")
    masses=[]
    for line1 in f:
        line1=line1.split(' ')
        m=float(line1[5])
        masses.append(m)
    np.savetxt("data/mass/"+file+"_mass.gz",masses)
    np.savetxt("data/mass/"+file+"_mass.txt",masses)
    
#print("loading masses")
#loadFromSU(fname)


#zapisuje w formacie masa, z_max, normalizacja
def preComputeMasses(fname,A,mergerRateFunCof):
    startTime = time.time() 
    mergerRateFunCof=float("%.3f"%(mergerRateFunCof))
    k,current,last = 0,0,0
    masses=np.loadtxt("data/mass/"+fname+"_mass.gz")
    toSave=[]
    for mass in masses:
        
        if(current!=last):
            print(str(int(k/masses.size*100))+"% a"+str(mergerRateFunCof)+' A'+str(A))
            last=current
        current=int(k/masses.size*100)
        k+=1
        zMax=z_max(A,mass)
        if(zMax>15):
            zMax=15
#        print(zMax)
        mergerRateFun=lambda z:(1+z)**(mergerRateFunCof)
        dnsdTylkoOdz=lambda z: DNSDistribution_z(z,mergerRateFun,zMax) #### DNSDistribution_z z zadanym mergerRateFun
        DNSDistribution_z_Norm=integrate.quad(dnsdTylkoOdz,0,zMax)[0] ###uwzglednia rozne zmax dla roznych mas
        probZ=lambda z: dnsdTylkoOdz(z)/DNSDistribution_z_Norm 
#        ProbMax_z=probZ(z_max(A,mass))
        t=minimize(lambda z:-1*probZ(z),0.01,bounds=((0,zMax),))
        ProbMax_z=-1*t.fun
        toSave.append([mass,zMax,ProbMax_z,DNSDistribution_z_Norm])
    
    np.savetxt("data/mass/"+fname+"_mass_A_"+str(A)+"_a"+str(mergerRateFunCof)+".gz",toSave)
    np.savetxt("data/mass/"+fname+"_mass_A_"+str(A)+"_a"+str(mergerRateFunCof)+".txt",toSave)
    doneTime = time.time()
    print('Done in this many minutes: '+str((doneTime-startTime)/60))

#print("preping masses")  
#for a in range(0,4):
#    print(str(a)+' '+str(800))
#    preComputeMasses(fname,800,a)
#    print(str(a)+' '+str(8000))
#    preComputeMasses(fname,8000,a)
#print("done&done")


m=1.160293799999999997e+01
m=5
A=8000
coef=3
zMax=z_max(A,m)
if(zMax>15):
    zMax=15
mergerRateFun=lambda z:(1+z)**(coef)
dnsdTylkoOdz=lambda z: DNSDistribution_z(z,mergerRateFun,zMax) #### DNSDistribution_z z zadanym mergerRateFun
DNSDistribution_z_Norm=integrate.quad(dnsdTylkoOdz,0,zMax)[0] ###uwzglednia rozne zmax dla roznych mas
#print(DNSDistribution_z_Norm)
probZ=lambda z: dnsdTylkoOdz(z)/DNSDistribution_z_Norm 
ProbMax_z=probZ(zMax)

xs=np.linspace(0,1.5*zMax,1000)
ys=[]
for x in xs:
    ys.append(probZ(x))

#plt.xscale('log')
#plt.yscale('log')
plt.plot(xs,ys)
plt.show()


suma=0
ile=1
t=minimize(lambda z:-1*probZ(z),0.01,bounds=((0,zMax),))
print(t.x,-1*t.fun)

ProbMax_z=-1*t.fun
#
startTime = time.time()
for x in range(ile):
#    print(x)
    suma+=MonteCarloProb(probZ,(0,(z_max(A,m))),(0,ProbMax_z))
#    suma+=randFromDistribution(probZ)
    
print(zMax,ProbMax_z,DNSDistribution_z_Norm)
print(1/(z_max(A,m)*ProbMax_z))
print(ile/suma)
doneTime = time.time()
print('Done in this many secs: '+str((doneTime-startTime)/(1)))
