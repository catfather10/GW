from scipy import integrate
import numpy as np
from scipy.optimize import fsolve
from baseFunctions import z_max,DNSDistribution_z



fname="ABHBH02"

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
        
        mergerRateFun=lambda z:(1+z)**(mergerRateFunCof)
        dnsdTylkoOdz=lambda z: DNSDistribution_z(z,mergerRateFun) #### DNSDistribution_z z zadanym mergerRateFun
        DNSDistribution_z_Norm=integrate.quad(dnsdTylkoOdz,0,z_max(A,mass))[0] ###uwzglednia rozne zmax dla roznych mas
        probZ=lambda z: dnsdTylkoOdz(z)/DNSDistribution_z_Norm 
        ProbMax_z=probZ(z_max(A,mass))
        toSave.append([mass,z_max(A,mass),ProbMax_z])
    
#    np.savetxt("data/mass/"+fname+"_mass_a"+str(mergerRateFunCof)+".gz",masses)
    np.savetxt("data/mass/"+fname+"_mass_A_"+str(A)+"_a"+str(mergerRateFunCof)+".txt",toSave)

print("preping masses")  
for a in range(0,4):
    print(str(a)+' '+str(800))
    preComputeMasses(fname,800,a)
    print(str(a)+' '+str(8000))
    preComputeMasses(fname,8000,a)
print("done&done")

#t=z_max(1000,)
#t1=z_max(1000,1)
#print(t)
#print(t1)
