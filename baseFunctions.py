from scipy import integrate
import numpy as np
from numpy.random import random as rng
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numba import jit
from scipy.optimize import minimize
plt.rc('text', usetex=True)

####stałe
H_0=70.4 # km/s  Mpc^-1
Omega_m=0.27 
Omega_lambda=1-Omega_m
c=299792 # km/s
D_H=c/H_0

@jit(nopython=True) 
def E(z):
    return np.sqrt(Omega_m*(1+z)**3+(1-Omega_m))
    
def D_c(z):
    z2=lambda z: 1/E(z)
    return D_H*integrate.quad(z2,0,z)[0] 

@jit
def D_L(z):
    return (1+z)*D_c(z)

def zMaxGet(A,m):
    toSolve = lambda z:A*(m*(1+z))**(5/6)/D_L(z)-2
    return fsolve(toSolve,.1)[0]
        
def DNSDistribution(z,mergerRateFun,zMax): ### dN/dz
    if(z<=zMax):
        return mergerRateFun(z)*4*np.pi*D_H*(D_c(z))**2/(E(z)*(1+z))
    else:
        return 0
                
@jit(nopython=True)
def SNR(Dl,theta,phi,psi,iota,mz,A):
    FP=0.5*(1+(np.cos(theta))**2)*np.cos(2*phi)*np.cos(2*psi)-np.cos(theta)*np.sin(2*phi)*np.sin(2*psi)
    FX=0.5*(1+(np.cos(theta))**2)*np.cos(2*phi)*np.sin(2*psi)+np.cos(theta)*np.sin(2*phi)*np.cos(2*psi)
    TH=2*np.sqrt(((FP*(1+(np.cos(iota))**2))**2) +4*(FX*np.cos(iota))**2) 
    return A*TH*mz**(5/6)/Dl
   
def MonteCarloSampling(probFun,xRange,yRange):
    k=0
    while (True):
        k+=1
        x=rng()*(xRange[1]-xRange[0])+xRange[0]
        y=rng()*(yRange[1]-yRange[0])+yRange[0]
        if(probFun(x)>y):
            return x
            
'''
def InverseTransformSampling(probFun):
    u=rng()
    toSolve=lambda x: integrate.quad(probFun,0,x)[0] -u
    return fsolve(toSolve,0)[0]
'''
"""
def probFunTest(x):
    if (x>=0 and x <= 4):
        return (x**3)/64
    else:
        return 0
print(integrate.quad(probFunTest,0,4))
        
xs = np.linspace(0,5,num=100)
probFunTestVect=np.vectorize(probFunTest)
ys=probFunTestVect(xs)
plt.plot(xs,ys)
plt.show()

print(InverseTransformSampling(probFunTest))        
        
#data=[]
#for t in range(0,1):
#    data.append(InverseTransformSampling(probFunTest))
#plt.hist(data,bins=100)
#plt.show()
"""
            

####szybki wykres DNSDistribution_z
#coef=0
#A=6400
#mz=13.054006
#zm=zMaxGet(A,mz)
#g2=lambda x: DNSDistribution(x,lambda z:(1+z)**coef,zm)
#print('najierw TU: ',g2(.1))
#norm=integrate.quad(g2,0,zm)[0]
#print('zmax: ',zm)
#xs=np.linspace(0,zm,num=1000)
#ys=[]
#for x in xs:
#    ys.append(DNSDistribution(x,lambda z:(1+z)**coef,zm)/norm)
#ys=np.array(ys)
##print(ys)
#
#
#
#print('norm',norm)
#t=minimize(lambda z:-1*g2(z)/norm,0.01,bounds=((0,zm),))
#print('probMax: ',-1*t.fun[0])
##ys=ys/norm
#print('trapz: ',integrate.trapz(ys,xs))
#plt.plot(xs,ys)
#plt.show()
#data=np.loadtxt('data/mass/ABHBH02_mass_A_6400_a0.0.gz')[:,0]
#print(max(data))
#
#            
#
