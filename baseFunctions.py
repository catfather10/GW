from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random as rng
from scipy.optimize import fsolve,brentq
from numba import jit

from InterpPdf import InterpPdf

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
            print('wylosowano po',k)
            return x
            
@jit(nopython=True)           
def SFR(z):
    return (1+z**2.7)/(1+((1+z)/2.9)**5.6)
    
def inverseSampling(intpdf):
    u=rng()
    toSolve = lambda x: intpdf.cdf(x)-u
    return brentq(toSolve,a=intpdf.xrange[0],b=intpdf.xrange[1],disp=True)

def rejectionEnvelopeSampling(probFun,intpdf):
    k=0
    while (True):
        k+=1
        u=rng()
        x=inverseSampling(intpdf)
        if(u<probFun(x)/(intpdf.pdf(x)*intpdf.norm)):
#            print('wylosowano po',k)
            return x  

"""
nowe zabawki
"""

A=16000
mmax=max(np.loadtxt('mass/ABHBH02_mass.gz'))
zm=zMaxGet(A,mmax)
xs=np.logspace(np.log10(0.001),np.log10(zm),num=1000)
print('zMax',zm)

print('######### SFR #########')
xspointsSFR=[0,5,6.5,20,zm+1]
yspointsSFR=[.35,.35,.03,.0004,.0001]
plt.plot(xspointsSFR,yspointsSFR,'o')
fintSFR=InterpPdf(xspointsSFR,yspointsSFR)
ysInterpSFR=[]
for x in xs:
    ysInterpSFR.append(fintSFR(x))
plt.plot(xs,ysInterpSFR,label='interp')
vDNSDistribution=np.vectorize(DNSDistribution)
print('normaSFR',integrate.quad(lambda z:DNSDistribution(z,SFR,zm),0,zm)[0])
normSFR=integrate.quad(lambda z:DNSDistribution(z,SFR,zm),0,zm)[0]
fSFR=lambda x:vDNSDistribution(x,SFR,zm)/normSFR

flag=False
for i in xs:
    if(fintSFR(i)<fSFR(i)):
        print('tu sie psuje',i)
        flag=True
        
        break
print('flaga',flag)
plt.plot(xs,fSFR(xs),label='SFR')
print(integrate.simps(fSFR(xs),xs))
plt.legend(loc=0)
plt.clf()

#print('######### a0 #########')
#fModelProb=lambda x: 1
#xspointsModel=[0,5,8,20,55,zm+1]
#yspointsModel=[.12,.12,.03,.008,.001,.0001]
#plt.plot(xspointsModel,yspointsModel,'o')
#fintModel=InterpPdf(xspointsModel,yspointsModel)
#ysInterpModel=[]
#for x in xs:
#    ysInterpModel.append(fintModel(x))
#plt.plot(xs,ysInterpModel,label='interp')
#vDNSDistribution=np.vectorize(DNSDistribution)
#print('normaModel',integrate.quad(lambda z:DNSDistribution(z,fModelProb,zm),0,zm)[0])
#normModel=integrate.quad(lambda z:DNSDistribution(z,SFR,zm),0,zm)[0]
#fModel=lambda x:vDNSDistribution(x,fModelProb,zm)/normModel
#
#flag=False
#for i in xs:
#    if(fintModel(i)<fModel(i)):
#        print('tu sie psuje',i)
#        flag=True
#        
#        break
#print('flaga',flag)
#plt.plot(xs,fModel(xs),label='Model')
#print(integrate.simps(fModel(xs),xs))
#plt.legend(loc=0)
#plt.clf()


print('########## koniec baseFun ##########')
