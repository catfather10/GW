import numpy as np
import matplotlib.pyplot as plt
from baseFunctions import zMaxGet
import itertools
"""
histogramy mas
"""
sourcefname='ABHBH02'
plt.rc('text', usetex=True)
def massHistIntrinsic():
    data=np.loadtxt('mass/'+sourcefname+'_mass.gz')
    mmax=max(data)
    print('max masa',mmax)
    print(zMaxGet(6400,mmax))
    print(data.size)
    plt.xlabel(r"mass [M_{SUN}]")
    plt.ylabel(r'$\frac{dN}{dM}$')
    plt.title('Intrinsic mass distribution, normalized')
    plt.hist(data,bins=300,normed=True)
    plt.savefig('mass/Intrinsic mass distribution_'+sourcefname+'.png')
    plt.clf()
#    plt.show()
massHistIntrinsic()

def massHist(A,a,mz):
    data=np.loadtxt('SNR/'+str(A)+'/SNRv3_mSU02_a'+str(a)+'_A'+str(A)+'_sample100000.gz')
    if(mz):
        ms=data[:,4]
#        print('min mass',min(ms))
        plt.xlabel(r"mass [M_{SUN}]")
        plt.ylabel(r'$\frac{dN}{dMz}$')
        plt.title('Redshifted mass distribution A: '+str(A)+' model a: '+str(a)+' ,normalized')
        plt.hist(ms,bins=300,normed=True)
        plt.savefig('mass/Redshifted mass distribution A'+str(A)+'model a'+\
                    str(a)+'_'+sourcefname+'.png')
    else:
        ms=data[:,2]
        plt.xlabel(r"mass [M_{SUN}]")
        plt.ylabel(r'$\frac{dN}{dM}$')
        plt.title('Mass distribution A: '+str(A)+' model a: '+str(a)+' ,normalized')
        plt.hist(ms,bins=300,normed=True)
        plt.savefig('mass/Mass distribution A'+str(A)+'model a'+\
                    str(a)+'_'+sourcefname+'.png')
    plt.clf()



#for A,a,mz in itertools.product([800,1600],[0,1], [True,False]):
#    print (A,a,mz)
#    massHist(A,a,mz)