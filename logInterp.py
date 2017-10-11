#!/home/mossowski/python-virtualenvs/myenv3.5/bin/python3

import bisect
import numpy as np
"""
Interpoluje piecewise poprzez wyrażenia potęgowe.
"""

class logInterp(object):
    def __init__(self, xs, ys):
        self.xs=np.log10(np.array(xs))
        self.ys=np.log10(np.array(ys))
        self.xrange=(min(self.xs),max(self.xs))
        self.aarray=[]
        self.barray=[]
        self.funs=[]
        if (len(xs)!=len(ys)):
            print('rozna ilosc danych')
            return -1
        if (not np.array_equal(self.xs,np.sort(self.xs))):
            print('x nie posortowne')
            return -1
        for i in range(len(xs)-1):
            a=(self.ys[i]-self.ys[i+1])/(self.xs[i]-self.xs[i+1])
            b=self.ys[i]-a*self.xs[i]
            self.aarray.append(a)
            self.barray.append(b)
            self.funs.append(lambda x: 10**(a*self.xs[i]+b))
#        print('a',self.aarray)
#        print('b',self.barray)
        
    def __call__ (self,x):
        x=np.log10(x)
        if(x>=self.xrange[1] or x<self.xrange[0]):
            if(x==self.xrange[1]):
                return self.ys[len(self.ys)-1]
            print('z poza zakresu interpolacji')
            return -1
        i=bisect.bisect_right(self.xs,x)-1
        return 10**(self.aarray[i]*x+self.barray[i])
        
#import matplotlib.pyplot as plt
#
#Ms=np.logspace(0,np.log10(400-1),num=1000)
#xs=[1,60,100,200,400]
#ys=[400,10000,15000,20000,10000]
#plt.plot(xs,ys,'o')
#from scipy.interpolate import interp1d
#f=interp1d(xs,ys)
#plt.plot(Ms,f(Ms))
#AdLigo=logInterp(xs,ys)
#plt.plot(Ms,Ms**(5/6)*400)
###
#s=[]
#for m in Ms:
#    s.append(AdLigo(m))
##    
#plt.plot(Ms,s)
#plt.loglog()
#plt.grid()
    
