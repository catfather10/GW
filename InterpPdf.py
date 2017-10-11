#!/home/mossowski/python-virtualenvs/myenv3.5/bin/python3

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:43:10 2017

@author: Maciek
"""
import bisect
import numpy as np


class InterpPdf(object):
    def __init__(self, xs, ys):
        self.xs=np.array(xs)
        self.ys=np.array(ys)
        self.xrange=(min(xs),max(xs))
        self.aarray=[]
        self.barray=[]
#        print('xs',self.xs)
#        print('ys',self.ys)
        self.partialIntegrals=[]

        if (len(xs)!=len(ys)):
            print('rozna ilosc danych')
            return -1
        if (not np.array_equal(self.xs,np.sort(self.xs))):
            print('x nie posortowne')
            return -1

        for i in range(len(xs)-1):
            a=(ys[i]-ys[i+1])/(xs[i]-xs[i+1])
            b=ys[i]-a*xs[i]
            self.aarray.append(a)
            self.barray.append(b)
            integ=(a/2*xs[i+1]**2+b*xs[i+1])-(a/2*xs[i]**2+b*xs[i])
            self.partialIntegrals.append(integ)
        self.norm=sum(self.partialIntegrals)
            
    def __call__ (self,x):
        if(x>=self.xrange[1] or x<self.xrange[0]):
            if(x==self.xrange[1]):
                return self.ys[len(self.ys)-1]
            print('z poza zakresu interpolacji')
            return -1
        i=bisect.bisect_right(self.xs,x)-1
        return (self.aarray[i]*x+self.barray[i])
        
    def pdf(self,x):
        return self.__call__(x)/self.norm
    
    def cdf(self,x):
        if(x>=self.xrange[1] or x<self.xrange[0]):
            if(x==self.xrange[1]):
                return 1
            print('z poza zakresu interpolacji')
            return -1
        i=bisect.bisect_right(self.xs,x)-1
        integral=0
        for c in range(i):
            integral+=self.partialIntegrals[c]
        integral+=(self.aarray[i]/2*x**2+self.barray[i]*x)\
                    -(self.aarray[i]/2*self.xs[i]**2+self.barray[i]*self.xs[i])
        return integral/self.norm
        

            
            
    
