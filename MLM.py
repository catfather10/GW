
import numpy as np
import bisect
import time
import itertools
import gzip
print('########## MLM ##########')

###############
def generateBF(detector,modelName0,modelName1,indices,part,dErrPPerc,dErrMPerc,mErrMPerc,mErrPPerc): #### err in %
    
    ##############
    def histDlM(model , binsnr):
        data=np.loadtxt('SNR/AdLigo/'+model+'AdLigoH1_sample100000.gz')[:,indices]
        data=np.append(data,[[2.4   , 0.0025]],axis=0)
        data=np.append(data,[[4500  , 65    ]],axis=0)

        
        H, edges=np.histogramdd(data,bins=(binsnr,binsnr),normed=True)
        xedges=edges[0][:-1]
        yedges=edges[1][:-1]
        H=H.T
        return H , xedges , yedges
    ##############
    def GaussProb2(HIST,X,Y,m,mErrMinus,mErrPlus,d,dErrMinus,dErrPlus,binsnr=200):
        i=bisect.bisect(Y,m)
        j=bisect.bisect(X,d)

        XX=np.tile(X,(binsnr,1))
        YY=np.tile(Y,(binsnr,1)).T

        SigmaX=np.ones((binsnr,binsnr))
        SigmaX[:,:j-1]=SigmaX[:,:j-1]*dErrMinus
        SigmaX[:,j+1:]=SigmaX[:,j+1:]*dErrPlus
        SigmaX[:,j-1]=SigmaX[:,j-1]*(dErrMinus+dErrPlus)/2

        SigmaY=np.ones((binsnr,binsnr))
        SigmaY[:i-1,:] = SigmaY[:i-1,:]*mErrMinus
        SigmaY[i:,:]   = SigmaY[i:,:]*mErrPlus
        SigmaY[i-1,:]  = SigmaY[i-1,:]*(mErrMinus+mErrPlus)/2

        norm =np.exp(-(YY-m*np.ones((binsnr,binsnr)))**2/(2*SigmaY**2))
        norm*=np.exp(-(XX-d*np.ones((binsnr,binsnr)))**2/(2*SigmaX**2))
        ss=np.sum(norm)
        if(ss==0):
            print("ups norma = 0")
            np.savetxt('normarownazero',[m,mErrMinus,mErrPlus,d,dErrMinus,dErrPlus,binsnr])
        return np.sum(norm*HIST)/ss
    
    ###########
    print('generating BF for',detector,modelName0,modelName1,'part',part,indices)

    names=modelName0+'_'+modelName1

    import os
    if not os.path.exists("MLM/AdLigo/"+names):
        os.makedirs("MLM/AdLigo/"+names)
    
    
    HIST0 , X0 , Y0 = histDlM(modelName0,binsnr) ### X = DL , Y = M
    HIST1 , X1 , Y1 = histDlM(modelName1 , binsnr) ### X = DL , Y = M
    
    f0 = lambda d , m : GaussProb2(HIST0, X0, Y0, m, m*mErrMPerc, m*mErrPPerc, d, d*dErrMPerc, d*dErrPPerc , binsnr)
    f1 = lambda d , m : GaussProb2(HIST1, X1, Y1, m, m*mErrMPerc, m*mErrPPerc, d, d*dErrMPerc, d*dErrPPerc , binsnr)
    
    
    def MLM(size,samples):
        print('size ',size,' samples ',samples)
        startTime = time.time()
        Os=[]
        for sampleIndex in range(samples):
            print("\r",sampleIndex,end='')
            O=1
            for i in range(size):
                text=fMaster.readline()[:-1]
                p=np.fromstring(text,dtype=np.float,sep=' ')
                O*=f1(p[0],p[1])
                O/=f0(p[0],p[1])
            if(O<=0):
                print('O <= 0',O)
            Os.append(O)
        
        np.savetxt('MLM/'+detector+'/'+names+'/BayesFactors_'+detector+'_'+str(size)+'_'+str(samples)+'.txt',Os)
        doneTime = time.time()
        print('In total done in this many minutes: '+str((doneTime-startTime)/60))
        title='Bayes Factors '+detector+' sampleSize: '+str(size)+' samples: '+str(samples)+' '+names
        return Os
        
    ##MLM

    totalStartTime=time.time()    
    index=0
    print('biore sie do czytania')
    fMaster=gzip.open('SNR/'+detector+'/SNR_'+modelName1+'master'+str(part)+str(indices)+'.gz','rt')
    print('przeczytane')
    samples=10000
    for a,b in itertools.product([0,1,2,3],range(1,10)):
        size=b*10**a
        if(size in[1,2]):
            continue
        if(size>=10):
            samples=1000
        if(size>3000):
            break
        print (size)
        MLM(size,samples)
        index+=size*samples
        print('index',index)
    
    totalDoneTime=time.time()
    print('Cały MLM zrobiony w tyle minut: '+str((totalDoneTime-totalStartTime)/60))
    print('Cały MLM zrobiony w tyle godzin: '+str((totalDoneTime-totalStartTime)/60/60))
