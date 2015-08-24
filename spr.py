def iseven(n):
    return n%2==0

def pkfind(x,y,n):
	from matplotlib.pyplot import ginput,plot,close,xlabel,ylabel,grid
	from numpy import zeros, array
	close('all')
	plot(x,y)
	grid(True)
	rng=ginput(2*n)
	close()
	xmax=zeros(n)
	ymax=zeros(n)
	for i in range(n):
		x1=rng[2*i][0]
		x2=rng[2*i+1][0]
		if x1<x2:
			xrng=(x>=x1)&(x<=x2)
		elif x2<x1:
			xrng=(x>=x2)&(x<=x1)
		xmax[i]=x[xrng][y[xrng].argmax()]
		ymax[i]=y[xrng].max()
	indmax=(xmax-x[0])/abs(abs(x[1])-abs(x[0]))
	return xmax,ymax
			
def cutwvfrm(t,x,zerooffset=False,window=('tukeywin',0.5)):
	import numpy as np
	from matplotlib.pyplot import ginput, plot, close
	plot(t,x)
	tx=ginput(2)
	close()
	tx1=tx[0]
	tx2=tx[1]
	if (tx1[0]<tx2[0]):
		t1=tx1[0]
		t2=tx2[0]
	elif (tx1[0]>tx2[0]):
		t1=tx2[0]
		t2=tx1[0]
	ind=(t>=t1)&(t<=t2)
	X=x[ind]
	if window[0]=='tukeywin':
		X=X*tukeywin(X.size,window[1])
	elif window[0]=='gausswin':
		X=X*gausswin(X.size,window[1])
	elif window[0]=='expwin':
		X=X*expwin(X.size,window[1])
	if zerooffset==False:
		T=t[ind]
	elif zerooffset==True:
		dt=abs(abs(t[1])-abs(t[0]))
		T=np.linspace(0.0,dt*X.size,X.size)
	return T,X,ind
	
def freqs(N,dt):
	from numpy import linspace
	f=linspace(0.,1/(2*dt),N/2+1)
	return f

def loaddata(pth):
	import numpy as np
	import cPickle as cp
	fltype=pth[-3::]
	if (fltype=='csv')|(fltype=='isf'):
		x=np.loadtxt(pth,delimiter=',',unpack=True)
	elif fltype=='dat':
		x=cp.load(pth,'rb')
	elif fltype=='npy':
		x=np.load(pth)
	else:
		x=np.loadtxt(pth,unpack=True)
	return x

def savedata(pth,x):
	import numpy as np
	import os
	import scipy.io as sio
	if type(pth)==tuple:
		fltype=pth[-1][-3::]
	else:
		fltype=pth[-3::]
	if (fltype=='csv')|(fltype=='isf'):
		np.savetxt(pth,x,delimiter=',')
	elif (fltype=='mat'):
		# for this kind of save, pth must be a tuple (pthtofile,filname) and x must be a dict {'varname':var}
		os.chdir(pth[0])
		sio.savemat(pth[1],x)
	elif (fltype=='npy'):
		np.save(pth,x)
	else:
		np.savetxt(pth,x)
		
def asignal(x):
	import numpy.fft as nf
	import numpy as np
	X=2*nf.fft(x)
	X[np.ceil(X.size/2)::]=0
	xa=nf.ifft(X)
	return(xa)
			
		
def tukeywin(N,alpha):
	import numpy as np
	n=np.linspace(0,alpha*N/2,alpha*N/2)
	w1=np.sin(np.pi/(alpha*N)*n)
	w2=np.ones(N-2*n.size)
	w3=np.cos(np.pi/(alpha*N)*n)
	w=np.hstack((w1,w2,w3))
	return w
	
def gausswin(N,stddev):
	import numpy as np
	n=np.linspace(0,N,N)
	w=np.exp(-((n-np.floor(N/2))**2)/(2*stddev**2))
	return w
	
def expwin(N,decay):
	import numpy as np
	n=np.linspace(0,N,N)
	w=np.exp(-decay*n)
	return w
	

def corr(x1,x2,dt,off=0.):
	import numpy as np
	Noff=int(off/dt)
	x2=np.hstack((np.zeros(Noff),x2))
	x1=np.hstack((x1,np.zeros(x2.size-x1.size)))
	x12=np.correlate(x1,x2,'full')
	t=np.linspace(-x12.size*dt/2,x12.size*dt/2,x12.size)	
	return t,x12
	

def beamfreqs(E,h,rho,l,bc):
	import numpy as np
	if bc=='cf':
		B=np.array([1.875104,4.694091,7.854757,10.995541])
	elif bc=='ff':
		B=np.array([4.730041,7.853205,10.995608,14.137165])
	s=(B**2/(4*np.pi))*h*np.sqrt(E/(3*rho*l**4))
	return s

def barfreqs(n,E,rho,l):
	import numpy as np
	s=(n*np.sqrt(E/rho))/(2*l)
	return s

def ftrans(x,dt,frange=None):
	from numpy.fft import rfft
	f=freqs(x.size,dt)
	X=rfft(x)
	if type(frange)==tuple:
		X=X[(f>=frange[0])&(f<=frange[1])]
		f=f[(f>=frange[0])&(f<=frange[1])]
	return f,X

def pltft(x,dt,pltype='mag',frange=None):
	import numpy as np
	import matplotlib.pyplot as plt
	f,X=ftrans(x,dt)
	if pltype=='mag':
		plt.plot(f,np.abs(X/len(X)))
		plt.xlabel(r'$f$')
		plt.ylabel(r'$|X (\omega)|$')
		if type(frange)==tuple:
			plt.xlim(frange)
		plt.show()
	elif pltype=='phase':
		plt.plot(f,np.degrees(np.unwrap(np.arctan(np.imag(X)/np.real(X)))))
		plt.xlabel(r'$f$')
		plt.ylabel(r'$\angle X (\omega)$')
		if type(frange)==tuple:
			plt.xlim(frange)
		plt.show()	
		
	
def hpfilter(x,dt,fc,fp):
	import numpy as np
	import numpy.fft as nf
	X=nf.rfft(x)
	N=X.size
	df=1/(dt*x.size)
	Np=int((N*df-fp)/df)
	Ntr=int((fp-fc)/df)
	Nzp=N-Np-Ntr
	ntr=np.linspace(0,Ntr,Ntr)
	Hp=np.ones(Np)
	Htr=np.sin(np.pi*ntr/(2*Ntr))
	Hzp=np.zeros(Nzp)
	xf=nf.irfft(X*np.hstack((Hzp,Htr,Hp)))	
	return xf
	
def lpfilter(x,dt,fp,fc):
	import numpy as np
	import numpy.fft as nf
	X=nf.rfft(x)
	N=X.size
	df=1/(dt*x.size)
	Np=int(fp/df)
	Ntr=int((fc-fp)/df)
	Nzp=N-Np-Ntr
	ntr=np.linspace(0,Ntr,Ntr)
	Hp=np.ones(Np)
	Htr=np.cos(np.pi*(ntr/(2*Ntr)))
	Hzp=np.zeros(Nzp)
	xf=nf.irfft(X*np.hstack((Hp,Htr,Hzp)))
	return xf
	
def bpfilter(x,dt,fc1,fp1,fp2,fc2):
	import numpy as np
	import numpy.fft as nf
	from matplotlib.pyplot import plot
	X=nf.rfft(x)
	N=X.size
	f=freqs(N,dt)
	df=1/(dt*x.size)
	Np=int((fp2-fp1)/df)
	Ntr1=int((fp1-fc1)/df)
	Ntr2=int((fc2-fp2)/df)
	Nzp1=int(fc1/df)
	Nzp2=N-Nzp1-Ntr1-Np-Ntr2
	ntr1=np.linspace(0,Ntr1,Ntr1)
	ntr2=np.linspace(0,Ntr2,Ntr2)
	Hp=np.ones(Np)
	Htr1=np.sin(np.pi*(ntr1/(2*Ntr1)))
	Htr2=np.cos(np.pi*(ntr2/(2*Ntr2)))
	Hzp1=np.zeros(Nzp1)
	Hzp2=np.zeros(Nzp2)
	xf=nf.irfft(X*np.hstack((Hzp1,Htr1,Hp,Htr2,Hzp2)))
	return xf
	
def resfreqs(x,dt,n,frange=None):
	from numpy import abs
	f,X=ftrans(x,dt,frange)
	fres,Xres=pkfind(f,abs(X),n)
	return fres

def dispersion(x,t,d,fc1,fp1,fp2,fc2,alpha=1):
	from matplotlib.pyplot import ginput, plot, close, grid
	from numpy.fft import rfft,fft
	from numpy import finfo, zeros, hstack, unwrap, double, pi, angle, arctan2, imag, real
	close('all')
	eps=finfo(double).tiny
	fc1=1e6*fc1
	fp1=1e6*fp1
	fp2=1e6*fp2
	fc2=1e6*fc2
	dt=abs(t[-1]-t[-2])
	x=x-x[0]
	plot(t,x)
	grid(True)
	tx=ginput(4)
	close()
	x1=x[(t>=tx[0][0])&(t<=tx[1][0])]
	x2=x[(t>=tx[2][0])&(t<=tx[3][0])]
	toff=t[(t>=tx[1][0])&(t<=tx[2][0])]
	N1=len(x1)
	N2=len(x2)
	N3=len(toff)
	x1=hstack((x1*tukeywin(N1,alpha),zeros(N2+N3)))
	x2=hstack((zeros(N1+N3),x2*tukeywin(N2,alpha)))
	x1=bpfilter(hstack((x1*tukeywin(N1,alpha),zeros(N2+N3))),dt,fc1,fp1,fp2,fc2)
	x2=bpfilter(hstack((zeros(N1+N3),x2*tukeywin(N2,alpha))),dt,fc1,fp1,fp2,fc2)
	H=-rfft(x2)/rfft(x1)
	f=freqs(len(x1),dt)
	phi=unwrap(angle(H));
	phi=phi[(f>=fp1)&(f<=fp2)]
	f=f[(f>=fp1)&(f<=fp2)]
	c=-(4.*d*1e-3*pi*f)/phi
	f=1e-6*f[(f>=fp1)&(f<=fp2)]
	return f,c,phi
	
def moments(x,y):
    
    from numpy import round,average,array
    from numpy.linalg import norm
    
    y=y/norm(y)
        
    xm=[]
    
    xm.append(average(x,weights=y))
    xm.append(average((x-xm[0])**2,weights=y))
    xm.append(average((x-xm[0])**3,weights=y)/xm[1]**(1.5))
    xm.append(average((x-xm[0])**4,weights=y)/xm[1]**2)
 
    return xm
    
def localmax(x):
    
    from numpy import diff,sign
    
    indmax=(diff(sign(diff(x))) < 0).nonzero()[0] + 1 
    
    return indmax
    
    
def localmin(x):
    
    from numpy import diff,sign
    
    indmin=(diff(sign(diff(x))) > 0).nonzero()[0] + 1 # local min

    
    return indmin
    
def PeakLimits(x,indmax,db):
    
    from numpy import log10
    
    indleft=indmax
    indright=indmax
    
    xx=x.copy()

    xx=xx/xx[indmax]    
    
    DB=0.

    while DB>db:
    
        indleft -=1
    
        DB=20*log10(xx[indleft])

    
    DB=0.
    
    while DB>db:
    
        indright +=1
    
        DB=20*log10(xx[indright])
    
    return indleft, indright
        
    
def SpacedArgmax(x,N,space):
    
    i=[]    
    xx=x.copy()
    
    for n in range(N):

        I=xx.argmax()
        
        xx[I-space/2:I+space/2]=0.
        
        i.append(I)

    i.sort()
    
    return i
   
#def H1Pairs(x,dt,frng,pairs):
#    from numpy.fft import rfft
#    from numpy import linspace,angle,conj
#

    
def H1(x,y,dt,frng,uwphase=True,nfreqs='All'):
    """
    Returns a frequency vector, transfer function between x and y, and the wrapped phase of the transfer
    function
    """    
    
    from numpy import zeros,conjugate,linspace, angle, unwrap, array, arange
    from numpy.fft import rfft    
    
    # X = rfft(x,int(1/(df*dt)))
   #  Y = rfft(y,int(1/(df*dt)))
    X=rfft(x)
    
    Xc=conjugate(X)
    
    H=[]
    phi=[]
    f=linspace(0.,1/(2*dt),len(X))
    
    
    df=f[1]-f[0]
    if1=NearestValue(f,frng[0])
    if2=NearestValue(f,frng[1])
    
    if nfreqs is 'All':
        
        find=arange(if1,if2,1)
        
    else:
        
        find=linspace(if1,if2,nfreqs).astype(int)
    

    for i in range(len(y)):
        
        Y=rfft(y[i])
        HH=(Y*Xc)/(X*Xc)
        
        pphi=angle(HH)
        
        if uwphase:
            pphi=unwrap(angle(HH))
            
            
        H.append(HH[find])
        phi.append(pphi[find])
    
    f=f[find]
    
    f=array(f)
    H=array(H)
    phi=array(phi)
    
    return f,H,phi
    

def ZeroCrossings(x):
    
    from numpy import sign, where, array
    
    zc=array(where((sign(x[0:-1])*sign(x[1::]))<0.))
    
    zc=zc[0,:]
    
    return zc
    
def EchoSeparate(x,N,db=-20,ws=0.1):

    from numpy import zeros,array
    from scipy.signal import hilbert
    
    xa=hilbert(x.copy())
    Xa=abs(xa)

    ipks=LocalMax(Xa)
        
    pks=Xa[ipks]  
    p=pks.copy()
    pmin=pks.min()
    ind=[]
    i=p.argmax()
    ind.append(ipks[i])
    p[i]=pmin
    il,ir=PeakLimits(Xa,ipks[i],db)
    ispace=round((ir-il)/2)
    n=0

    while len(ind)<N:
        
        i=p.argmax()
        ii=ipks[i]
        if all([abs(I-ii)>ispace for I in ind]):
            ind.append(ii)
            n+=1

        p[i]=pmin
    
    ind.sort()
    i0=il.copy()

    x=x.copy()

    xe=[]
    xx=zeros(len(x))
    xx[il:ir]=x[il:ir]*tukeywin(ir-il,ws)
    xe.append(xx)

    for n in range(1,N):

        il,ir=PeakLimits(Xa,ind[n],db)
        xx=zeros(len(x))
        xx[il:ir]=x[il:ir]*tukeywin(ir-il,ws)

        xe.append(xx)

    xe=array(xe)
    xe=xe[:,i0::].transpose()

    return xe
   
def NearestValue(vec,val):
    
    return abs(vec-val).argmin()
    
def NPeakLimits(x,N,frac,nextzero=True):
    
    from scipy.signal import hilbert
    
    
    zc=ZeroCrossings(x)
    
    while True:
            
        ind=[]
        xr=abs(hilbert(x))
        
               
        for n in range(N):
        
            try:
        
                il,ir=PeakLimits(xr,xr.argmax(),frac)
            
            except:
                
                frac=frac*1.01
                break
                
           
        
            xr[il:ir]=0.
            
            if nextzero is True:
        
                il=zc[NearestValue(zc,il)]
                ir=zc[NearestValue(zc,ir)]
        
            ind.append((il,ir))
            
            
        if len(ind)==N:
            
            break
       
        
    ind.sort()
    
    return ind
    
# def DecayConstant(x,N,frac=0.1):
#
#     from numpy import correlate
#     from numpy.linalg import norm
#
#     ind=NPeakLimits(x,1,frac)
#
#     x1=x[ind[0][0]:ind1[0][1]]
#     x2=x[ind[0][1]::]/norm(x1)
#
#     xx=

# def AmpDelay(x,dt,N,frac=0.01):
#
#     from numpy import correlate,array,sign
#     from peakutils import indexes
#     from numpy.linalg import norm
#
#
#     x1=EchoSeparate(x,1,frac)
#     x1=x1[0]
#     x2=x.copy()/norm(x1)
#     x1=x1/norm(x1)
#
#     xx=correlate(x2,x1,'full')
#
#     xxa=abs(xx)
#
#     il,ir=PeakLimits(xxa,xxa.argmax(),)
#
#     Thresh=0.05
#
#     while True:
#
#         ind=indexes(xxa,thres=Thresh,min_dist=(ir-il))
#
#         if len(ind)-1<N:
#
#             Thresh=Thresh*0.9
#
#         else:
#
#             break
#
#
#
#     i0=ind[0]
#     ind=ind[1::]
#
#
#
#     A=[]
#     T=[]
#
#     for n in range(N):
#
#         T.append(dt*(ind[n]-i0))
#         A.append(xxa[ind[n]]*sign(xx[ind[n]].real))
#
#     A=array(A)
#     T=array(T)
#
#     return T,A
    
      
# def EchoSeparate(x,N,db=-14,ws=1.,mindelay=150):
#
#     from numpy import correlate,array,sign,zeros
#     from peakutils import indexes
#     from numpy.linalg import norm
#     from scipy.signal import hilbert
#     # from matplotlib.pylab import plot,show
#
#
#
#     xa=abs(hilbert(x))
#
#     il,ir=PeakLimits(xa,xa.argmax(),db)
#
#     x1=zeros(len(x))
#     win=tukeywin(ir-il,ws)
#     lw=len(win)
#     print(lw)
#     print()
#     x1[il:ir]=x.copy()[il:ir]*win
#
#     x2=x.copy()/norm(x1)
#     x1=x1/norm(x1)
#
#     xx=correlate(x2,x1,'full')
#
#     # xx=xx[len(xx)/2::]
#
#     xxa=abs(hilbert(xx))
#
#     # plot(xxa)
#  #    show()
#
#     Thresh=0.05
#
#     while True:
#
#         ind=indexes(xxa,thres=Thresh,min_dist=mindelay)
#
#         if len(ind)-1<N:
#
#             Thresh=Thresh*0.9
#
#         else:
#
#             break
#
#
#
#     i0=ind[0]
#     ind=ind[1::]
#
#     xe=[]
#     xe.append(x1)
#
#
#     for n in range(N):
#
#
#         Xe=zeros(len(x))
#         Xe[il+ind[n]-i0:ir+ind[n]-i0]=win*x2[il+ind[n]-i0:ir+ind[n]-i0]
#         xe.append(Xe)
#
#         # Xe[ind[n]-il:ind[n]+ir]=x2[ind[n]-il:ind[n]+ir]*win
#         # xe.append(Xe)
#
#
#     return xe
    
    
# x=signal, N = number peaks, md = minimum peak distance
def DecayConstant(x,N,md):
    
    from numpy import log, polyfit, array, diff
    
    ind,A=Delays(x,N,mindelay=md)
    print(diff(ind))
    
    dc=-1*polyfit(array(range(N)),log(abs(A)),1)[0]
    
    return dc
    
def DiffCentral(x):
    
    from numpy import zeros,diff
    
    dx=zeros(x.shape)
    
    dx[0]=2*(x[1]-x[0])
    dx[-1]=2*(x[-1]-x[-2])
    
    for i in range(1,len(dx)-1):
        
        dx[i]=x[i+1]-x[i-1]
        
    return dx
    
def LocalMax(x):
    
    from numpy import diff,sign

    indmax=(diff(sign(DiffCentral(x))) < 0).nonzero()[0] + 1
    
    return indmax
      
def AmplitudeDelayPhase(x,N,dt,scale=1,db=-40,ws=0.01, debug=False):

    from numpy import correlate,array,angle,zeros,real,imag, mean,linspace
    from numpy.linalg import norm
    from scipy.signal import hilbert
    from matplotlib.pyplot import plot, show, figure

    X=x.copy()
    xa=abs(hilbert(X))
    il,ir=PeakLimits(xa,xa.argmax(),db)
    x1=zeros(len(X))
    win=tukeywin(ir-il,ws)
    x1[il:ir]=X[il:ir]*win
    x2=x.copy()/norm(x1)
    x1=x1/norm(x1)
    
    xa=ACorrelate(x1,x2,M=scale)
    xa=xa[abs(xa).argmax()::]
    Xa=abs(xa)      
    
    if debug:
        # figure()
        plot(dt*linspace(0,len(Xa)-1,len(Xa)),Xa)
        show()
    
    ipks=localmax(Xa)
   
    pks=Xa[ipks]  
    p=pks.copy()
    pmin=pks.min()
    ind=[]

    for n in range(N):

        i=p.argmax()
        ind.append(ipks[i])
        p[i]=pmin

    ind.sort()

    T=(dt/scale)*array(ind)
    A=Xa[ind]
    phi=angle(xa[ind])

    return A,T,phi
    
def ACorrelate(x,y,M=1):
    from numpy.fft import fft,ifft,fftshift
    from numpy import conj,zeros,array,max,hstack, argmax
    from matplotlib.pyplot import plot
    
    N = 1 << (argmax([max(array([len(x),len(y)])) & (1<<i) for i in range(0, 32)]) + 1)
    # N=max([len(x),len(y)])
    X=fft(x, 2*N)
    Y=fft(y, 2*N)

    Cyx=2*Y*conj(X)
    Cyx[0]=Cyx[0].copy()/2
    Cyx[N]=Cyx[N].copy()/2
    Cyx[N+1::]=0
    Cyx=hstack((Cyx.copy(),zeros(2*N*(M-1))))

    cyx=fftshift(ifft(Cyx))*M

    return cyx
    
def find_pulses(x, k, h, n):
    """
        SOURCE: http://www.researchgate.net/publication/228853276_Simple_Algorithms_for_Peak_Detection_in_Time-Series    
    
        :param x(list): raw signal
        :param k(int): roughly half the length of a pulse
        :param h(int): smaller = more sensitive to peaks. Generally (0,1]
        :param n(int): number of peaks to detect. If 0, just detects peak with h, otherwise
        reduces h until number of peaks is met
    """
    from numpy import array, std, correlate, argmax
    from scipy.signal import gaussian

    ref = gaussian(k, k/6) # 3 standard deviations
    x = correlate(abs(x), ref, 'full')
    S = S1(x, k)
    s = std(S)
    h = [0] + [h] + [3] # assuming max h will be three and min 0
    curh = 1
    while(True):
        candidates = []
        for i in range(2*k, len(S)-2*k): # ignore first and last k since they are artifical
            if (S[i]) > h[curh]*s:
                candidates.append(i)
    
        pulses = []
        start = 0
        for i in range(0, len(candidates) - 1):
            if candidates[i+1] - candidates[i] > k: # found break-point of cluster
                pulses.append(candidates[start:i+1][argmax([S[j] for j in candidates[start:i+1]])])
                start = i+1
        # at the end of the for-loop there is definiately a peak which isn't added because there
        # was no peak infront of it itself to trigger the if-statement. Add this last peak
        pulses.append(candidates[start:len(candidates)][argmax([S[j] for j in candidates[start:len(candidates)]])])
                
        if(n==0 or n==len(pulses)): break
        if(len(pulses) < n):
            h.append(0.5*h[curh] + 0.5*h[curh-1])
            h.sort()
        else:
            h.append(0.5*h[curh] + 0.5*h[curh+1])
            h.sort()
            curh = curh + 1
    return array(pulses)-0.5*k, candidates, S, s
    
def find_pulses_max(x, pulses, k):
    """
    Finds the maximum of abs(x) given a rough location and pulse width.
    """
    from numpy import argmax
    peaks = []
    absx = abs(x)
    for i in pulses:
        peaks.append(i-k + argmax(absx[i-k:i+k]))
    return peaks
     
def S1(data, k):
    """ Scoring function for find_peaks.
    """
    from numpy import array, concatenate
    S1 = [0 for i in range(0, len(data))]
    x = concatenate((array([0 for i in range(0,k)]), data, array([0 for i in range(0,k)])), axis=0)
    
    S1[0] = k*x[k] + sum([(x[k] - xj) for xj in x[k+1:2*k+1]])
    
    for i in range(k+1, len(data)+k): # i+k+1 = n+k-1 + k + 1 = n+2k
        S1[i-k] = S1[i-k-1] + (2*k+1)*x[i] - (2*k+1)*x[i-1] + x[i-k-1] - x[i+k]

    return S1



