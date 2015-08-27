def getSpeedOfSound(signal, dt, d):
    from numpy import correlate, argmax
    from matplotlib.pyplot import plot, ginput, cla        

    if(type(signal) == list):
        return [getSpeedOfSound(signal[i], dt, d[i]) for i in range(len(signal))]

    corr = (correlate(abs(signal), abs(signal), 'full'))
    cla()
    plot(corr)
    print("Please provide bounds")
    corr_max_bounds = ginput(2)
    
    zero_point = len(corr)/2
    corr_max = corr_max_bounds[0][0] + argmax(corr[corr_max_bounds[0][0]:corr_max_bounds[1][0]])

    deltaT = (corr_max - zero_point)*dt
    
    return (2*d)/(deltaT)
    
def getSpeedOfSoundAndThickness(signal_s, signal_w, dt, N, mindelay=100, c_w=1.487, c3=0, d3=0):
    """
        Return specimen depth and speed of sound through specimen.
        If d3=0 and c3=0, the assumption is that the specimen lies flat at some boundary.
        If the specimen is not on some boundary, then either c3 or d3 must be specified.
        If c3 is specified, it is necessary to select a third peak in the correlation.
    """
    from numpy import correlate, argmax
    from matplotlib.pyplot import plot, ginput, cla
    from spr import Delays
    
    if(type(signal_s) == list):
        tc = [getSpeedOfSoundAndThickness(s, signal_w, dt, N, mindelay, c_w, c3, d3) for s in signal_s]
        thickness = [x[0] for x in tc]
        waveSpeed = [x[1] for x in tc]
        return thickness, waveSpeed

    ind, A = Delays(2*signal_w+signal_s, N, mindelay=mindelay)
    zero_point = len(signal_s) # len of cross-correlation is len of signal * 2
    first_max = argmax(correlate(signal_s, signal_w, 'full'))
    second_max = first_max + ind[1]

    tc = abs(first_max - zero_point)*dt # in microseconds #tw-t1
    tb = abs(second_max - zero_point)*dt # tw-t2

    thickness = 0
    speed = 0    
    
    if(c3==0 and d3==0):
        thickness = 0.5*c_w*(tc)
        speed = 2*thickness / (tc-tb)
    
    if(c3 != 0):
        print("Please provide bounds for the third max => tw - tm")
        print("Note that this may not necessarily be the third furthest to the right")
        corr_max_bounds[2] = ginput(2)
        third_max = argmax(corr[corr_max_bounds[2][0][0]:corr_max_bounds[2][1][0]]) + corr_max_bounds[2][0][0]
        ta = (third_max - zero_point)*dt # tw-tm
        
        thickness = 0.5 * ( c_w*(tc) - c3*(tb-ta) )
        speed = 2*thickness / (tc-tb)
        
        print(c3*(tb-ta))
    
    if(d3!=0):
        thickness = 0.5 * ( c_w*(tc) - 2*d3 )
        speed = 2*thickness / (tc-tb) 

    return ( thickness, speed )
    
def myEquation(H, d, w0, w, a0, n, c0):
    from numpy import exp
    # this allocates more memory then it needs to...but it looks beautiful
    sum = 0
    for i in range(len(H)):
        sum = sum + abs( H[i] * exp(1j*Tau(d, w0, a0, n, c0, w[i])) - 1)
    return sum

def Tau(d, w0, a0, n, c0, w):
    from numpy import pi
    return 2.*d *( 2*a0*(pow(w/w0, n-1) - 1)/(pi*(1-n)*w0) + 1/c0)
    
#def myEquation2(w, a0, n, c0):
#    from numpy import exp
#    return exp(1j*Tau2(w, a0, n, c0))
#
#def Tau2(w, a0, n, c0):
#    from numpy import pi
#    d = 1.12
#    w0 = 5.5
#    return 2.*d / ((2*a0*(pow(w/w0, n-1) - 1)/pi/(1-n)/w0) + c0)
    
def minimize(equation, setParameters, start, stop, step):
    from numpy import concatenate, arange
    
    minParams = concatenate((setParameters, start))
    minVal = equation(*minParams)    

    # all possible param values. Probably a bad idea.
    params = [arange(start[i], stop[i]+step[i], step[i]) for i in range(0, len(start))] 
    
    start = [0 for i in range(0, len(params))]
    start[-1] = -1 # trust me
    end = [len(params[i]) for i in range(0, len(params))]
    current = list(start)
    clfi = len(current) - 1 # current for-loop index

    counter = 0

    while(current!=end):
        current[clfi] = current[clfi] + 1
        if (current[clfi] == end[clfi]):
            clfi = clfi-1
        else:
            if ( clfi == (len(current) - 1) ): # inner-most for-loop
                counter = counter + 1
                testParams = concatenate((setParameters, [params[i][current[i]] for i in range(0, len(params))]))
                testVal = equation(*testParams)
                if (testVal < minVal):
                    minVal = testVal
                    minParams = list(testParams)[len(setParameters):]
            else:
                clfi = clfi + 1
                current[clfi] = start[clfi]

    return minVal, minParams    
    
def exponential_fct(x, alpha, eta):
    from numpy import ndarray
    if type(x) is list or type(x) is ndarray:
        return [alpha*pow(x[i], eta) for i in range(0, len(x))]
    return alpha*pow(x, eta)

def exponential_fit(xdata, ydata):
    from scipy.optimize import curve_fit
    return curve_fit(exponential_fct, xdata, ydata)

def getPhaseVelocityAndAttenuation(signal, dt, d2, frng, winsharp=0.1,df=0.1):
    from matplotlib.pyplot import close, plot, ginput, show, waitforbuttonpress
    from numpy import zeros, angle, conjugate, unwrap, pi, linspace, log, ndarray, argmax
    from numpy.fft import rfft
    from spr import tukeywin
    from spr import EchoSeparate
    
    x = signal

    if type(x) is list:
        
        CA = [getPhaseVelocityAndAttenuation(signal[i], dt[i], d2[i], frng, winsharp, df) for i in range(0, len(signal))]
        C = [c[0] for c in CA]
        Alpha = [a[1] for a in CA]
        
    elif type(x) is ndarray:
        
        xe = EchoSeparate(signal, 2)
    
        x=xe[0]
        x=x*tukeywin(len(x),winsharp)

        # removed negative sign for primer to steel
        y=xe[1]
        y=y*tukeywin(len(y),winsharp)

        TT=1.

        X = rfft(x,int(1/(df*dt)))
        X_CONJ = conjugate(X)
        Y = rfft(y,int(1/(df*dt)))
        H = (Y*X_CONJ)/(X*X_CONJ)

        w=2*pi*linspace(1.e-15,1./(2*dt),len(X))

        phi=unwrap(angle(Y))-unwrap(angle(X))

        Tau=dt*(argmax(abs(xe[1]))-argmax(abs(xe[0])))-phi/w
        c=2*d2/Tau

        H=H[(w>=2*pi*frng[0])&(w<=2*pi*frng[1])]

        C=c[(w>=2*pi*frng[0])&(w<=2*pi*frng[1])]

        Alpha=-log(abs(H)/TT)/(2*d2)
        
    return C, Alpha

def getSpeedOfSoundInWater(T, S=0, D=0):
    from math import pow
    """
        Return speed of sound in water. Accurate for 0<T<35, 0<D<1000.
        Ref: Speed of sound in water: A simple equation for realistic parameters. Herman Medwin 
        
    :Arguments
        - T     - Temperature (C)
        - S     - Salinity (ppt)
        - D     - Depth (m)
    """
    return (1449.2 + 4.6*T - 0.055*pow(T,2) + 0.00029*pow(T,3) + (1.34 - 0.010*T)*(S-35) + 0.016*D) * 0.001 # mm/us

def Save(data,filename,writemode='new'):
        
    
    import pickle,os
    
    # data is a dictionary containing waveforms
    
    if os.path.isdir('/Users/jlesage/Dropbox/ShawCor/'):
        
        pth='/Users/jlesage/Dropbox/ShawCor/'
        
    elif os.path.isdir('c:/Users/undel3/Dropbox/ShawCor'):
    
        pth='c:/Users/undel3/Dropbox/ShawCor/'
        
    else:
        
        pth=input('Input Valid Path to Store '+filename+':' )
        
    fl=pth+filename+'.p'
    print(fl)
        
    # if (os.path.isfile(fl))&(writemode is 'append'):
    
    if writemode[0] is 'replace':
        
        old=pickle.load(open(fl,'rb'))
        pickle.dump(data,open(fl,'wb'))
        return old
     
    elif writemode is 'new':
            
        pickle.dump(data,open(fl,'wb'))
        
def Load(filename):
    
    import pickle,os
    
    if os.path.isdir('/Users/jlesage/Dropbox/ShawCor/'):
        
        pth='/Users/jlesage/Dropbox/ShawCor/'
        
    elif os.path.isdir('c:/Users/undel3/Dropbox/ShawCor'):
    
        pth='c:/Users/undel3/Dropbox/ShawCor/'
        
    else:
        
        pth=input('Input Valid Path to Store '+filename+':' )
        
    fl=pth+filename+'.p'
        
    s=pickle.load(open(fl,'rb'), encoding='latin1')
    
    return s
    
def LoadMultiple(files,key,ind):
    
    import pickle
    
    x=[]
    
    for f in files:
        
        xx=pickle.load(open('/Users/jlesage/Dropbox/ShawCor/PipeSample'+f+'.p','rb'))
        
        xx=xx[ind][key]
        
        for j in range(len(xx)-1):
            
            x.append(xx[j])
            
    return x
    
def PipeGrid(angular, axial):
    #import itertools
    wSteps = round((angular[1]-angular[0])/angular[2]) # number of w steps
    hSteps = round((axial[1]-axial[0])/axial[2]) # number of h steps    
    locations = [i for i in range(0, (wSteps+1)*(hSteps+1))]
    for i in range(0, wSteps+1):
        for j in range(0, hSteps+1):
            if i%2==0:
                locations[i*(hSteps+1) + j] = (angular[0]+angular[2]*i, axial[0]+axial[2]*j)
            else:
                locations[i*(hSteps+1) + j] = (angular[0]+angular[2]*i, axial[0]+axial[2]*(hSteps-j))
    return locations
    #return list(itertools.product(*([range(angular[0],angular[1]+angular[2],angular[2]), range(axial[0],axial[1]+axial[2],axial[2])])))
    
    
def GetSignals(nlocs,Keys,Vals,navg=512):
    
    from Ultrasonic import GetData
    
    t0,dt,x=GetData(navg,nlocs)

    data={'TimeOrigin':t0,'SamplingPeriod':dt,'Signals':x}
    
    for i in range(len(Keys)):
        
        data[Keys[i]]=Vals[i]
        
    return data
    
def ReflectionSpectrum(data,srange,ds,winsharp=0.1):
    
    from spr import tukeywin
    from numpy import linspace
    from numpy.fft import rfft
    from numpy.linalg import norm
    
    x=data['Data']
    dt=data['SamplingPeriod']
    
    s=[]
    X=[]
    
    
    for i in range(len(x)):
        
        xx=x[i]*tukeywin(len(x[i]),winsharp)
     
        XX=rfft(xx,int(round(1/(dt[i]*ds)-len(xx))))

        ss=linspace(0.,1/(2*dt[i]),len(XX))

        XX=XX[(ss>=srange[0])&(ss<=srange[1])]
        XX=XX/norm(XX)

        ss=ss[(ss>=srange[0])&(ss<=srange[1])]
            
        X.append(XX)
        s.append(ss)
        
    data['ReflectionSpectrum']=X
    data['Frequency']=s
            
    

    
    # def GetMinFreq(data,ds,srange,window=('tukeywin',0.1)):
    #
    #     from spr import cutwvfrm,localmin,moments
    #     from numpy import linspace
    #     from numpy.fft import rfft
    #     from numpy.linalg import norm
    #
    #
    #
    #     X=[]
    #     smin=[]
    #     mu=[]
    #     s=[]
    #
    #     print(len(x))
    #
    #     for i in range(len(x)):
    #
    #         tt,xx,ind=cutwvfrm(t[i],x[i],False,window)
    #
    #         XX=rfft(xx,int(round(1/(dt[i]*ds)-len(xx))))
    #
    #         ss=linspace(0.,1/(2*dt[i]),len(XX))
    #
    #         XX=XX[(ss>=srange[0])&(ss<=srange[1])]
    #         XX=XX/norm(XX)
    #
    #         ss=ss[(ss>=srange[0])&(ss<=srange[1])]
    #
    #         sm=moments(ss,abs(XX))
    #
    #         indmin=localmin(abs(XX))
    #
    #         mu.append(sm)
    #
    #         smin.append(ss[indmin])
    #
    #         X.append(XX)
    #
    #         s.append(ss)
            



def MeasureBeta(t,x,sigma,SNR,fc=5.,gateoff=2*7.9e-3/6000.):
    
    from ultra import SignalIndex
    from scipy.signal import hilbert
    from numpy import blackman,zeros,linspace
    from numpy.fft import rfft
    from spr import pkfind,cutwvfrm
    
    dt=abs(t[1]-t[0])
    
    xa=abs(hilbert(x))
    
    # indgate=int(gateoff/dt)
 #
 #
 #    indmax1=xa.argmax()
 #
 #    il1,ir1=SignalIndex(xa,indmax1,sigma,SNR)
 #
 #    indmax2=xa[indgate+il1:indgate+ir1+1].argmax()+indgate
 #
 #    il2,ir2=SignalIndex(xa,indmax2,sigma,SNR)
 #
 #    indmax3=xa[2*indgate+il1:2*indgate+ir1+1].argmax()+2*indgate
 #
 #    il3,ir3=SignalIndex(xa,indmax3,sigma,SNR)
 #
 #    print(indgate)
 #
    # print(indmax1)
    # print(il1)
    # print(ir1)
    #
    # print(indmax2)
    # print(il2)
    # print(ir2)
 #
 #
 #    print(indmax3)
 #    print(il3)
 #    print(ir3)
 
    T,X,ind1=cutwvfrm(t,x)
    T,X,ind2=cutwvfrm(t,x)
 #    T,X,ind3=cutwvfrm(t,x)
 #
    
    
    x1=zeros(x.shape)
    
    x1[ind1]=x[ind1]
    
    x2=zeros(x.shape)
    
    x2[ind2]=x[ind2]
    
    # x3=zeros(x.shape)
    
    # x3[ind3]=x[ind3]
    
    # x1[il1:ir1+1]=blackman(ir1-il1+1)*x[il1:ir1+1]
   #  x2[il2:ir2+1]=blackman(ir2-il2+1)*x[il2:ir2+1]
  #   x3[il3:ir3+1]=blackman(ir3-il3+1)*x[il3:ir3+1]
    
    X1=rfft(x1)
    X2=rfft(x2)
    # X3=rfft(x3)
    
    # beta=X1*X3/(X2*X2)
    
    beta=X1/X2
    
    f=linspace(0,1/(2*dt),len(beta))*1e-6
    
    betac=beta[int(fc/(f[1]-f[0]))]
    
    return f,beta,betac,x1,x2
    
    
def ComputeBeta(f,rho1,rho2,c1,c2,eta1,eta2,K):
    
    from numpy import sqrt,pi
    
    w=2*pi*f
    
    Z1=rho1*c1*sqrt(1+1j*eta1)
    Z2=rho2*c2*sqrt(1+1j*eta2)
    
    # beta=(-1./4.)*((Z1**2+Z2**2)/(Z1*Z2)-2.+(Z1*Z2*w**2)/(K**2))
    
    beta=((Z1 - Z2 + (Z1*Z2*w*1j)/K)*(Z2 - Z1 + (Z1*Z2*w*1j)/K))/(4*Z1*Z2)
    
    return beta
    
    
def MeasureRavg(t,x,npeaks):

    from spr import pkfind
    from scipy.signal import hilbert
    from numpy import mean
    
    Ravg=[]
    
    for i in range(len(x)):
    
        # tp,X=pkfind(t[i],abs(hilbert(x[i])),npeaks)
        
        tp,X=pkfind(t[i],abs(hilbert(x[i])),npeaks)
        
    
        R=[]
    
        for n in range(npeaks-1):
        
            R.append(X[n+1]/X[n])
            
        Ravg.append(mean(R))
        
    if len(Ravg)==1:
        
        Ravg=Ravg[0]
        
    return Ravg
        
    
def ComputeR(f,rho1,rho2,c1,c2,eta1,eta2,K):
    
    from numpy import sqrt,pi
    
    w=2*pi*f
    
    Z1=rho1*c1*sqrt(1+1j*eta1)
    Z2=rho2*c2*sqrt(1+1j*eta2)
    
    R=(Z2-Z1+1j*w*Z1*Z2/K)/(Z1+Z2-1j*w*Z1*Z2/K)
    
    return R


def PlateSpectrum(srng,layer,prop,perturbation,wavetype='Longitudinal'):
    
    # This function computes the frequency spectrum for the pipe (modelled as a 1 dimensional, multilayered plate) subject
    # to a perturbation in wavespeed, attenuation, density or thickness
    
    # srng is a list of the form [frequency 1, frequency 2, frequency step ] 
    
    # layer is the index of the layer who's properties are perturbed:  0 - plastic, 1 - bulk adhesive, 2 - interface layer
    # 3 - primer, 4 - steel
    
    # prop is a string: 'l'-thickness, 'c' - speed of sound, 'rho' - density, 'alpha' - attenuation
    
    # perturbation is the amount by which the prop is to be perturbed from its nominal value (-1 < perturbation < 1)
    
    # wavetype is the type of wave propagation being studied (default - 'Longitudinal')
    
    from Elastodynamics.plate import ComputeU
    from spr import localmax
    from numpy import arange, mean, zeros, complex128
    from copy import deepcopy
    
    if wavetype is 'Longitudinal':
        
        P={'l':[1.1,1.7,0.01,0.1,4.9],'c':[2.2,1.9,0.,3.,5.9],'rho':[0.9,0.93,0.,1.2,7.8],'alpha':[0.4,0.6,0.,0.6,0.01]}
    
    elif wavetype is 'Shear':
        
        P={'l':[1.1,1.7,0.01,0.1,4.9],'c':[2.2,1.9,0.,3.,5.9],'rho':[0.9,0.93,0.,1.2,7.8],'alpha':[0.4,0.6,0.,0.6,0.01]}
        
    
    P['c'][2]=mean((P['c'][1],P['c'][3]))
    P['rho'][2]=mean((P['rho'][1],P['rho'][3]))
    P['alpha'][2]=mean((P['alpha'][1],P['alpha'][3]))
    
    
    
    s=arange(srng[0],srng[1],srng[2])
    
    # F=zeros(6,len(s),dtype=complex)
    
    F=zeros((6,len(s)),dtype=complex128)
    F[0,:]=1.+1j*0.

    U=[]
    sres=[]
    
    for p in perturbation:
        
        PP=deepcopy(P)
                
        PP[prop][layer]=(PP[prop][layer])*(1+p)
                
        UU=ComputeU(s,PP['l'],PP['c'],PP['rho'],PP['alpha'],F)
        sres.append(s[localmax(abs(UU[:,0]))])
        U.append(UU)
        
    
    return s,sres,U
    
def ComputeResponse(sc,T,rho,c,alpha,d):
    
    from numpy import tan,pi,linspace,exp,zeros,hstack,vstack,array
    from scipy.signal import gausspulse
    from numpy.fft import rfft,ifft

    dt=1/(10*sc)
    t=linspace(0,T,round(T/dt))
    X=rfft(gausspulse((t-0.25*T),sc))
    
    Z=[]

    for i in range(len(rho)):
        Z.append(rho[i]*c[i])


    w=2*pi*linspace(0,1/(2*dt),len(X))

    R01=(Z[1]-Z[0])/(Z[1]+Z[0])
    T01=R01+1
    T10=1-R01
    R12=(Z[2]-Z[1])/(Z[1]+Z[2])
    T12=R12+1
    T21=1-R12

    Z234=Z[3]*(Z[4]-1j*Z[3]*tan(w*d[3]/c[3]))/(Z[3]-1j*Z[4]*tan(w*d[3]/c[3])) 
    R234=(Z234-Z[2])/(Z234+Z[2])

    T234=R234+1

    R45=(Z[5]-Z[4])/(Z[5]+Z[4])
    T45=R45+1
    

    Z432=Z[3]*(Z[2]-1j*Z[3]*tan(w*d[3]/c[3]))/(Z[3]-1j*Z[2]*tan(w*d[3]/c[3])) 
    R432=(Z432-Z[4])/(Z432+Z[4])

    T432=R432+1

    Y0=exp(-2*d[0]*alpha[0])*exp(-1j*w*2*d[0]/c[0])*R01*X
    Y1=exp(-2*d[1]*alpha[1])*exp(-1j*w*2*d[1]/c[1])*T01*T10*R12*Y0/R01

    Y2=exp(-2*d[2]*alpha[2])*exp(-1j*w*2*d[2]/c[2])*T12*T21*R234*Y1/R12
  
    Y3=exp(-2*d[4]*alpha[4])*exp(-1j*w*2*d[4]/c[4])*T234*T432*R45*Y2/R234
    
    Y4=exp(-2*d[4]*alpha[4])*exp(-1j*w*2*d[4]/c[4])*R432*R45*Y3

    Y5=exp(-2*d[4]*alpha[4])*exp(-1j*w*2*d[4]/c[4])*R432*R45*Y4

    y=2*ifft(vstack((Y0,Y1,Y2,Y3,Y4,Y5)),n=2*len(X)-1).transpose()

    x=2*ifft(X,n=2*len(X)-1)

    t=linspace(0,T,len(x))

    return t,x,y
    
def TransmissionReflection(s,rho,c,L,NpWl=10):
    
    from numpy import array,identity,pi,linspace,dot,arange,hstack,vstack,zeros,exp
    from scipy.linalg import expm
    from numpy.linalg import solve
    from Elastodynamics.TMatrix import TMatrix1d
    
    Ndiv=round(L[0]*s[-1]/min(c))*NpWl
    
    l=L[0]/Ndiv
    
    Y=linspace(l/2,L[0]-l/2,Ndiv)/L[0]
    
    
    RT=zeros((4,1))
    
    
    W=2*pi*s
    
    P = lambda x,y,z: TMatrix1d(z,(rho[0]*y+rho[1]*(1-y)),(c[0]*y+c[1]*(1-y)),x)
    
    for w in W:
        
        # P = lambda y: expm(l*array([[0.,1/((y*rho[1]*c[1]**2+(1-y)*rho[0]*c[0]**2))],[-(y*rho[1]+(1-y)*rho[0])*w**2,0.]]))
                        
        P2=P(L[1],1,w) 
    
        P02=identity(2)
        P30=P2
        
        for y in Y:
        
            P02=dot(P(l,y,w),P02)
            P30=dot(P(l,1-y,w),P30)
            
            # print(P(y))
   #          print(P(1-y))

        P03=dot(P2,P02)
    
        Z0=rho[0]*c[0]
        Z3=rho[2]*c[2]
    
        k0=w/c[0]
        k3=w/c[2]
    
        h=L[0]+L[1]
    
        RT03=solve(array([[-1j*w*Z0*P03[0,1]+P03[0,0],exp(1j*k3*h)],[-1j*w*Z0*P03[1,1]+P03[1,0],1j*w*Z3*exp(1j*k3*h)]]),dot(P03,array([[1],[1j*w*Z0]])))
        RT30=solve(array([[-1j*w*Z3*P30[0,1]+P30[0,0],exp(1j*k0*h)],[-1j*w*Z3*P30[1,1]+P30[1,0],1j*w*Z0*exp(1j*k0*h)]]),dot(P30,array([[1],[1j*w*Z3]])))
    
        RT=hstack((RT,vstack((RT03,RT30))))
    
    
    return RT[:,1::]
    
def DiffusiveFilter(sc,T,rho,c,L):
    
    from numpy import linspace,zeros,vstack,array,dot
    from scipy.signal import gausspulse
    from numpy.fft import rfft,ifft

    dt=1/(10*sc)
    t=linspace(0,T,round(T/dt))
    X=rfft(gausspulse((t-0.25*T),sc,bwr=-2.5))
    
    s=linspace(1e-5,1/(2*dt),len(X))
    
    RT=TransmissionReflection(s,rho,c,L)

    Y=X*RT
    
    
    y=2*ifft(Y,axis=1,n=2*Y.shape[1]-1)
    
    t=linspace(0,T,y.shape[1])
    
    return t,y,Y,RT
    

def LayerH(x,dt,N=5,Nfreqs=11,asteel=0.01,csteel=5.9):

    from spr import EchoSeparate,PeakLimits
    from numpy import linspace,angle,unwrap,exp,pi,polyfit,hstack,array
    from numpy.fft import rfft

    F=[]
    failind=[]

    for i in range(len(x)):

        try:
            
            xe=EchoSeparate(x[i]-mean(x[i]),N,db=-14)
            print(len(xe))
            
        except:
            
            failind.append(i)
            continue
            
        Xe=rfft(xe,axis=0)
        H=Xe[:,1::]/Xe[:,0:-1]
        imax=abs(Xe[:,0]).argmax()

        il,ir=PeakLimits(abs(Xe[:,0]),imax,db=-6)

        s=linspace(0,1/(2*dt),Xe.shape[0])

        phi=unwrap(angle(H),axis=0)
        phi=phi[il:ir,:]

        s=s[il:ir]

        p=[polyfit(2*pi*s,phi[:,i],1,full=True) for i in range(phi.shape[1])]

        H=abs(H[il:ir+1:int((ir-il)/Nfreqs),:])

        T1=-p[0][0][0]
        T2=-p[1][0][0]
        T3=-p[2][0][0]
        T4=-p[3][0][0]
        
        Phi1=p[0][0][1]
        Phi2=p[1][0][1]
        Phi3=p[2][0][1]
        Phi4=p[3][0][1]

        R1=1-p[0][1][0]/(len(s)*phi[:,0].var())
        R2=1-p[1][1][0]/(len(s)*phi[:,1].var())
        R3=1-p[2][1][0]/(len(s)*phi[:,2].var())
        R4=1-p[3][1][0]/(len(s)*phi[:,3].var())
        
        H[:,2]=exp(asteel*csteel*T4)*H[:,2]
        H[:,3]=exp(asteel*csteel*T4)*H[:,3]
            
        H=H.flatten()

        F.append(hstack((array([T1,T2,T4-T3,Phi1,Phi2,Phi3,Phi4]),H)))
            
    return array(F),failind
    
def AmpDelay(x,dt,N=4,asteel=0.01,csteel=5.9):
    
    from spr import AmplitudeDelayPhase
    from numpy import array,mean,exp
    from sklearn.preprocessing import scale
    
    F=[]
    
    for xx in x:
    
        A,T,phi=AmplitudeDelayPhase(xx-mean(xx),N,dt,db=-14)
        
        # F.append([A[0],T[0],phi[0],A[1],T[1],phi[1],A[2],T[2],phi[2],A[3],T[3],phi[3]])
        
        
        F.append([A[0],T[0],phi[0],A[1],T[1],phi[1],A[2]*exp(asteel*csteel*(T[-1]-T[-2])),T[-1]-2*T[-2]+T[-3],phi[2],A[3]*exp(2*asteel*csteel*(T[-1]-T[-2])),phi[3]])
        
    # return scale(array(F))
    
    return F
class Pipe:
    
    def __init__(self,PipeId=None,BondStrength=[]):
        # PipeId - Number identifying Pipe 
        # BondStrength - List identifying the range of peel strengths for the pipe (If sample is just called "Weak" use [0,5])
        #                (If sample is just called "Strong" use [150,300] ) 
        self.setConfiguration()        
        self.PipeId = PipeId
        self.BondStrength = BondStrength
        self.Signals = []
        self.Locations = []
        
    def setConfiguration(self):
        """ Reads 'config.ini' for variables associated with directories
        """
        import configparser, os
        self.config = configparser.ConfigParser()
        module_path = os.path.dirname(__file__)
        self.config.read_file(open(module_path + '\config.ini'))
        
    def ManualScan(self, samplingFrequency, Locations, Averages=512):
        
        from Ultrasonic import GetSignal
        from numpy.linalg import norm
        
        # This function takes scans at various locations along a pipe, 
        
        # Locations -  a list of lists defining the locations on the pipe in mm
        
        # ** Note: Make sure not to change the horizontal scale on the scope while scanning
        
        for l in Locations:
        
            input('Press any key to collect signal at point'+str(l)+'mm')
        
            t0,dt,x = GetSignal(navg=Averages, samplingFrequency = samplingFrequency)
            
            self.AddSignal(x/norm(x),l)
            
        self.SamplingPeriod = dt
        
    def AutoScan(self, Locations, samplingFrequency, Averages=512):
        import pygclib
        from Ultrasonic import GetSignal
        from numpy.linalg import norm
        
        curAngle = 0
        curZ = 0
        
        for l in Locations:
            deltaAngle, deltaZ = l[0]-curAngle, l[1] - curZ
            pygclib.move('Z', 10, deltaZ)
            pygclib.rotate(5, deltaAngle)
            curAngle = curAngle + deltaAngle
            curZ = curZ + deltaZ
            
            print('Acquiring Signal at '+str(curAngle)+' degrees, '+str(curZ)+' mm')
            
            t0,dt,x = GetSignal(navg=Averages, samplingFrequency = samplingFrequency)
            
            self.AddSignal(x/norm(x),l)
            
            
        self.SamplingPeriod = dt
        
        
    def ZeroMean(self):
        
        from numpy import mean
        
        s=self.Signals.copy()
        
        for i in range(len(s)):
            
            self.Signals[i]=s[i]-mean(s[i])
                
        
        
    def LocationIndices(self,Locations):
        
        from numpy import array,tile
        from numpy.linalg import norm
        from copy import deepcopy
        
        L=array(deepcopy(self.Locations))
        NL=L.shape[0]
        
        ind=[]
        
        for l in Locations:
            ind.append(norm(L-tile(array(l),(NL, 1)),1).argmin())
            
        return ind
        
    def isGoodSignal(self, signal, samplingPeriod, numSteelReverbs = 1, cSteel = 5.96):
        from numpy import diff, mean, array
        """
        (Attempts to) Returns False if signal was captured over double cladded point or from a peel-tested area
        """
        from spr import AmplitudeDelayPhase
        signal = signal - mean(signal)
        # T = time delay of i+1th pulse from the front-wall pulse in microseconds
        A, T, phi = AmplitudeDelayPhase(signal, 3+numSteelReverbs, samplingPeriod)
        if(max(A) > 1):
            print('Amplitude Greater Than 1')
            return False
        dA = diff(A)        
        # first difference must be greater than 0 and all subsequent must be less
        if (not (dA[1::]<0).all() ):
            print('Steel reverb amplitudes not decreasing')
            print(array(T)/samplingPeriod)
            AmplitudeDelayPhase(signal, 3+numSteelReverbs, samplingPeriod, debug=True)
            return False
        dT = diff(T)
        # heuristic observation that pulses cannot be less than 1.5 microseconds apart
        if(len(dT) == 0 or min(dT) < 1.25): 
            print('Delta T under 1.25 micros')
            return False
        # steel reverberation pulses must indicate thickness between 5 and 11
        if ((dT[2::]*cSteel/2 < 5).any() or (dT[2::]*cSteel/2>11 ).any()): 
            print('Steel wall thickness too big or too small: ' + str(dT[2::]*cSteel/2))
            return False
        return True
        
    def checkSignals(self):
        from matplotlib.pyplot import plot, figure, close, ginput
        for i in range(0, len(self.Signals)):
            print('Signal ' + str(i))            
            s = self.Signals[i]
            if not self.isGoodSignal(s, self.SamplingPeriod):
                figure()
                plot(s)
                ginput(timeout=0)
                close("all")
                
    def filterSignals(self):
        from tkinter import messagebox
        import tkinter
        root = tkinter.Tk()
        #root.withdraw()
        from matplotlib.pyplot import plot, figure, close
        toRemove = []
        for i in range(0, len(self.Signals)):
            print('Signal ' + str(i))            
            s = self.Signals[i]
            if not self.isGoodSignal(s, self.SamplingPeriod):
                figure()
                plot(s)
                if(messagebox.askyesno("Print", "Rmove this signal?")):
                    toRemove.append(i)
                close("all")
        root.destroy()
        self.Locations = [self.Locations[i] for i in range(0, len(self.Locations)) if i not in toRemove]
        self.Signals = [self.Signals[i] for i in range(0, len(self.Signals)) if i not in toRemove]
                      

    def ReturnSignals(self,Locations):
        
        from numpy import array
        from copy import deepcopy
        
        x=array(deepcopy(self.Signals))
        
        ind=self.LocationIndices(Locations)
        
        return list(x[ind,:])
        
    
    def DeleteSignals(self,Locations):
        
        from numpy import array,delete
        
        ind=self.LocationIndices(Locations)
        
        x=list(delete(array(self.Signals),ind,0))
        l=list(delete(array(self.Locations),ind,0))
        self.Locations=l
        self.Signals=x
        
        
    def AddSignal(self,Signal,Location,WriteMode='Append',SamplingPeriod=None):
        
        if WriteMode is 'Append':
        
            (self.Signals).append(Signal)
            (self.Locations).append(Location)
            
        elif WriteMode is 'Overwrite':
            
            self.Signals=Signal
            self.Locations=Location
        
        if SamplingPeriod is not None:
            
            self.SamplingPeriod = SamplingPeriod


    def Export(self,Filename,Path='C:/Users/utex3/Dropbox/ShawCor/pipe_auto_scans/',Format='Dictionary'):
        
        Path = self.config['DEFAULT']['pipe_c_scans_db']
        
        if Format == 'Text':
        
            from numpy import hstack,array,savetxt
        
            # Export Raw Data to a structured text file (comma delimited)
            data=hstack((array(self.Locations),array(self.Signals)))
        
            savetxt(Path+Filename+'.txt',data,delimiter=',',header=str(self.PipeId)+','+str(self.BondStrength[0])+','+str(self.BondStrength[1])+','+str(self.SamplingPeriod))
            
            
        elif Format == 'Dictionary':
            
            from pickle import dump
            
            data={'PipeId':self.PipeId,'BondStrength':self.BondStrength,'Locations':self.Locations,'Signals':self.Signals,'SamplingPeriod':self.SamplingPeriod}
            
            dump(data,open(Path+Filename+'.p','wb'))
        
    def Load(self,File):
        
        from copy import deepcopy
        import os
        if File.split('.')[1] == 'txt':
            from numpy import loadtxt
            File = self.config['DEFAULT']['pipe_c_scans_db'] + File
            data = loadtxt(File,delimiter=',')
            
            self.Signals=list(data[:,2::])
            self.Locations=list(data[:,0:2])
            
            with open(File,'r') as f:
                
                header=f.readline()
                
            header=header[2::].split(',')
            
            self.PipeId = int(header[0])
            self.BondStrength = [float(header[1]),float(header[2])]
            self.SamplingPeriod = float(header[3].rstrip())
            
        elif File.split('.')[1] == 'p':
        
            from pickle import load
            
            pipe = load(open(self.config['DEFAULT']['pipe_c_scans_db'] + File,'rb'))

            if type(pipe) is dict:
                
                self.PipeId = pipe['PipeId']
                self.BondStrength = pipe['BondStrength']
                self.Signals = pipe['Signals']
                self.Locations = pipe['Locations']
                self.SamplingPeriod = pipe['SamplingPeriod']


class PipeSet:
            
    def __init__(self,Path='/Users/jlesage/Dropbox/ShawCor/pipe_auto_scans/'):

        from os import listdir

        files=listdir(Path)
        Pipes=[]

        for f in files:
        
            p=Pipe()
            p.Load(Path+f)
            if len(p.BondStrength)==2:
                Pipes.append(p)

        self.Pipes=Pipes
    

    def ExtractFeatures(self,ScansPerPipe):

        from random import sample

        for p in self.Pipes:

            # p.Features = LayerH(p.Signals,p.SamplingPeriod)
            
            ind=sample(range(len(p.Signals)),ScansPerPipe)
                        
            signals = [p.Signals[i] for i in ind]
            
            p.Features = AmpDelay(signals,p.SamplingPeriod)
            p.FeatureIndices = ind
            

    def MakeTrainingSet(self,StrengthRanges):

        from numpy import vstack,zeros,hstack,ones,mean,shape
        from sklearn import preprocessing
        
        l,m=shape(self.Pipes[0].Features)
    
        
        X=zeros((1,m))
        y=zeros(1)
        
        for p in self.Pipes:
                                
            bs=mean(p.BondStrength)
                        
            for i in range(len(StrengthRanges)):
    
                if StrengthRanges[i][0]<=bs<=StrengthRanges[i][1]:
                    
                    X=vstack((X,p.Features))
                    y=hstack((y,i*ones(l)))
                            
        y=y[1::]
                    
        self.y=y.astype(int)
        
        X=X[1::,:]
        
        self.X=preprocessing.scale(X)

