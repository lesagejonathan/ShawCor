from numpy.fft import *
from numpy import *
from numpy.linalg import *
from scipy.signal import *
from spr import *
from scipy.optimize import *
from matplotlib import *
import os, pickle
from Ultrasonic import GetSignal
from scipy.linalg import expm
from Elastodynamics.TMatrix import TMatrix1d
from matplotlib.pyplot import plot, show

def VelocityAttenuation(s,dt,d,c0,fbnd,df=0.1,db=-25):

    from numpy import zeros

    NFFT = FFTLengthPower2(round(1/(df*2*dt)))

    s = s.copy()

    s = s-mean(s)

    sa = hilbert(s)

    i0 = round((2*d/c0)/dt)

    ix = abs(sa).argmax()

    il,ir = PeakLimits(abs(sa),ix,db=db)

    iw = max([ix-il,ir-ix])

    iy = abs(sa[ix+i0-iw:ix+i0+iw]).argmax()+ix+i0-iw

    w = tukeywin(2*iw,alpha=1.)

    # print(shape(w))
    # print(shape(s[ix-iw:ix+iw]))

    x = w*s[ix-iw:ix+iw]

    imx = abs(x).argmax()

    y = w*s[iy-iw:iy+iw]

    imy = abs(y).argmax()


    x = hstack((x[imx::],zeros(NFFT-len(x)),x[0:imx]))

    y = hstack((y[imy::],zeros(NFFT-len(y)),y[0:imy]))

    f = linspace(0,1/(2*dt),NFFT/2+1)

    ff = (f>=fbnd[0])&(f<=fbnd[1])

    X = rfft(x)

    Y = rfft(y)*exp(-1j*2*pi*f*(iy-ix)*dt)

    H = -Y/X

    phi = unwrap(angle(H))

    H = H[ff]

    f = f[ff]

    # plot(f,log(abs(H)))
    # plot(f,-4*pi*f*d/phi[ff])
    # show()

    c = -4*pi*f*d/phi[ff]

    T = -phi[ff]/(2*pi*f)

    A = log(abs(H))

    p = polyfit(f,A,1)

    x0 = [-p[0]/(2*d),2*d/(dt*(iy-ix))] 

    C = exp(p[1])

    # func = lambda x: 0.5*real(dot(conj(H-exp(-2*d*x[0]*f)*exp(-1j*4*pi*f*d*(1/x[1] - (2/pi)*x[0]*log(f/f[0])))).transpose(),H-exp(-2*d*x[0]*f)*exp(-1j*4*pi*f*d*(1/x[1] - (2/pi)*x[0]*log(f/f[0])))))

    func = lambda x: 0.5*real(dot(conj(H/C-exp(-2*d*x[0]*f)*exp(-1j*4*pi*f*d/x[1])).transpose(),H/C-exp(-2*d*x[0]*f)*exp(-1j*4*pi*f*d/x[1])))

    xopt = optimize.fmin(func,x0,full_output=True)

    # f = linspace(1e-6,1/(2*dt),len(X))

    # Ym = -C*exp(-2*d*xopt[0]*f)*exp(-1j*4*pi*f*d/xopt[1])*exp(1j*4*xopt[0]*d*log(2*pi*f)/pi)*X

    # ym = ifft(2*Ym,n=2*len(Ym)-2)

    # y = ifft(2*Y,n=2*len(Y)-2)

    # Hm = C*exp(-2*d*xopt[0][0]*f)*exp(-1j*4*pi*f*d*(1/xopt[0][1] - (2/pi)*xopt[0][0]*(log(2*pi*f) - log(2*pi*f[0]))))

    Hm = C*exp(-2*d*xopt[0][0]*f)*exp(-1j*4*pi*f*d*(1/xopt[0][1]))




    return xopt,f,Hm,H,c,x0


def getGroupVelocity(signal, dt, d, moving_average_n = 1, correlateSignal=True):
    """ Calculate the speed of sound using time interval between signal peaks.
        Correlates the signal with itself, plots it, and allows the user to select the bounds for two peaks.
        The function then takes the index of the maximum inside the bounds as the time delay.
        
        If correlate is false, the function simply plots the 'signal'
    """
    # from numpy import correlate, argmax
    # from matplotlib.pyplot import plot, ginput, figure
    # from spr import moving_average
    if correlateSignal:
        corr = (correlate(abs(signal), abs(signal), 'full'))
        corr = moving_average(corr, moving_average_n)
    else:
        corr = signal
    figure()
    plot(corr)
    print("Please provide bounds")
    bounds = ginput(4)
    t1 = bounds[0][0] + argmax(corr[bounds[0][0]:bounds[1][0]])
    t2 = bounds[2][0] + argmax(corr[bounds[2][0]:bounds[3][0]])
    deltaT = abs(t2 - t1)*dt
    return (2*d)/(deltaT)
    
def getGroupAttenuation(signal, dt, d, moving_average_n = 1, correlateSignal=True):
    # from numpy import log, correlate
    # from matplotlib.pyplot import plot, ginput, figure
    # from spr import moving_average
    if correlateSignal:
        corr = (correlate(abs(signal), abs(signal), 'full'))
        corr = moving_average(corr, moving_average_n)
    else:
        corr = signal
    figure()
    plot(corr)
    print("Please provide bounds")
    bounds = ginput(4)
    a0 = max(corr[bounds[0][0]:bounds[1][0]])
    a1 = max(corr[bounds[2][0]:bounds[3][0]])
    return (1/(2*d))*log(a0/a1)
    
    
    
def getSpeedOfSoundAndThickness(signal_s, signal_w, dt, N, mindelay=100, c_w=1.487, c3=0, d3=0):
    """
        Return specimen depth and speed of sound through specimen.
        If d3=0 and c3=0, the assumption is that the specimen lies flat at some boundary.
        If the specimen is not on some boundary, then either c3 or d3 must be specified.
        If c3 is specified, it is necessary to select a third peak in the correlation.
    """
    # from numpy import correlate, argmax
    # from matplotlib.pyplot import plot, ginput, cla
    # from spr import Delays
    
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
    # from numpy import exp
    # this allocates more memory then it needs to...but it looks beautiful
    sum = 0
    for i in range(len(H)):
        sum = sum + abs( H[i] * exp(1j*Tau(d, w0, a0, n, c0, w[i])) - 1)
    return sum

def Tau(d, w0, a0, n, c0, w):
    # from numpy import pi
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
    # from numpy import concatenate, arange
    
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
    # from numpy import ndarray
    if type(x) is list or type(x) is ndarray:
        return [alpha*pow(x[i], eta) for i in range(0, len(x))]
    return alpha*pow(x, eta)

def exponential_fit(xdata, ydata):
    from scipy.optimize import curve_fit
    return curve_fit(exponential_fct, xdata, ydata)

def getPhaseVelocityAndAttenuation(signal, dt, d2, frng, winsharp=0.1,df=0.1):
    # from matplotlib.pyplot import close, plot, ginput, show, waitforbuttonpress
    # from numpy import zeros, angle, conjugate, unwrap, pi, linspace, log, ndarray, argmax
    # from numpy.fft import rfft
    # from spr import tukeywin
    # from spr import EchoSeparate
    
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
    # from math import pow
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
        
    
    # import pickle,os
    
    # data is a dictionary containing waveforms
    
    if os.path.isdir('/Users/jlesage/Dropbox/ShawCor/'):
        
        pth='/Users/jlesage/Dropbox/ShawCor/'
        
    elif os.path.isdir('c:/Users/undel3/Dropbox/ShawCor'):
    
        pth='c:/Users/undel3/Dropbox/ShawCor/'
    
    elif os.path.isdir('c:/Users/utex3/Dropbox/ShawCor'):
        pth = 'c:/Users/utex3/Dropbox/ShawCor/'
        
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
    
    # import pickle,os
    
    if os.path.isdir('/Users/jlesage/Dropbox/ShawCor/'):
        
        pth='/Users/jlesage/Dropbox/ShawCor/'
        
    elif os.path.isdir('c:/Users/undel3/Dropbox/ShawCor'):
    
        pth='c:/Users/undel3/Dropbox/ShawCor/'
        
    elif os.path.isdir('c:/Users/utex3/Dropbox/ShawCor'):
        pth = 'c:/Users/utex3/Dropbox/ShawCor/'
        
    else:
        
        pth=input('Input Valid Path to Store '+filename+':' )
        
    fl=pth+filename+'.p'
        
    s=pickle.load(open(fl,'rb'), encoding='latin1')
    
    return s
    
def LoadMultiple(files,key,ind):
    
    # import pickle
    
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
    
    
def GetSignals(nlocs,Keys,Vals,navg=512, sF = 50):
    
    from Ultrasonic import GetSignal
    
    signals = []    
    data = {}
    
    for i in range(0, nlocs):
        t0,dt,x=GetSignal(navg, sF)
        signals.append(x)
        data = {'TimeOrigin':t0,'SamplingPeriod':dt}
    
    data['signals'] = signals    
    
    for i in range(len(Keys)):    
        data[Keys[i]]=Vals[i]
        
    return data

    
def ComputeResponse(sc,T,rho,c,alpha,d):
    
    # from numpy import tan,pi,linspace,exp,zeros,hstack,vstack,array
    # from scipy.signal import gausspulse
    # from numpy.fft import rfft,ifft

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
    
    # from numpy import array,identity,pi,linspace,dot,arange,hstack,vstack,zeros,exp,ceil,sqrt
    # from scipy.linalg import expm
    # from numpy.linalg import solve
    # from Elastodynamics.TMatrix import TMatrix1d
    
    Ndiv=ceil(NpWl*L[0]*s[-1]/min(c))
    
    l=L[0]/Ndiv
    
    # print(Ndiv)
   #  print(l)
    
    Y=linspace(l/2,L[0]-l/2,Ndiv)/L[0]
    
    
    RT=zeros((4,1))
    
    M0=rho[0]*c[0]**2
    M1=rho[1]*c[1]**2
    
    
    W=2*pi*s
    
    Rho = lambda y: rho[1]*y+rho[0]*(1-y)
    
    C = lambda y: sqrt((M1*y+M0*(1-y))/Rho(y))
    
    P = lambda x,y,z: TMatrix1d(z,Rho(y),C(y),x)
    
    for w in W:
        
                        
        P2=P(L[1],1,w) 
    
        P02=identity(2)
        P30=P2
        
        for y in Y:

            P02=dot(P(l,y,w),P02)
            P30=dot(P(l,1-y,w),P30)


        P03=dot(P2,P02)
    
        Z0=rho[0]*c[0]
        Z3=rho[2]*c[2]
    
        k0=w/c[0]
        k3=w/c[2]
    
        h=L[0]+L[1]
    

        RT03=solve(array([[1j*w*Z0*P03[0,1]+P03[0,0],exp(-1j*k3*h)],[1j*w*Z0*P03[1,1]+P03[1,0],-1j*w*Z3*exp(-1j*k3*h)]]),dot(P03,array([[1],[-1j*w*Z0]])))
        RT30=solve(array([[1j*w*Z3*P30[0,1]+P30[0,0],exp(-1j*k0*h)],[1j*w*Z3*P30[1,1]+P30[1,0],-1j*w*Z0*exp(-1j*k0*h)]]),dot(P30,array([[1],[-1j*w*Z3]])))
    
        RT=hstack((RT,vstack((RT03,RT30))))
    
    
    return RT[:,1::]
    
def DiffusiveFilter(sc,T,rho,c,L,dt=0.001,BW=0.7):
    
    # from numpy import linspace,zeros,vstack,array,dot,conj,pi
    # from scipy.signal import gausspulse
    # from numpy.fft import rfft,ifft

    t=linspace(0,T,round(T/dt))
    X=rfft(gausspulse((t-0.25*T),sc,bw=BW))
    
    s=linspace(0.,1/(2*dt),len(X))
    s[0]=1e-6
    
    RT=TransmissionReflection(s,rho,c,L)

    Y=X*RT
    
    s[0]=0.
    
    Y[:,0] = Y[:,1]
    
    y=ifft(2*Y,axis=1,n=2*Y.shape[1]-1)
        
    x=ifft(2*X,n=2*len(X)-1)
    
    t=linspace(0,T,y.shape[1])
    
    return t,vstack((x,y)),s,vstack((X,Y)),RT
    
def PrimerH(s,R,T,c,alpha,d):
    
    # from numpy import exp, linspace, pi
    
    s = linspace(s[0],s[1],s[2])
    
    HH = exp(-1j*2*d*2*pi*s/c)*exp(-2*alpha*d*s**2)
    
    H = (R[0]+(T[0]*T[1]-R[0]*R[1])*R[2]*HH)/(1-R[1]*R[2]*HH)
    
    return s,H

    
def ReflectionSequence(rho,c,alpha,d,dt,eps=1e-6):
    
    # from numpy import exp, hstack, zeros, array
    
    Z = [rho[i]*c[i] for i in range(len(rho))]
    R = [(Z[i+1]-Z[i])/(Z[i+1]+Z[i]) for i in range(len(Z)-1)]
    T = [4*Z[i]*Z[i+1]/((Z[i]+Z[i+1])**2) for i in range(len(Z)-2)]
    Nt = [round(2*d[i]/(dt*c[i])) for i in range(len(d))]
   
    A = T[0]*exp(-2*(d[0]*alpha[0]+d[1]*alpha[1]))
    

    h=hstack((zeros(Nt[0]-1),R[0]*exp(-2*d[0]*alpha[0])))

    h=hstack((h,zeros(Nt[1]-1),A*R[1]))
        
    e = 1.
    n = 1
    BB = 0.
    
    while e>eps:

        B = T[1]*R[2]*exp(-2*n*d[2]*alpha[2])*(R[2]**(n-1))*((-R[1])**(n-1))

        h=hstack((h,zeros(Nt[2]-1),A*B))

        e = abs(B-BB)

        BB = B

        n+=1
        
    return h
  
def PulseDistortionFeatures(x,dt):
    
    # from spr import ACorrelate, EchoSeparate,moments
    # from numpy import linspace
    # from numpy.linalg import norm
    # from matplotlib.pylab import plot,show
    
    F=[]
    
    for xx in x:
        
        xe=EchoSeparate(xx,2,db=-20,ws=0.01)
        
        # xc1=ACorrelate(xe[:,0]/norm(xe[:,0]),xe[:,1]/norm(xe[:,1]))
  #
        xc2=ACorrelate(xe[:,0],xe[:,0])
        
        xc3=ACorrelate(xe[:,0],xe[:,1])
        
        # t=linspace(-dt*len(xc1)/2,dt*len(xc1)/2,len(xc1))
        
        t=linspace(0,dt*len(xc2),len(xc2))
        
        # plot(xe[:,0])
  #       plot(xe[:,1])
  #
  #       show()
        
        m0=moments(abs(xc2),t)
        
        # print(m0)
        
        m1=moments(abs(xc3),t)
        
        # print(m1)
        
        F.append([m1[0]-m0[0],m1[1]-m0[1],m1[2]-m0[2],m1[3]-m0[3],m1[4]-m0[4]])


    return F
    
def LayerH(x,dt,N=5,Nfreqs=11,asteel=0.01,csteel=5.9):

    # from spr import EchoSeparate,PeakLimits
    # from numpy import linspace,angle,unwrap,exp,pi,polyfit,hstack,array
    # from numpy.fft import rfft

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
    
def AmpDelay(x,dt,N=4,asteel=0.042,csteel=6.):
    
    # from spr import AmplitudeDelayPhase
    # from numpy import array,mean,exp
    from sklearn.preprocessing import scale
    
    F=[]
    
    for xx in x:
    
        A,T,phi=AmplitudeDelayPhase(xx-mean(xx),N,dt,db=-30)
        
        # F.append([A[0],T[0],phi[0],A[1],T[1],phi[1],A[2],T[2],phi[2],A[3],T[3],phi[3]])
        
        try:
            
            F.append([A[0],T[0],phi[0],A[1],T[1],phi[1],A[2]*exp(asteel*csteel*(T[-1]-T[-2]))*(T[-1]-T[-2]),T[-1]-2*T[-2]+T[-3],phi[2],A[3]*exp(2*asteel*csteel*(T[-1]-T[-2]))*2*(T[-1]-T[-2]),phi[3]])
            
            # F.append([A[1],T[1]-T[0],phi[1],(A[3]/A[2])*exp(asteel*csteel*(T[-1]-T[-2]))*(T[-1]-T[-2]),T[-1]-2*T[-2]+T[-3]])
            # F.append([A[1],T[1]-T[0],phi[1]])
                        
        
        except:
            pass
    
    return F
    
def SpectralFeatures(x,dt,srng=[0.5,12.]):
    
    # from spr import EchoSeparate, moments
    # from numpy.fft import rfft
    # from numpy import linspace,array
    # from matplotlib.pyplot import
    
    
    F=[]
    
    for xx in x:
        
        xe=EchoSeparate(xx,2,db=-30)
        
        # plot(xe)
                
        Xe=rfft(xe,axis=0)
        
        
        Nf=Xe.shape[0]
        
        Xe=abs(Xe)/Nf
   
        
        s=linspace(0.,1/(2*dt),Nf)
        
        # print(len(s))
                
        # sind=array([(s>=srng[0])&(s<=srng[1])]).flatten()
        
        # print(sind.shape)
  #
  #       print(len(sind))
        
        # s=s[sind]
    #
    #     Xe=Xe[sind,:]
 
        
        m0=moments(Xe[:,0],s)
        m1=moments(Xe[:,1],s)
        
        FF = [m1[i]-m0[i] for i in range(len(m0))]
        
        F.append(FF)
        
        
    return F
    
def CorrelationFeatures(x,Npts):

    # from spr import EchoSeparate
    # from numpy import correlate,zeros,hstack
    # from numpy.linalg import norm

    F=[]

    for xx in x:

        xe=EchoSeparate(xx,1,db=-30)
        
        xe=xe.flatten()
        
        A=norm(xe)
        
        xc=correlate(xx/A,xe/A,'full')
        
        xc=xc[abs(xc).argmax()::]
                
        if len(xc)>=Npts:
            
            xc=xc[0:Npts]/abs(xc[0:Npts]).max()
                    
        else:
            
            xc=hstack((xc/abs(xc).max(),zeros(Npts-len(xc))))
                
        
        F.append(xc)
    
    return F

def AdhesiveFilter(x,dt,d,c,alpha):

    from numpy import zeros
    from numpy.fft import fft, ifft

    im = abs(x).argmax()

    x = hstack((x[im::],x[0:im]))

    X = rfft(x)

    f = linspace(0.0001,1/(2*dt),floor(len(x)/2)+1)

    # H = exp(-2*d*alpha*f)*exp(-1j*4*pi*f*d*(1/c-20*alpha/pi**2))*exp(1j*4*f*alpha*d*log(2*pi*f)/pi)

    H = exp(-2*d*alpha*f)*exp(-1j*4*pi*f*d*(1/c-20*alpha/pi**2))*exp(1j*4*f*alpha*d*log(2*pi*f)/pi)


    Y = X*H

    y = ifft(2*Y,n=2*len(Y)-1)

    return f,x,y,X,Y


# def PrimerReflectionFit(H,f,d1,d2,c1,c2,alpha1,alpha2):

def PrimerReflectionFit(H,f,d2,c2,alpha2):


    # Z0 = 0.943*2.05
    # Z1 = 0.94*1.97

    # H1 = exp(-2*d1*alpha1*f)*exp(-1j*4*pi*f*d1/c1)  #*exp(1j*4*alpha1*d1*log(2*pi*f)/pi)
    # H2 = exp(-2*d2*alpha2*f)*exp(-1j*4*pi*f*d2/c2)   #*exp(1j*4*alpha2*d2*log(2*pi*f)/pi)

    H2 = exp(-2*d2*alpha2*f)
    HH2 = exp(1j*4*pi*f*d2/c2)
    HHH2 = exp(-1j*4*pi*f*d2/c2)*H2

    # H1 = exp(-2*d1*alpha1*f)*exp(-1j*4*pi*f*d1*(1/c1-20*alpha1/pi**2))*exp(1j*4*f*alpha1*d1*log(2*pi*f)/pi)
    # H2 = exp(-2*d2*alpha2*f)*exp(-1j*4*pi*f*d2*(1/c2-20*alpha2/pi**2))*exp(1j*4*f*alpha2*d2*log(2*pi*f)/pi)

    # H2 = exp(-1j*4*pi*f*d2*(1/c2)) 

    # A = 4*Z0*Z1/(Z1+Z0)**2

    # H1 = exp(1j*4*pi*c*f/d)
    # H2 = exp(-2*alpha*f*d)
    # H3 = exp()


    # A = 4*Z0*Z1/((Z1+Z0)*(Z1-Z0))


    # B = hstack((H1,H1*H2,H*H2))

    # B = hstack((ones((shape(H2))),H2,H*H2))

    B = hstack((HH2,H2,H*HHH2))

    # B = hstack(())


    Bt = conj(B.transpose())
    # Bt = B.transpose()

    # v = []
    # r = []

    # fbands = linspace(f[0],f[-1],Nfbands)

    # print(fbands)

    # print(shape(B))
    # print(shape(Bt))

    # for i in range(len(fbands)-1):

    #     ff = (f>=fbands[i])&(f<=fbands[i+1])

    #     vv = solve(dot(Bt[:,ff],B[ff,:]),dot(Bt[:,ff],Y[ff]))

    #     E = Y[ff]-dot(B[ff,:],vv)

    #     rr = 0.5*dot(conj(E.transpose()),E)

    #     v.append(vv)
    #     r.append(rr)

    # print(c2)
    # print(alpha2)
    print(d2)

    print(cond(dot(Bt,B)))

    v = solve(dot(Bt,B),dot(Bt,H))

    # v = dot(inv(dot(Bt,B)),dot(Bt,H))

    # print(v)

    E = H-dot(B,v)

    # r = real(0.5*dot(conj(E.transpose()),E)[0][0])

    r = 0.5*dot(conj(E.transpose()),E)

    # r = 0.5*dot(conj(E.transpose()),E)


    r = real(r[0,0])

    print(r)
    print(v[1]/v[0])

    # print(shape(r))

    return v,r


def PrimerFitResidual(x,*params):

    # v,r = PrimerReflectionFit(params[0],params[1],x[0],x[1],x[2],x[3],x[4],x[5])

    v,r = PrimerReflectionFit(params[0],params[1],x[0],x[1],x[2])


    # v,r = PrimerReflectionFit(params[],params[3],x[0],x[1],params[0],params[1],x[2],x[3])


    return r

def AdhesivePrimerH(f,v,p):
    

    d1 = p[0]
    d2 = p[1]
    c1 = p[2]
    c2 = p[3]

    alpha1 = p[4]
    alpha2 = p[5]

    H1 = exp(-2*d1*alpha1*f)*exp(-1j*4*pi*f*d1/c1)   
    H2 = exp(-2*d2*alpha2*f)*exp(-1j*4*pi*f*d2/c2)  


    # H = H1*(v[0]+v[1]*H2)/(1-v[2]*H2)

    H = (v[0]+v[1]*H2)/(1-v[2]*H2)
    
    return H


def AdhesiveFit(d,c,alpha,f,X1,X2,X3):


    B = hstack((X1*exp(-1j*4*f*pi*d/c)*exp(-2*alpha*f*d),X3))   #,ones((len(X2),1))))

    Bt = conj(B.transpose())

    v = solve(dot(Bt,B),dot(Bt,X2))

    E = X2-dot(B,v)

    r = 0.5*dot(conj(E.transpose()),E)

    r = real(r[0,0])

    return v,r

def AdhesiveFitResidual(x,*params):

    v,r = AdhesiveFit(x[0],x[1],x[2],params[0],params[1],params[2],params[3])

    return r

    
def AdhesivePrimerFeatures(x,gates,dt,d,c,alpha,frng,df=0.01):

    from numpy import zeros
    from scipy.optimize import minimize
    from scipy.signal import tukey 
    from matplotlib.pyplot import plot,ginput,show,close

    NFFT = FFTLengthPower2(round(1/(df*2*dt)))

    F = []

    for xx in x:

        xa = hilbert(xx[abs(xx).argmax()::])

        ind = array(range(len(xa)))

        plot(ind,real(xa))

        pts = ginput(6)

        close()

        im1 = abs(xa[pts[0][0]:pts[1][0]]).argmax() + pts[0][0]

        iw1 = max([im1-pts[0][0],pts[1][0]-im1])

        x1 = real(xa[im1-iw1:im1+iw1])

        x1 = x1*tukey(len(x1),alpha=0.01)

        x1 = x1-mean(x1)

        im2 = abs(xa[pts[2][0]:pts[3][0]]).argmax() + pts[2][0]

        iw2 = max([im2-pts[2][0],pts[3][0]-im2])

        x2 = real(xa[im2-iw2:im2+iw2])

        x2 = x2*tukey(len(x2),alpha=0.01)

        x2 = x2 - mean(x2)

        im3 = abs(xa[pts[4][0]:pts[5][0]]).argmax() + pts[4][0]

        iw3 = max([im3-pts[4][0],pts[5][0]-im3])

        x3 = real(xa[im3-iw3:im3+iw3])

        x3 = x3*tukey(len(x3),alpha=0.01)

        x3 = x3 - mean(x3)

        # plot(x1)
        # plot(x2)
        # plot(x3)


        x1 = hstack((x1[floor(len(x1)/2)+1::],zeros(NFFT-len(x1)),x1[0:floor(len(x1)/2)+1]))

        x2 = hstack((x2[floor(len(x2)/2)+1::],zeros(NFFT-len(x2)),x2[0:floor(len(x2)/2)+1]))

        x3 = hstack((x3[floor(len(x3)/2)+1::],zeros(NFFT-len(x3)),x3[0:floor(len(x3)/2)+1]))

        T = dt*(im2-im1)

        f = linspace(0.,1/(2*dt),NFFT/2+1)

        X1 = rfft(x1)*exp(1j*2*pi*f*T)

        X2 = rfft(x2)

        X3 = rfft(x3)

        ff = (f>=frng[0])&(f<=frng[1])

        X1 = X1[ff]
        X1 = X1.reshape((len(X1),1))
        # X1 = X1/norm(X1)
        X2 = X2[ff]
        X2 = X2.reshape((len(X2),1))
        X3 = X3[ff]
        X3 = X3.reshape((len(X3),1))
        # X3 = X3/norm(X3)
        f = f[ff]
        f = f.reshape((len(f),1))

        # X1 = X1/max(abs(X2))
        # X2 = X2/max(abs(X2))
        # X3 = X3/max(abs(X2))

        params = (f,X1,X2,X3)

        varranges = (slice(dt*(pts[2][0]-im1)*c[1][0]/2, dt*(im2-im1)*c[1][0]/2, 0.01), slice(c[1][0],c[1][1],0.1), slice(alpha[1][0],alpha[1][1],0.05))

        # X = rfft(x1)

        # # X = X[ff]

        # # X = X.reshape((len(X),1))




        # x2 = hstack((x2[abs(hilbert(x2)).argmax()::],zeros(NFFT-len(x2)),x2[0:abs(hilbert(x2)).argmax()]))







        # t = linspace(0,len(xxx)*dt,len(xxx))


        # x1 = xxx[(t>=gates[0][0])&(t<=gates[0][1])]

        # x2 = xxx[(t>=gates[1][0])&(t<=gates[1][1])]

        # x1a = abs(hilbert(x1))

        # i1m = x1a.argmax()

        # il1,ir1 = PeakLimits(x1a,i1m,db=-25)

        # i1 = max([i1m-il1,ir1-i1m])


        # x2a = abs(hilbert(x2))

        # i2m = x2a.argmax()

        # il2,ir2 = PeakLimits(x2a,i2m,db=-45)

        # i2 = max([i2m-il2,ir2-i2m])


        # x1 = x1[i1m-i1:i1m+i1]

        # w1 = tukey(len(x1),0.01)

        # x1 = x1*w1

        # x2 = x2[i2m-i2:i2m+i2]

        # w2 = tukey(len(x2),0.01)

        # x2 = x2*w2

        # x1 = hstack((x1[abs(hilbert(x1)).argmax()::],zeros(NFFT-len(x1)),x1[0:abs(hilbert(x1)).argmax()]))

        # x2 = hstack((x2[abs(hilbert(x2)).argmax()::],zeros(NFFT-len(x2)),x2[0:abs(hilbert(x2)).argmax()]))

        # plot(x1)
        # plot(x2)

        # show()





        # f = linspace(0.,1/(2*dt),NFFT/2+1)

        # df = f[1]-f[0]

        # T1 = gates[1][0]+dt*i2m-dt*i2-(gates[0][0]+dt*i1m)

        # T2 = gates[1][0]+dt*i2m-(gates[0][0]+dt*i1m)

        # ff = (f>=frng[0])&(f<=frng[1])

        # X = rfft(x1)

        # # X = X[ff]

        # # X = X.reshape((len(X),1))

        # Y = rfft(x2)  #*exp(-1j*T2*2*pi*f)

 

        # Y = rfft(x2)

        # print(A)

        # Y = rfft(x2,NFFT)*exp(1j*2*pi*f*dt*i1m)

        # Y = Y.reshape((len(Y),1))

        # H = A*Yn*conj(Xn)/(Xn*conj(Xn))

        # H = A*(Y/X)

        # H = (Y/X)*exp(alpha[0]*c[0]*(T2-T1)*f)

        # # phi0 = angle(H*exp(1j*T2*2*pi*f))

        # # phi0 = phi0[ff]

        # H = H[ff]



        # H = H.reshape((max(shape(H)),1))

        # # H = H/norm(H)


        # f = f[ff]

        # f = f.reshape((len(f),1))

        # param = (c[0],c[1],H,f)

        # x0 = array([d[0],d[1],alpha[0],alpha[1]])

        # x0 = array([d[0],d[1],c[0],c[1],alpha[0],alpha[1]])


        # x1 = array([d[0][0],d[1][0],c[0][0],c[1][0],alpha[0][0],alpha[1][0]])

        # x2 = array([d[0][1],d[1][1],c[0][1],c[1][1],alpha[0][1],alpha[1][1]])


        # param = (H,f)

        # d[0] = (c[0][0]*T1/2,c[0][1]*T2/2)

        # d[1] = (d[1][0],c[1][0]*(T2-T1)/2)

        # print(d[0])
        # print(d[1])

        # bnds = (d[0],d[1],c[0],c[1],alpha[0],alpha[1])

        # x0 = (mean(d[0]),mean(d[1]),mean(c[0]),mean(c[1]),mean(alpha[0]),mean(alpha[1]))


        # bnds = (d,c[1],alpha[1])

        # x0 = (d[1]*0.75,mean(c[1]),mean(alpha[1]))

        # varranges = (slice(d[0],d[1],0.01),slice(c[1][0],c[1][1],0.1),slice(alpha[1][0],alpha[1][1],0.01))


        # varranges = (slice(c[0][0]*T1/2,c[0][1]*T2/2,dt*c[0][0]/2),slice(d[0],d[1],dt*c[1][0]/2),slice(c[0][0],c[0][1],c[0][2]),slice(c[1][0],c[1][1],c[1][2]),slice(alpha[0][0],alpha[0][1],alpha[0][2]),slice(alpha[1][0],alpha[1][1],alpha[1][2]))

        # print(varranges[0])
        # print(varranges[1])

        res = optimize.brute(AdhesiveFitResidual,varranges,args=params,full_output=True,finish=False)

        # res = optimize.fmin(PrimerFitResidual,x0,args=param,full_output=True)

        # R = minimize(PrimerFitResidual,x0,args=param,method='SLSQP',bounds=bnds,options={'ftol':1e-8,'eps':1e-8})



        # res = optimize.fminbound(PrimerFitResidual,x1,x2,args=param,full_output=True)


        # v,r = PrimerReflectionFit(H,f,res[0][0],res[0][1],res[0][2],res[0][3],res[0][4],res[0][5],Nfbands)

        # v,r = PrimerReflectionFit(H,f,res[0][0],res[0][1],c[0],c[1],res[0][2],res[0][3])

        # v,r = PrimerReflectionFit(H,f,R.x[0],R.x[1],R.x[2],R.x[3],R.x[4],R.x[5])

        # v,r = PrimerReflectionFit(H,f,R.x[0],R.x[1],R.x[2])

        v,r = AdhesiveFit(res[0][0],res[0][1],res[0][2],f,X1,X2,X3)

        T = dt*(im2-im1) - 2*res[0][0]/res[0][1]

        X2m = X1*exp(-4*pi*1j*f*res[0][0]/res[0][1])*exp(-2*res[0][2]*res[0][0])*v[0]/2 + X3*v[1]/2 #+ ones((len(X2),1))*v[2]


        # F.append([R.x[0],R.x[]])

        F.append([T*2.9/2,res[0][0],res[0][1],res[0][2],res[1],v])


        # F.append([res[1],res[0][0],res[0][1],res[0][2],res[0][3],res[0][4],res[0][5],real(v[0]),imag(v[0]),real(v[1]),imag(v[1]),real(v[2]),imag(v[2])])

    return F,f,X2,X2m

        
def TimeFrequencyFeatures(x,dt,srng,alpha=0.25):
    
    from spr import DSTFT, PeakLimits, FFTLengthPower2
    from numpy import array, hstack, real, imag, linspace, floor, shape
    
    NFFT = FFTLengthPower2(round(1/(srng[2]*dt)))
        
    s = linspace(0.,1/(2*dt),NFFT/2+1)   
    
    sind = (s>=srng[0])&(s<=srng[1])
    
    
    H = []
    
    
    for xx in x:
        
        HH = DSTFT(xx[0],xx[1],alpha,NFFT)
                        
        HH = HH[:,sind]
                                
        H.append(list(hstack((real(HH).flatten(),imag(HH).flatten()))))
        
    return array(H)

def AnalyticSignalFeatures(x,dt,gates):

    from scipy.signal import tukey


    F = []

    for xx in x:

        xa = hilbert(xx.copy())

        xa = xa[abs(xa).argmax()::]/abs(xa).max()

        t = linspace(0,len(xa)*dt,len(xa))

        i1m = abs(xa[(t>=gates[0][0])&(t<=gates[0][1])]).argmax() + round(gates[0][0]/dt)

        il1,ir1 = PeakLimits(abs(xa),i1m,db=-20)

        i2m = abs(xa[(t>=gates[1][0])&(t<=gates[1][1])]).argmax() + round(gates[1][0]/dt)

        il2,ir2 = PeakLimits(abs(xa),i2m,db=-35)

        x1a = abs(xa[il1:ir1])
        x1a = x1a*tukey(len(x1a),1.)

        x2a = abs(xa[il2:ir2])
        x2a = x2a*tukey(len(x2a),0.1)

        x1m = moments(x1a,t[il1:ir1])

        x2m = moments(x2a,t[il2:ir2])

        # plot(x1a)
        # plot(x2a)

        # show()



        # x1 = xa[(t>=gates[0][0])&(t<=gates[0][1])]

        # x1m = moments(abs(x1),t[(t>=gates[0][0])&(t<=gates[0][1])])

        # # x2 = xa[(t>=gates[1][0])&(t<=gates[1][1])]

        # x2m = moments(abs(x2),t[(t>=gates[1][0])&(t<=gates[1][1])])


        # F.append([x2m[0]/x1m[0],(gates[1][0]+x2m[1])-(gates[0][0]+x1m[1]),x2m[2]-x1m[2],x2m[3]-x1m[3],x2m[4]-x1m[4]])

        F.append([(x2m[0]-x1m[0])/x1m[0],(x2m[1]-x1m[1])/x1m[1],(x2m[2]-x1m[2])/x1m[2],(x2m[3]-x1m[3])/x1m[3],(x2m[4]-x1m[4])/x1m[4]])

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
        self.SteelThickness = []
        
    def setConfiguration(self):
        """ Reads 'config.ini' for variables associated with directories
        """
        import configparser, os
        self.config = configparser.ConfigParser()
        config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),'config.ini')
        self.config.read_file(open(config_file))
        
    def ManualScan(self, Locations, samplingFrequency, Averages=512):
        
#        from Ultrasonic import GetSignal
#        from numpy.linalg import norm
        
        # This function takes scans at various locations along a pipe, 
        
        # Locations -  a list of lists defining the locations on the pipe in mm
        
        # ** Note: Make sure not to change the horizontal scale on the scope while scanning
        
        for l in Locations:
        
            input('Press any key to collect signal at point'+str(l)+'mm')
        
            t0,dt,x = GetSignal(navg=Averages, samplingFrequency = samplingFrequency)
            
            self.AddSignal(x/norm(x),l)
            
        self.SamplingPeriod = dt
        self.Locations = Locations
        
    def AutoScan(self, Locations, samplingFrequency, Averages=512):
        import pygclib
#        from Ultrasonic import GetSignal
#        from numpy.linalg import norm
        
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
        self.Locations = Locations
        
        
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


    def Export(self,Filename,Path=None):
        
        if Path == None:
            Path = self.config['DEFAULT']['pipe_c_scans_db']
        
        if Filename.split('.')[-1] == 'txt':
        
            from numpy import hstack,array,savetxt

            # Export Raw Data to a structured text file (comma delimited)
            data=hstack((array(self.Locations),array(self.Signals)))

            savetxt(Path+Filename,data,delimiter=',',header=str(self.PipeId)+','+str(self.BondStrength[0])+','+str(self.BondStrength[1])+','+str(self.SamplingPeriod))
            
        elif Filename.split('.')[-1] == 'p':
            
            from pickle import dump
            
            data={'PipeId':self.PipeId,'BondStrength':self.BondStrength,'Locations':self.Locations,'Signals':self.Signals,'SamplingPeriod':self.SamplingPeriod,'SteelThickness':self.SteelThickness}
            
            dump(data,open(Path+Filename,'wb'))
        
    def Load(self, File, Path=None):

        if Path==None:
            Path = self.config['DEFAULT']['pipe_c_scans_db']
        if File.split('.')[1] == 'txt':
            from numpy import loadtxt
            File = Path + File
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
            
            pipe = load(open(Path + File,'rb'))

            if type(pipe) is dict:
                
                self.PipeId = pipe['PipeId']
                self.BondStrength = pipe['BondStrength']
                self.Signals = pipe['Signals']
                self.Locations = pipe['Locations']
                self.SamplingPeriod = pipe['SamplingPeriod']
                self.SteelThickness = pipe['SteelThickness']


class PipeSet:
            
    def __init__(self,SteelThick=None,Path='/Users/jlesage/Dropbox/ShawCor/pipe_auto_scans/'):

        from os import listdir
        # self.setConfiguration()
        # if Path==None:
        #     Path = self.config['DEFAULT']['pipe_c_scans_db']


        files=[f for f in listdir(Path) if f.endswith('.p')]
        Pipes=[]

        if SteelThick is None:
            
            for f in files:
        
                p=Pipe()
                p.Load(f,Path=Path)  
                p.ZeroMean()              
                
                if (len(p.BondStrength)==2):
                    Pipes.append(p)
                    
                    
        elif type(SteelThick) is float:
            
            for f in files:
        
                p=Pipe()
                p.Load(f,Path=Path)
                p.ZeroMean()
            
                if (len(p.BondStrength)==2)&(p.SteelThickness==SteelThick):
                    Pipes.append(p)

        self.Pipes=Pipes

        self.Gates = [(0.55,1.75),(1.8,4.3)]

                
    def SplitSignals(self,Period,db=-30):
        
        from scipy.signal import hilbert
        from numpy import hstack,zeros
        from spr import PeakLimits
        
        
        for p in self.Pipes:
            
            x = p.Signals
            
            for i in range(len(x)):
                                
                xa = abs(hilbert(x[i]))
                
                il,ir = PeakLimits(xa,xa.argmax(),db)
                
                p.Signals[i] = [xd[il:ir+1],xd[ir::]]

    def DownSample(self,Period):

        from scipy.signal import decimate

        for p in self.Pipes:

            if Period!=p.SamplingPeriod:
            
                q = round(Period/p.SamplingPeriod)
                p.SamplingPeriod = Period
                x = p.Signals

                for i in range(len(x)):
                
                    p.Signals[i] = decimate(x[i],q)
    
    # def setConfiguration(self):
    #     """ Reads 'config.ini' for variables associated with directories
    #     """
    #     import configparser, os
    #     self.config = configparser.ConfigParser()
    #     config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),'config.ini')
    #     self.config.read_file(open(config_file))

    def ExtractFeatures(self,ScansPerPipe):

        from random import sample
        
        # Npts = round((3.+5*self.Pipes[0].SteelThickness/5.99)/self.Pipes[0].SamplingPeriod)

        for p in self.Pipes:

            # p.Features = LayerH(p.Signals,p.SamplingPeriod)
            
            ind=sample(range(len(p.Signals)),ScansPerPipe)
                        
            signals = [p.Signals[i] for i in ind]
            
            # p.Features = AmpDelay(signals,p.SamplingPeriod)
            
            # p.Features = SpectralFeatures(signals,p.SamplingPeriod)
            
            # p.Features = PulseDistortionFeatures(signals,p.SamplingPeriod)
            
            
            # p.Features = CorrelationFeatures(signals,Npts)
            
            # p.Features = TimeFrequencyFeatures(signals,p.SamplingPeriod,[3.,10.,0.5])
            
            # p.Features = ReflectionFeatures(signals,p.SamplingPeriod,Nreverbs=3,dbref=-30,alpha=0.1)

            p.Features = AnalyticSignalFeatures(signals,p.SamplingPeriod,self.Gates)
            
            

    def MakeTrainingSet(self,StrengthRanges,Scale='standard'):
        
        ''' StrengthRanges list of tuples defining the Bond Strength Ranges defining each class '''

        from sklearn import preprocessing
        from numpy import zeros
        
        m=len(self.Pipes[0].Features[0])
        
        X=zeros((1,m))
        y=zeros(1)
        
        for p in self.Pipes:

            # if ((p.PipeId in IdList[0])&(p.PipeId not in IdList[1]) or (IdList == None)):

            l=len(p.Features)
            bs=mean(p.BondStrength)
                            
                                        
            for i in range(len(StrengthRanges)):
        
                if StrengthRanges[i][0]<=bs<=StrengthRanges[i][1]:
                                            
                    X=vstack((X,p.Features))
                    y=hstack((y,i*ones(l)))
                            
        y=y[1::]
                    
        self.y=y.astype(int)
        
        X=X[1::,:]
        
        if Scale=='standard':

            ss = preprocessing.StandardScaler()
        
            self.FeatureScaling = ss.fit(X)

            X = ss.transform(X)

        elif Scale=='robust':

            ss = preprocessing.RobustScaler()
        
            self.FeatureScaling = ss.fit(X)

            X = ss.transform(X)
            
        self.X=X

    def FitRBFClassifier(self,C_range=logspace(-3,3,4),gamma_range=logspace(-3,3,4)):

        from sklearn.cross_validation import StratifiedShuffleSplit
        from sklearn.grid_search import GridSearchCV
        from sklearn import svm

        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(self.y, n_iter=10, test_size=0.1, random_state=42)
        grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
        grid.fit(self.X, self.y)

        self.RBFClassifier = svm.SVC(C=grid.best_params_['C'],gamma=grid.best_params_['gamma'])
        self.RBFClassifier.fit(self.X,self.y)
        self.RBFScore = self.RBFClassifier.score(self.X,self.y)
        self.MaxDistance = max(self.RBFClassifier.decision_function(self.X))
        self.MinDistance = min(self.RBFClassifier.decision_function(self.X))

        self.RBFClassifierCVScore = grid.best_score_

    def FitLinearClassifier(self,C_range=logspace(-3,3,4)):

        from sklearn.cross_validation import StratifiedShuffleSplit
        from sklearn.grid_search import GridSearchCV
        from sklearn import svm

    
        param_grid = dict(C=C_range)
        cv = StratifiedShuffleSplit(self.y, n_iter=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid=param_grid, cv=cv)
        grid.fit(self.X, self.y)

        self.LinearClassifier = svm.SVC(kernel='linear',C=grid.best_params_['C'])
        self.LinearClassifierCVScore = grid.best_score_


    def TestScan(self):

        while True:

            ui = input('Press (t) to test current location, Press (q) to quit: ')

            if ui == 't':

                t0,dt,x = GetSignal(64,100,dev=0)

                x = x-mean(x)
                x = x/norm(x)

                X = AnalyticSignalFeatures([x],dt,self.Gates)

                X = self.FeatureScaling.transform(array(X))

                X = X.reshape(1, -1)

                pred = self.RBFClassifier.predict(X)

                pred_strength = self.RBFClassifier.decision_function(X)

                if pred == 0:

                    print('Weak')
                    print(str(pred_strength[0]/self.MinDistance))

                elif pred == 1:

                    print('Strong')
                    print(str(pred_strength[0]/self.MaxDistance))


            else:

                break






