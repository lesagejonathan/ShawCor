def ComputeK(s,l,c,rho,alpha,na=2.,sc=6.):
    """
    :s (numpy array): frequencies
    :l (list): layer thicknesses
    :c (list): speed of sounds
    :rho (list): densities
    :alpha (list): attenuations
    :na (float) : exponent on the attenuation frequency dependence
    :sc (float) : center frequency
    """

    from numpy import array,zeros,exp,pi,sqrt,sign
    
    K=zeros((len(l)+1,len(l)+1),dtype=complex)
    
    
    w=2*pi*s    
                
    for n in range(len(l)):
        
        
        # eta=c[n]*alpha[n]/(2*(pi*sc)**2)

        Alpha=alpha[n]*(s/sc)**na
 
        # k=(w/c[n])*1j+alpha[n]
        
        # k=(w/c[n])*sqrt(-1/(1+1j*eta*w))
        
        k=(w/c[n])*1j+Alpha
        
        M=(rho[n]*w**2)/k**2
        
        # M=rho[n]*(1+1j*w*eta)*c[n]**2
            
        K[n,n]=K[n,n]+(-(M*k*(exp(2*k*l[n]) + 1))/(exp(2*k*l[n]) - 1))
        K[n,n+1]=K[n,n+1]+(2*M*k*exp(k*l[n]))/(exp(2*k*l[n]) - 1)
        K[n+1,n]=K[n+1,n]+(2*M*k*exp(k*l[n]))/(exp(2*k*l[n]) - 1)
        K[n+1,n+1]=K[n+1,n+1]+ (-(M*k*(exp(2*k*l[n]) + 1))/(exp(2*k*l[n]) - 1))


    return K
    
def ComputeU(s,l,c,rho,alpha,F):
    
    from numpy import array, zeros, abs
    from numpy.linalg import solve
    
    
    
    U=[]
    
    
    
    for i in range(len(s)):
                
        K=ComputeK(s[i],l,c,rho,alpha)
        
        U.append(1j*s[i]*solve(K,F[:,i]))
        
        # U.append(solve(ComputeK(s[i],l,c,rho,eta),F[:,i]))
        
    # U=array(U).reshape((F.shape[0],len(s)))
    U=array(U)
        
    return U
    
def ComputeF(s,l,c,rho,alpha,U):
    
    from numpy import array, zeros, abs, dot
    
    
    F=[]
    
    
    
    for i in range(len(s)):
                
        K=ComputeK(s[i],l,c,rho,alpha)
        
        F.append(dot(K,U[:,i]))
        
        # U.append(solve(ComputeK(s[i],l,c,rho,eta),F[:,i]))
        
    # U=array(U).reshape((F.shape[0],len(s)))
    F=array(F)
        
    return F
        
        
def Computeu(dt,l,c,rho,alpha,f):
    
    from numpy.fft import fft, ifft,rfft,irfft
    from numpy import linspace,hstack,real,zeros
    
    
    
    # F=fft(f)
    
    F=rfft(f)
    
    
    s=linspace(1e-6,1/(2*dt),F.shape[1])#/2+1)
    
    # if F.shape[1]%2 is 1:
  #
  #       s=linspace(1e-15,1/(2*dt),F.shape[1]/2+1)
  #
  #       s=hstack((s,-s[::-1]))[0:-1]
  #
  #   elif F.shape[1]%2 is 0:
  #
  #       s=linspace(1e-10,1/(2*dt),F.shape[1]/2)
  #
  #       s=hstack((s,-s[::-1]))
        
        
    # Fm=abs(F[0,:])
 #    maxFm=Fm.max()
 #
 #    ampfrac=1e-9*maxFm
 #
 #    # UU=ComputeU(s[Fm>=ampfrac],l,c,rho,alpha,F)
 #
 #    UU=ComputeU(s[s>=1.],l,c,rho,alpha,F)
    


    # U=zeros((len(Fm),UU.shape[1]),dtype=complex)
 #
 #
 #
 #    U[s>=1.,:]=UU
 #
 #    U[0,:]=U[1,:]

    U=ComputeU(s,l,c,rho,alpha,F)
    
    
    
    # U[:,0]=abs(F[0,:])*U[:,0]
    
    # U[0,:]=0.+0.*1j
    
    u=ifft(U,n=2*len(s)-1,axis=0)
    
    t=linspace(0.,len(u)*dt,len(u))
    
    return t,u
    
    
def Computef(dt,l,c,rho,alpha,u):
    
    from numpy.fft import fft, ifft,rfft,irfft
    from numpy import linspace,hstack,real,zeros
    
    
    
    # F=fft(f)
    
    U=rfft(u)
    
    
    s=linspace(1e-6,1/(2*dt),U.shape[1])#/2+1)
    
    # if F.shape[1]%2 is 1:
  #
  #       s=linspace(1e-15,1/(2*dt),F.shape[1]/2+1)
  #
  #       s=hstack((s,-s[::-1]))[0:-1]
  #
  #   elif F.shape[1]%2 is 0:
  #
  #       s=linspace(1e-10,1/(2*dt),F.shape[1]/2)
  #
  #       s=hstack((s,-s[::-1]))
        
        
    Um=abs(U[0,:])
    maxUm=Um.max()
    
    ampfrac=1e-9*maxUm 
    
    print(s[Um>=ampfrac].shape)
    
    FF=ComputeF(s[Um>=ampfrac],l,c,rho,alpha,U)
    
    print(FF.shape)
    
    F=zeros((len(Um),FF.shape[1]),dtype=complex)
    
   
    
    F[Um>=ampfrac,:]=FF
    
    F[0,:]=F[1,:]
    
    # U[0,:]=0.+0.*1j
    
    f=irfft(F,axis=0)
    
    t=linspace(0.,len(f)*dt,len(f))
    
    return t,f
    
def GaussianPulse(sc,T, BW, dof=[1., 0., 0., 0., 0., 0.], amp=1.):
    """Compute forcing function from t0 to t0+T at dof with frequency sc
    sc : center frequency
    t0:    
    T: total time span
    dof: where to apply loads
    """
    from numpy import zeros,arange,exp,cos,pi,sin
    
    dt=1/(2*(sc+BW/2+10.))
    
    t0=2./BW
    t=arange(0.,T,dt)
    
    g=amp*exp(-0.5*((2.67*BW)**2)*(t-t0)**2)*cos(2*pi*sc*(t-t0))+0*1j
    
    f=zeros((len(dof),len(g)),dtype=complex)
    
    for i in range(len(dof)):
        
        f[i,:]=dof[i]*g
        
        
    return dt,f
    
    
def GumblePulse(sc,t0,T,BW,dof=[1.,0.,0.,0.,0.,0.],amp=1.):
    
    from numpy import zeros,arange,exp,cos,pi,sin,sqrt
    
    dt=1/(2*(sc+BW/2+50.))
    
    t=arange(0.,T,dt)
    
    B=sqrt(6)/(2.67*BW*pi)
    z=(t-t0)/B
    
    g=amp*(1/B)*exp(-(z+exp(-z)))*cos(2*pi*sc*(t-t0))+0*1j
    
    f=zeros((len(dof),len(g)),dtype=complex)
    
    for i in range(len(dof)):
        
        f[i,:]=dof[i]*g
        
        
    return dt,f

