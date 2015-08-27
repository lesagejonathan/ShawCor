def TMatrix1d(w,rho,c,l):
    
    from numpy import array,exp
    
    k=w/c
    
    Z=rho*c
    
    e1=exp(1j*k*l)
    e2=exp(-1j*k*l)
    
    T=array([[(e1+e2)/2,1j*(e2-e1)/(2*Z*w)],[(Z*w*1j/2)*(e1-e2),(e1+e2)/2]])
    
    return T
    
    

