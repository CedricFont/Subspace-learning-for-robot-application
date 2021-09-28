import numpy as np

def RK4(func, X0, u, t):
    """
    Runge and Kutta 4 integrator.
    """
    if t.shape[0] == 1:
        dt = t
        nt = 2
    else :
        dt = t[1] - t[0]
        nt = len(t)
    X  = np.empty([len(X0), nt])
    X[:,0] = X0
    
    for i in range(nt-1):
        
        k1 = func(X[:,i], t[i], u[i])
        k2 = func(X[:,i] + dt/2. * k1, t[i] + dt/2., u[i])
        k3 = func(X[:,i] + dt/2. * k2, t[i] + dt/2., u[i])
        k4 = func(X[:,i] + dt    * k3, t[i] + dt, u[i])
        X[:,i+1] = X[:,i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        
    return X

def deg2rad(x):
    return x*np.pi/180

def rad2deg(x):
    return x*180/np.pi

def discreteSimulation(A, B, X0, u, t):
    """
    Simulates discrete-time dynamics given by the equation x(k+1) = Ax(k) + Bu(k)
    """
    dt = t[1] - t[0]
    nt = len(t)
    X  = np.zeros([len(X0), nt])
    X[:,0] = X0
    for i in range(nt-1):
        
        X[:,i+1] = A@X[:,i] + B*u[i]
        
    return X