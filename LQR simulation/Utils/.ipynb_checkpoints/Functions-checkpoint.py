import numpy as np

def RK4(func, X0, u, t):
    """
    Runge and Kutta 4 integrator.
    """
    if len(t) == 1:
        dt = t
        nt = 2
    else :
        dt = t[1] - t[0]
        nt = len(t)
    X  = np.empty([len(X0), nt])
    X[:,0] = X0
    
    for i in range(nt-1):
        
        k1 = func(X[:,i], u[i])
        k2 = func(X[:,i] + dt/2. * k1, u[i])
        k3 = func(X[:,i] + dt/2. * k2, u[i])
        k4 = func(X[:,i] + dt    * k3, u[i])
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

class SimplePendulum:
    """
    Defines a simple pendulum object
    """
    def __init__(self, mass, length):
        self.m = mass
        self.l = length
        self.g = 9.81
    
    X = [] # Trajectory from a simulation
    X0 = [] # IC for simulation
    U = [] # Input for a simulation (torque)
    T = [] # Time vector for a simulation
    dt = 1e-2 # Time-step for simulation
    
    def dynamics(self, theta, u):
        x1 = theta[0]
        x2 = theta[1]
        dx1dt = x2
        dx2dt = 2/(self.m*self.l**2)*(u - self.m*self.g*np.sin(x1))
        return np.array([dx1dt, dx2dt])

