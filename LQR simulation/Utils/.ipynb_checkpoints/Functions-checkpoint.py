import numpy as np
from numpy.linalg import inv

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

def pseudoInverse(X):
    return inv(X.T@X)@X.T

class SimplePendulum:
    """
    Defines a simple pendulum object
    """
    def __init__(self, mass, length):
        self.m = mass
        self.l = length
        self.g = 9.81
    
    X = None # Trajectory from a simulation
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
    
class DelayedLeastSquare:
    """
    Defines a delay embedded version of a least square problem
    for linear dynamics discovery
    """
    def __init__(self, data, tau, horizon, u, nb_u):
        self.H = horizon
        self.D = tau # Maximum delay
        self.X = data # Observation matrix
        self.U = u
        self.nb_S = self.X.shape[0] # Number of states
        self.nb_U = nb_u # Number of control inputs
        self.N = self.X.shape[1] # Total number of data per state
        self.A, self.B = None, None # Linear dynamics matrices
        
    def solve(self):
        Y = np.flip(self.X[:,self.N - self.H:self.N]).T 
        Phi = np.empty([self.H,(self.nb_S + self.nb_U)*self.D])
        
        # TODO : implement for multi-input
        for i in range(self.D):
            Phi[:,i*self.nb_S:(i+1)*self.nb_S] = np.flip(self.X[:,self.N-self.H-i-1:self.N-i-1],axis=1).T # States
            Phi[:,self.nb_S*self.D+i] = np.flip(self.U[self.N-self.H-i-1:self.N-i-1])
            
        X = (pseudoInverse(Phi)@Y).T
        self.A, self.B = X[:,0:self.nb_S], X[:,self.D*self.nb_S]
             

