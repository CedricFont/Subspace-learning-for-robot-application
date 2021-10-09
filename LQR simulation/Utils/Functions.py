import numpy as np
from numpy.linalg import inv, svd, lstsq

def RK4(func, X0, u, t, type=None):
    """
    Runge and Kutta 4 integrator.
    """
    if isinstance(t, np.ndarray) == False:
        dt = t
        nt = 2
    else :
        dt = t[1] - t[0]
        nt = len(t)
    X  = np.empty([len(X0), nt])
    X[:,0] = X0
    
    if type == 'controller':
        for i in range(nt-1):
            k1 = func(X[:,i], u(X, i))
            k2 = func(X[:,i] + dt/2. * k1, u(X, i))
            k3 = func(X[:,i] + dt/2. * k2, u(X, i))
            k4 = func(X[:,i] + dt    * k3, u(X, i))
            X[:,i+1] = X[:,i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    elif type == 'controller-step-by-step':
        for i in range(nt-1):
            k1 = func(X0, u)
            k2 = func(X0 + dt/2. * k1, u)
            k3 = func(X0 + dt/2. * k2, u)
            k4 = func(X0 + dt    * k3, u)
            X = X0 + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    else:
        for i in range(nt-1):
            k1 = func(X[:,i], u[i])
            k2 = func(X[:,i] + dt/2. * k1, u[i])
            k3 = func(X[:,i] + dt/2. * k2, u[i])
            k4 = func(X[:,i] + dt    * k3, u[i])
            X[:,i+1] = X[:,i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        
    return X

def delayEmbeddedSimulation(A, B, X0, U, system_size=2):
    """
    Simulates a discrete-time system with embedded time-delays
    return : trajectory vector X
    """
    nb_delays = np.int(A.shape[1]/system_size - 1)
    N = len(U) # Simulation duration
    X = np.empty(shape=[system_size,N])
    X[:,0] = X0
    X_line = np.empty(shape=[np.int(system_size*(nb_delays+1))]) # Line vector for multiplying the extended A matrix
    
    for i in range(N-1):
        if i <= nb_delays:
            Ap = A[:,0:system_size*(i+1)]
            for j in range(i+1):
                X_line[j*system_size:(j+1)*system_size] = X[:,i-j].T
            Xp = X_line[0:system_size*(i+1)]
        else:
            # Don't need to update Ap, already completely extended
            for j in range(nb_delays+1):
                X_line[j*system_size:(j+1)*system_size] = X[:,i-j].T
            Xp = X_line
#         return (B*U[i]).T
        X[:,i+1] = Ap@Xp.T + (B*U[i]).T
        
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

def PD(X, i, r, K, Td, dt):
    if i == 0:
        return K*(r - X[:,i])
    else:
        return K*(r - X[:,i]) + K*Td*(-X[:,i] + X[:,i-1])/dt 
    
def PID(X, i, r, K, Td, dt):
    if i == 0:
        return K*(r - X[:,i])
    else:
        return K*(r - X[:,i]) + K*Td*(X[:,i] - X[:,i-1])/dt 
    
def squareReference(N, T, L, delay_precision=0, precision_percentage=0):
    """
    Creates a square signal for reference tracking.
    N -> nb of time-steps
    T -> period of oscillation (in number of time-steps)
    L -> levels of the square reference
    """
    count, delay_count = 0, delay_precision
    r = L[0]
    signal, precision = np.empty(shape=[N]), np.empty(shape=[N])
    for i in range(N):
        if i % T == 0:
            count +=1
            r = L[count%2] # Oscillates between 0 and 1
            delay_count = 0
            precision[i-np.int(delay_precision/10):i] = precision_percentage
        signal[i], delay_count = r, delay_count+1
        if delay_count < delay_precision: precision[i] = precision_percentage # Lower percentage of the precision
        else: precision[i] = 1 # 100% of the tracking precision
    return signal, precision

def sineReference(N, T, A):
    """
    Creates a sinusoidal reference for tracking
    N -> nb. of time-steps
    T -> period of the sin
    L-> low and high amplitudes
    """
    if A[0] != A[1]:
        offset = (A[0] + A[1])/2
        amplitude = (A[1] - A[0])/2
    return [amplitude*np.sin(2*np.pi*i/T) + offset for i in range(N)]     

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
    N = len(T)
    ref, ref_T = None, None
    
    def dynamics(self, theta, u):
        if isinstance(u, np.ndarray) == True:
            u = u[1]
        x1 = theta[0]
        x2 = theta[1]
        dx1dt = x2
        dx2dt = 2/(self.m*self.l**2)*(u - self.m*self.g*np.sin(x1))
        return np.array([dx1dt, dx2dt])
    
    def dynamicsDelayed(self, theta, u, tau):
        if isinstance(u, np.ndarray) == True:
            u = u[1]
        output = np.zeros(shape=[2*tau])
        x1 = theta[0]
        x2 = theta[1]
        dx1dt = x2
        dx2dt = 2/(self.m*self.l**2)*(u - self.m*self.g*np.sin(x1))
        output[0:2] = np.array((dx1dt,dx2dt))
        return output
    
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
        self.Traj = None # Vector for storing the whole trajectory
        
    def solve(self):
        Y = self.X[:,self.N - self.H:self.N].T 
#         Phi = np.empty([self.H,(self.nb_S + self.nb_U)*self.D])
        Phi = np.empty([self.H,self.nb_S*self.D+1])
        
        # TODO : implement for multi-input
        for i in range(self.D):
            Phi[:,i*self.nb_S:(i+1)*self.nb_S] = self.X[:,self.N-self.H-i-1:self.N-i-1].T # States
#             Phi[:,self.nb_S*self.D+i] = self.U[self.N-self.H-i-1:self.N-i-1]
        Phi[:,self.nb_S*self.D] = self.U[self.N-self.H-1:self.N-1]
            
#         U, S, V_T = svd(Phi, full_matrices=True)
#         X = V*np.multiply(U.T*Y,np.reciprocal(S))
        X, residuals, _, _ = lstsq(Phi,Y,rcond=None)
        X = X.T
        self.A, self.B = X[:,0:self.nb_S*self.D], X[:,self.D*self.nb_S:(self.nb_S + self.nb_U)*self.D]
        
    def computeTrajectory(self, X0, U, system_size=2):
        self.Traj = delayEmbeddedSimulation(self.A, self.B, X0, U, system_size=2)
             

