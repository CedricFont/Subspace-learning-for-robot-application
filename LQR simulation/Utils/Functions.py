import numpy as np
from numpy.linalg import inv, svd, lstsq
from scipy import linalg as slg
import pbdlib as pbd

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
    N = len(U)+1 # Simulation duration
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
    
def PID(X, i, r, K, Kd, Ki, dt, t_max):
    if i == 0:
        return K*(r - X[:,i]) + Ki*(np.sum(r - X[:,0:i]))*dt
    else:
        if i <= t_max: t_max = 0
        return K*(r - X[:,i]) + Kd*(X[:,i] - X[:,i-1])/dt + Ki*(np.sum(r - X[:,i-t_max:i]))*dt
    
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
            precision[i] = precision_percentage
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
    def __init__(self, mass, length, time, X0, dt=1e-2):
        self.m = mass
        self.l = length
        self.g = 9.81
        self.T = time
        self.N = len(self.T)
        self.X, self.X0 = np.empty(shape=[2,self.N]), X0
        self.X[:,0], self.X_ref2 = X0, np.empty(shape=[self.N,2])
        self.U, self.U2 = np.empty(self.N-1), np.empty(self.N-1)
        self.dt = dt
        
    ref, ref2 = None, None
    
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
        self.tau = tau # Maximum delay
        self.X = data # Observation matrix
        self.U = u
        self.nb_S = self.X.shape[0] # Number of states
        self.nb_U = nb_u # Number of control inputs
        self.N = self.X.shape[1] # Total number of data per state
        self.A, self.B = None, None # Linear dynamics matrices
        self.A_p, self.B_p = None, None # Linear dynamics matrices
        self.A_y, self.B_y = None, None # Linear dynamics matrices
        self.Traj, self.S, self.S_y = None, None, None # Vector for storing the whole trajectory and Hankel matrix sigma SVD matrix
        self.precision, self.optimal_truncation = None, None
        self.Y, self.Phi = None, None
        self.residuals = np.empty(shape=[2,1])
        
    def matricesConstruction(self):
        """
        self.H -> time horizon
        self.N -> total number of time-steps gathered from simulation
        """
        self.Phi = np.empty(shape=[self.H,self.nb_S*self.tau + 1]) # +1 for the input column
        self.Y = self.X[:,self.N - self.H:self.N]
        
        # Stack shifted state vectors from most recent to most ancient
        for delay in range(self.tau):
            self.Phi[:,delay*self.nb_S:(delay+1)*self.nb_S] = self.X[:,self.N - self.H - delay - 1:self.N - delay - 1].T
            
        self.Phi[:,self.tau*self.nb_S] = self.U[self.N - self.H - 1:self.N - 1] # Input
        
        self.Phi = self.Phi.T
    
    # SVD decompositions for low rank approximation
    def SVD(self, truncation_ratio=.9):
        self.matricesConstruction()
        U, self.S, V_T = svd(self.Phi, full_matrices=False)
        U_y, self.S_y, V_y = svd(self.Y, full_matrices=False)
        self.SVD_OptimalTruncationValue(truncation_ratio)
        
    def SVD_OptimalTruncationValue(self, truncation_ratio):
        SVD_sum = np.sum(self.S)
        for i in range(len(self.S)):
            if np.sum(self.S[:i]/SVD_sum) > truncation_ratio:
                self.optimal_truncation = i+1
                return 
        
    def truncate(self, rank, keep_matrices=True, double_SVD=False, rank2=0):
        self.matricesConstruction()
        if rank == 0: rank = self.Phi.shape[0]
        U, self.S, V_T = svd(self.Phi, full_matrices=False)
        U_zilda, S_zilda, V_zilda = U[:,0:rank], self.S[0:rank], V_T.T[:,0:rank]
        U_1, U_2 = U_zilda[0:self.nb_S*self.tau,:], U_zilda[self.nb_S*self.tau,:]
        self.A_p, self.B_p = self.Y@V_zilda@inv(np.diag(S_zilda))@U_1.T, self.Y@V_zilda@inv(np.diag(S_zilda))@U_2.T
        
        if keep_matrices: self.A, self.B = self.A_p, self.B_p
            
        if double_SVD:
            U_y, self.S_y, V_y = svd(self.Y, full_matrices=False)
            U_zilda_y, S_zilda_y, V_zilda_y = U_y[:,0:rank2], self.S_y[0:rank2], V_y.T[:,0:rank2]
            self.A_y, self.B_y = U_zilda_y.T@self.A_p@U_zilda_y, U_zilda_y.T@self.B_p
            if keep_matrices: self.A, self.B = self.A_y, self.B_y
        
    def solve(self, truncation_ratio=.95):
        self.matricesConstruction()
        self.X, _, _, _ = lstsq(self.Phi.T,self.Y.T,rcond=truncation_ratio)
        self.X = self.X.T
        self.A, self.B = self.X[:,0:self.nb_S*self.tau], self.X[:,self.tau*self.nb_S:]
        
        self.computeResiduals() # least square resiudas
        
    def computeTrajectory(self, X0, U, system_size=2):
        self.Traj = delayEmbeddedSimulation(self.A, self.B, X0, U, system_size=2)
        
    def computePrecision(self, original_trajectory):
        self.precision = np.sqrt(np.sum(np.square((original_trajectory-self.Traj)),axis=1))
        
    def computeResiduals(self):
        self.residuals[0] = np.sum(np.square(self.Y.T[:,0] - self.Phi.T@self.X.T[:,0])) # Position residuals
        self.residuals[1] = np.sum(np.square(self.Y.T[:,1] - self.Phi.T@self.X.T[:,1])) # Velocity residuals
             
class LQR_Transform:
    """
    Facilitates the transformation of a delay-embedded system into a LQR
    form
    """
    def __init__(self,dynamics_instance,object_instance):
        self.tau = dynamics_instance.tau
        self.nb_states = dynamics_instance.nb_S
        self.A_before, self.B_before = dynamics_instance.A, dynamics_instance.B
        self.A, self.B = np.zeros(shape=[2*self.tau,2*self.tau]), np.zeros(shape=[2*self.tau])
        self.dynamics, self.object = dynamics_instance, object_instance
        
    def LQR_Instance(self):
        self.A[0:2,:], self.A[2:2*self.tau,0:2*self.tau-2] = self.A_before, np.eye(2*self.tau-2)
        self.B[0:2] = self.B_before[0]
        self.B = self.B[:,np.newaxis]
        self.LQR = pbd.LQR(self.A, self.B, nb_dim=self.A.shape[0], dt=self.object.dt, horizon=self.object.N)
        
    def LQR_setParameters(self,u_std,x_std):
        self.LQR_trackingTrajectory()
        self.LQR_costDefinition(u_std,x_std)
        
    def LQR_trackingTrajectory(self):
        # Trajectory for tracking
        tracking_traj = np.zeros(shape=[self.object.N,2*self.tau]) # Vector for storing the trajectory
        tracking_traj[:,0] = self.object.ref
        self.LQR.z = tracking_traj

        # Trajectory timing (1 via-point = 1 time-step, should be as long as the horizon)
        seq_tracking = 1*[0]

        for i in range(1,self.object.N):
            seq_tracking += (1)*[i]

        self.LQR.seq_xi = seq_tracking
        
    def LQR_costDefinition(self,u_std,x_std):
        # Control precision 
        self.u_std = u_std
        self.LQR.gmm_u = u_std

        # Tracking precision
        Q_tracking = np.zeros(shape=[self.object.N,2*self.tau,2*self.tau])

        for i in range(self.object.N):
            Q_tracking[i,0:2,0:2] = np.diag([x_std,0]) # Put zero velocity precision

        self.LQR.Q = Q_tracking
        
    def LQR_rollout(self,X0):
        xs, us = self.LQR.make_rollout(X0)
        self.X = np.mean(xs, axis=0)
        self.us = np.mean(us, axis=0)
        self.xs_std  = np.std(xs, axis=0)
        
    def LQR_cost(self,X,U,ref):
        Qu = np.diag(np.ones(self.dynamics.N-1)*self.u_std)
        Q_tracking_modified = self.LQR.Q[:,0:self.nb_states-1,0:self.nb_states-1]
        
        cost_X = 0
        for i in range(self.dynamics.N):
            cost_X += (X[0,i] - ref[i]).T*Q_tracking_modified[i,0,0]*(X[0,i] - ref[i])
        
        self.J = U.T@Qu@U + cost_X
        return self.J
    
    def LQR_getK(self):
        self.K = np.array(self.LQR.K)[:,0,:]
