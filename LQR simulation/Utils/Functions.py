import numpy as np
from numpy.linalg import inv, svd, lstsq, pinv
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
    
# TODO : make the limit more general
def PID(X, i, r, K, Kd, Ki, dt, t_max, limit, type=None, integral=None):
    if type == 'std':
        error = r - X[:,i]
        integral = integral + error*dt
        U = K*error + Kd*(X[:,i] - X[:,i-1])/dt + Ki*integral
        return U, integral
    else:
        if i == 0:
            U = K*(r - X[:,i]) + Ki*(np.sum(r - X[:,0:i]))*dt
            if U[0] > limit: U[0] = limit
            return U
        else:
            if i <= t_max: t_max = 0
            U = K*(r - X[:,i]) - Kd*(X[:,i] - X[:,i-1])/dt + Ki*(np.sum(r - X[:,i-t_max:i]))*dt
            if U[0] > limit: U[0] = limit
            return U
    
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

def sineReference(N, dt, T, A, offset, varying=False, freq_vector=None, amp_vector=None):
    """
    Creates a sinusoidal reference for tracking
    N -> nb. of time-steps
    T -> period of the sin
    L -> low and high amplitudes
    varying -> creates time varying frequency and amplitude
    BEWARE : the nb of items in freq_vector & amp_vector must be a integer fraction of N
    """
    if varying:
        nbf, nba = N/len(freq_vector), N/len(amp_vector)
        freq, amp = -1, -1
        sine = np.empty(shape=[N])
        for i in range(N):
            if i%nbf == 0: freq += 1
            if i%nba == 0: amp += 1
            sine[i] = deg2rad(amp_vector[amp])*np.sin(2*np.pi*i/freq2period(freq_vector[freq],dt)) + deg2rad(offset)
    else:
        sine = [deg2rad(A*np.sin(2*np.pi*i/T)) + deg2rad(offset) for i in range(N)] 
    return sine 

def varSine(amp, freq, nb_periods, offset, dt):
    periods = np.reciprocal(freq)/dt # period of each sinusoid
    duration = np.multiply(periods,nb_periods) # nb of time steps for each sinusoid
    N = len(amp)
    sine = []
    
    for i in range(N):
        sine_storage = np.empty(shape=np.int(duration[i]))
        for j in range(np.int(duration[i])):
            sine_storage[j] = deg2rad(amp[i])*np.sin(2*np.pi/periods[i]*j) + deg2rad(offset[i])
        sine = np.hstack((sine,sine_storage))
        
    return sine

def freq2period(freq, dt):
    return 1/freq/dt

def inline(V):
    Vp = np.empty(shape=[V.shape[0]*V.shape[1]])
    n = V.shape[0]
    for i in range(V.shape[1]):
        Vp[i*n:(i+1)*n] = V[:,i]
    return Vp

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
#         dx2dt = 2/(self.m*self.l**2)*(u - self.m*self.g*np.sin(x1))
        dx2dt = -self.g/self.l*np.cos(x1) + u/(self.m*self.l**2)
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
        
class HAVOK:
    """
    Defines all necessary methods to identify a system using input and state measurement data
    """
    def __init__(self, X, U):
        self.X = X
        self.U = U
        self.N = X.shape[1] # Total number of points
        self.nb_S = X.shape[0] # Number of states
        self.nb_U = U.shape[0] # Number of control inputs
        
    def HANKEL(self, horizon):
#         self.n_h = horizon # Number of points in one trajectory
#         self.H = np.empty(shape=[self.n_h*self.nb_S,self.N-self.n_h])
#         for i in range(self.N-self.n_h):
#             self.H[:,i] = inline(self.X[:,i:i+self.n_h]).T
        self.n_h = horizon # Number of points in one trajectory
        self.H = np.empty(shape=[self.nb_S*self.n_h,self.N-self.n_h])
#         self.H = np.empty(shape=[self.n_h,self.N-self.n_h])
#         self.X = np.sin(self.X)
        for i in range(self.n_h):
            self.H[self.nb_S*i:self.nb_S*(i+1),:] = self.X[:,i:self.N - self.n_h + i]
            
    def SVD(self, tau):
        self.tau = tau # Number of embedded delays desired
        self.u, self.s, self.vh = svd(self.H, full_matrices=False)
        self.sigma = self.s
        self.v = self.vh.T
        # Restrict to desired subspace
        self.u, self.s, self.v = self.u[:,:self.tau], self.s[:self.tau], self.v[:,:self.tau]
        self.Y = self.v.T
        self.C = self.u[0:self.nb_S,:]@np.diag(self.s) # Mapping between subspace and original space
        
    def LS(self, p):
        Y_cut = self.Y[:,:self.Y.shape[1]-1]
        self.YU = np.concatenate((Y_cut,self.U[:Y_cut.shape[1],np.newaxis].T), axis=0)
#         u, s, vt = svd(self.YU)
#         u, s, vt = u[:,:p], s[:p], vt[:p,:]
        Y = self.Y[:,1:self.Y.shape[1]]
#         AB = Y@pinv(self.YU)
#         AB = Y@vt.T@inv(np.diag(s))@u.T
        AB, self.res, _, _ = lstsq(self.YU.T,Y.T,rcond=None)
        AB = AB.T
        self.LS_residuals(AB)
        self.A, self.B = AB[:,:self.tau], AB[:,self.tau:AB.shape[1]]
        
    def Simulate(self, X0, U_testing=None):
        if U_testing is None: U = self.U
        else: U = U_testing
        Y0 = pinv(self.C)@X0
        N = np.eye(self.tau) - pinv(self.C) @ self.C # Nullspace projection operator
        Y0 = Y0 + N @ (self.Y[:,0] - Y0) # Corresponding position in subspace

        self.Y_traj = np.empty(shape=[self.tau,self.N])
        self.Y_traj[:,0] = Y0
        for i in range(self.N-1):
            self.Y_traj[:,i+1] = self.A@self.Y_traj[:,i] + self.B[:,0]*U[i]
        self.X_traj = self.C@self.Y_traj
        
    def LS_residuals(self, AB):
        self.residuals = np.sum(np.square(self.C@(AB@self.YU - self.Y[:,1:self.Y.shape[1]])),axis=1)
        
    def ConstructLQR(self, x_std, u_std, dt, ref, custom_trajectory=None):
        self.u_std, self.x_std = u_std, x_std
        
        # Build LQR problem instance
        self.LQR = pbd.LQR(self.A, self.B, nb_dim=self.tau, dt=dt, horizon=self.N)

        # Reference tracking
        if custom_trajectory is not None:
            reference = ref
        else:
            reference = np.zeros([self.N,self.nb_S])
            reference[:,0] = ref 

        self.LQR.z = (pinv(self.C)@reference.T).T

        # Trajectory timing (1 via-point = 1 time-step, should be as long as the horizon)
        seq_tracking = 1*[0]

        for i in range(1,self.N):
            seq_tracking += (1)*[i]

        self.LQR.seq_xi = seq_tracking

        # Control precision 
        self.LQR.gmm_u = u_std

        # Tracking precision
        x_std = 1e6 # Importance of tracking the position
        Q_tracking = np.empty(shape=[self.N,self.tau,self.tau])

        for i in range(self.N):
            if custom_trajectory is not None:
                cost = custom_trajectory[:,i]
            else:
                cost = np.array([x_std,0])
            Q_tracking[i,:,:] = self.C.T@np.diag(cost)@self.C # Put zero velocity precision

        self.LQR.Q = Q_tracking
        self.LQR.ricatti()
        
    def LQR_simulate(self, X0):
        N = np.eye(self.tau) - pinv(self.C)@self.C # Nullspace projection operator
        Y0 = np.zeros((1,self.tau))
        Y0[0,:] = pinv(self.C)@X0 + N@(self.Y[:,0] - pinv(self.C)@X0)
        ys, self.LQR_U = self.LQR.make_rollout(Y0)
        ys_mean = np.mean(ys, axis=0)
        xs = self.C@ys_mean.T # Map back to original space
        xs = xs.T
        self.LQR_X = xs
        
    def LQR_cost(self, X, U, ref):
        Qu = np.diag(np.ones(self.N-1)*self.u_std)
        Q_tracking_modified = self.LQR.Q[:,0:self.nb_S-1,0:self.nb_S-1]
        
        cost_X = 0
        for i in range(self.N):
            cost_X += (X[0,i] - ref[i]).T*Q_tracking_modified[i,0,0]*(X[0,i] - ref[i])
        
        self.J = U.T@Qu@U + cost_X
        self.xQx, self.uRu = cost_X, U.T@Qu@U
        return self.J
              
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
