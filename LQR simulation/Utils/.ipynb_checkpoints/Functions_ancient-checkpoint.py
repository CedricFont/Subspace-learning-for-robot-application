import numpy as np
from numpy.linalg import inv, lstsq, pinv
from numpy import linalg as lg
from scipy import linalg as slg
from scipy.linalg import svd
import pbdlib as pbd
import matplotlib.pyplot as plt

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
            if U[0] > limit[1]: U[0] = limit[1]
            if U[0] < limit[0]: U[0] = limit[0]
            return U
        else:
            if i <= t_max: t_max = 0
            U = K*(r - X[:,i]) - Kd*(X[:,i] - X[:,i-1])/dt + Ki*(np.sum(r - X[:,i-t_max:i]))*dt
            if U[0] > limit[1]: U[0] = limit[1]
            if U[0] < limit[0]: U[0] = limit[0]
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
        
    ref, dX, dU = None, None, None
    
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
    def __init__(self, X, U, **kwargs):
        self.X = X
        self.U = U
        self.N = X.shape[1] # Total number of points
        if len(kwargs) == 1: self.N += 1
        self.nb_S = X.shape[0] # Number of states
        self.nb_U = U.shape[0] # Number of control inputs
        
    def HANKEL(self, horizon, delay_spacing=None):
        self.n_h = horizon # Number of points in one trajectory
        self.spacing = delay_spacing
        if delay_spacing is not None: s = delay_spacing
        else: s = 1
        # Augmenting the state ####################################################
#         self.H = np.empty(shape=[(self.nb_S + 1)*self.n_h,self.N-self.n_h*s]) # Include input
#         for i in range(self.n_h):
#             self.H[(self.nb_S+1)*i:(self.nb_S+1)*(i+1),:] = np.concatenate((self.X[:,s*i:self.N - self.n_h*s + s*i],
#                                                                             self.U[:,s*i:self.N - self.n_h*s + s*i]), axis=0) # Delaying states and input
        # My original way ##########################################################
#         for i in range(self.N-self.n_h):
#             self.H[:,i] = inline(self.X[:,i:i+self.n_h]).T
#             self.H[0::2,i] = self.X[0,i:i+self.n_h].T
#             self.H[1::2,i] = self.X[1,i:i+self.n_h].T
        # Using only 1 state for learning ##########################################
#         self.H = np.empty(shape=[self.n_h,self.N-self.n_h])
#         for i in range(self.N-self.n_h):
#             self.H[:,i] = self.X[0,i:i+self.n_h].T
        
#         self.H = np.empty(shape=[self.nb_S*(self.N-self.n_h),self.n_h])
#         self.H = np.empty(shape=[self.n_h,self.N-self.n_h])
#         self.X = np.sin(self.X)
        # Same way as in the paper #################################################
#         # Only putting the state ###################################################
        self.H = np.empty(shape=[self.nb_S*self.n_h,self.N-self.n_h*s])
        for i in range(self.n_h):
            self.H[self.nb_S*i:self.nb_S*(i+1),:] = self.X[:,s*i:self.N - self.n_h*s + s*i]
        # Extend the input #########################################################
#         self.Ue = np.empty(shape=[self.n_h,self.N-self.n_h*s])
#         for i in range(self.n_h):
#             self.Ue[i,:] = self.U[s*i:self.N - self.n_h*s + s*i]
        
        # Only putting the state ###################################################
#         n_col = (self.N - self.n_h)//s+1
#         self.H = np.empty(shape=[self.nb_S*self.n_h,n_col])
#         for i in range(self.n_h):
#             self.H[self.nb_S*i:self.nb_S*(i+1),:] = self.X[:,i:self.N - self.n_h  + i:s]
#         # Extend the input #########################################################
#         self.Ue = np.empty(shape=[self.n_h,n_col])
#         for i in range(self.n_h):
#             self.Ue[i,:] = self.U[i:self.N - self.n_h + i:s]
        # Putting both states and input ############################################
#         self.H = np.empty(shape=[(self.nb_S+1)*self.n_h,self.N-self.n_h*s])
#         for i in range(self.n_h):
#             self.H[self.nb_S*i:self.nb_S*(i+1),:] = self.X[:,s*i:self.N - self.n_h*s + s*i]
#             self.H[self.nb_S*(i+1),:] = self.U[s*i:self.N - self.n_h*s + s*i]
        # Adding the input #########################################################
#         ns = self.nb_S + 4
#         self.H = np.empty(shape=[ns*self.n_h,self.N-self.n_h])
#         U = self.U[:,np.newaxis]
#         for i in range(self.n_h):
#             self.H[ns*i:ns*(i+1),:] = np.concatenate((self.X[:,i:self.N - self.n_h + i],np.sin(self.X[:,i:self.N - self.n_h + i]),U[i:self.N - self.n_h + i,:].T),axis=0)
#             self.H[ns*i:ns*(i+1),:] = np.concatenate((self.X[:,i:self.N - self.n_h + i],np.square(self.X[:,i:self.N - self.n_h + i]),
#                                                      np.cos(self.X[:,i:self.N - self.n_h + i])),axis=0)
        ############################################################################
#         self.H = np.empty(shape=[self.n_h,self.nb_S*(self.N - self.n_h)])
#         for i in range(self.n_h):
#             self.H[i,0::2] = self.X[0,i:self.N - self.n_h + i]
#             self.H[i,1::2] = self.X[1,i:self.N - self.n_h + i]
            
    def SVD(self, tau):
        # Perform SVD ###########################################################
        self.tau = tau # Number of embedded delays desired
        self.u, self.s, self.vh = svd(self.H, full_matrices=False)
        self.sigma = self.s
        self.v = self.vh.T
        # Restrict to desired subspace ##########################################
        self.u, self.s, self.v = self.u[:,:self.tau], self.s[:self.tau], self.v[:,:self.tau]
        self.Y = self.v.T
        self.C = self.u[0:2,:]@np.diag(self.s) # Mapping between subspace and original space
#         self.Cu = self.u[2,:]@np.diag(self.s) # Mapping between subspace and input
#         self.Cu = self.Cu[:,np.newaxis].T
        # Construct projection matrix into U ####################################
#         self.ut = self.u.T
#         self.P= self.ut@inv(self.ut.T@self.ut)@self.ut.T
#         self.Ue = self.Ue[:self.tau,:]
#         self.Up = self.P@self.Ue
        #########################################################################
#         self.ut = self.u.T
#         self.P= self.ut@inv(self.ut.T@self.ut)@self.ut.T
#         self.Ur = np.empty(shape=[self.tau,self.N - self.n_h*self.spacing])
#         for i in range(self.tau):
#             self.Ur[i,:] = self.U[self.spacing*i:self.N - self.n_h*self.spacing + self.spacing*i]
#         self.U_proj = self.P@self.Ur
#         self.Cu = self.U[0:self.N - self.n_h*self.spacing]@pinv(self.U_proj)
#         self.Cu = self.Cu[:,np.newaxis].T
        #########################################################################
#         self.Cu1 = self.Cu[:,np.newaxis].T
#         unit_vector = np.zeros([1,self.tau])
#         unit_vector[0,0] = 1
#         self.Cu = unit_vector@inv(self.P)
#         u_in, s_in, v_in = svd(self.Ue)
#         self.Cu = u_in[0,:self.tau]@np.diag(s_in[:self.tau])
#         self.Cu = self.Cu[:,np.newaxis].T
        # Project extended input subspace #######################################
#         u_u, s_u, v_uh = svd(self.Ue)
#         u_u = u_u[:,:self.tau]
#         s_u = s_u[:self.tau]
#         self.Cu = u_u[0,:]@np.diag(s_u)
        
#         self.Ue = v_uh.T[:,:self.tau].T
        
        #########################################################################
#         self.Cu = self.u[2,:]@np.diag(self.s)
#         self.Cu = self.Cu[:,np.newaxis].T
        
        
    def LS(self, p, rcond=None):
        Y_cut = self.Y[:,:self.Y.shape[1]-1]
        self.YU = np.concatenate((Y_cut,self.U[:Y_cut.shape[1],np.newaxis].T), axis=0)
#         self.YU = np.concatenate((Y_cut,pinv(self.Cu)@self.U[:Y_cut.shape[1],np.newaxis].T), axis=0)
#         self.Ue = self.u[0,:,np.newaxis]@self.U[:,np.newaxis].T
#         self.YU = np.concatenate((Y_cut,self.U_proj[:,:Y_cut.shape[1]]), axis=0)
#         self.YU = np.concatenate((Y_cut,pinv(self.Cu)@self.U[:,:Y_cut.shape[1]]), axis=0) # Mapping the input
#         u, s, vt = svd(self.YU)
#         u, s, vt = u[:,:p], s[:p], vt[:p,:]
        Y = self.Y[:,1:self.Y.shape[1]]
        AB, self.res, _, _ = lstsq(self.YU.T,Y.T,rcond)
        AB = AB.T
        self.LS_residuals(AB)
        self.A, self.B = AB[:,:self.tau], AB[:,self.tau:AB.shape[1]]
        
    def Simulate(self, X0, U_testing=None):
        if U_testing is None: U = self.U
        else: U = U_testing
#         U = pinv(self.Cu)@(U[:,np.newaxis].T)
#         U = (pinv(self.Cu)@U[:,np.newaxis].T).T
#         U = (self.Cu.T@U[:,np.newaxis].T).T
        Y0 = pinv(self.C)@X0
        N = np.eye(self.tau) - pinv(self.C) @ self.C # Nullspace projection operator
        Y0 = Y0 + N @ (self.Y[:,0] - Y0) # Corresponding position in subspace

        self.Y_traj = np.empty(shape=[self.tau,self.N])
        self.Y_traj[:,0] = Y0
        for i in range(self.N-1):
            self.Y_traj[:,i+1] = self.A@self.Y_traj[:,i] + self.B[:,np.newaxis].T*U[i]
#             self.Y_traj[:,i+1] = self.A@self.Y_traj[:,i] + self.B@U[:,i]
        self.X_traj = self.C@self.Y_traj
        
    def TrajError(self,X):
        self.traj_error = np.sqrt(np.sum(np.square(X - self.X_traj),axis=1))
        
    def LS_residuals(self, AB):
        self.residuals = (AB@self.YU - self.Y[:,1:self.Y.shape[1]])@(AB@self.YU - self.Y[:,1:self.Y.shape[1]]).T
        
    def ConstructLQR(self, *args, x_std, u_std, dt, ref):
        self.u_std, self.x_std = u_std, x_std
        
        if len(args) == 2: # Via-points
            reference = args[0]
            custom_precision = args[1]
            N = reference.shape[1]
        else:
            N = self.N
            reference = np.zeros([N,self.nb_S])
            reference[:,0] = ref 
            
        # Build LQR problem instance
        self.LQR = pbd.LQR(self.A, self.B, nb_dim=self.tau, dt=dt, horizon=len(ref))

        self.LQR.z = (pinv(self.C)@reference.T).T

        # Trajectory timing (1 via-point = 1 time-step, should be as long as the horizon)
        seq_tracking = 1*[0]

        for i in range(1,N):
            seq_tracking += (1)*[i]

        self.LQR.seq_xi = seq_tracking

        # Control precision 
        self.LQR.gmm_u = u_std

        # Tracking precision
        x_std = 1e6 # Importance of tracking the position
        Q_tracking = np.empty(shape=[N,self.tau,self.tau])

        for i in range(N):
            if len(args) == 2:
                cost = custom_precision[:,i]
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
        
    def LQR_cost(self, X, U, ref, horizon=None):
        if horizon is not None: N = horizon
        else: N = self.N
            
        Qu = np.diag(np.ones(N-1)*self.u_std)
        Q_tracking_modified = self.LQR.Q[:,0:self.nb_S-1,0:self.nb_S-1]
        
        cost_X = 0
        for i in range(N):
            cost_X += (X[0,i] - ref[i]).T*Q_tracking_modified[i,0,0]*(X[0,i] - ref[i])
        
        self.J = U.T@Qu@U + cost_X
        self.xQx, self.uRu = cost_X, U.T@Qu@U
        return self.J
    
    def RMSE(self, X_pred, X_true):
        delta_X_norm = lg.norm(X_true - X_pred, ord=2, axis=0)
        X_true_norm = lg.norm(X_true, ord=2, axis=0)
        return 100*delta_X_norm.sum()/X_true_norm.sum()
    
class SLFC:
    """
    Subspace learning for control
    """
    def __init__(self, X, U):
        self.X = X
        self.U = U
        self.N = X.shape[1] # Total number of points
        self.nb_S = X.shape[0] # Number of states
        self.nb_U = 1 # Number of control inputs
        
    def delayEmbeddings(self, nx, nu, d=1):
        self.X_lift = np.empty(shape=[self.nb_S*nx + self.nb_U*nu - 1 + 1,
                                      self.N - max(nx,nu)*d])
#         for i in range(nx):
#             self.X_lift[self.nb_S*i:self.nb_S*(i+1),:] = self.X[:,
#                                                                d*i:self.N - max(nx,nu)*d + d*i]
#         for i in range(nu):
#             self.X_lift[nx*self.nb_S + i,:] = self.U[d*i:self.N - max(nx,nu)*d + d*i]

        for i in range(nx):
            self.X_lift[self.nb_S*i:self.nb_S*(i+1),:] = self.X[:,
                                                               d*i:self.N - max(nx,nu)*d + d*i]
        for i in range(1,nu):
            self.X_lift[nx*self.nb_S + i,:] = self.U[d*i:self.N - max(nx,nu)*d + d*i]
            
        self.X_lift[nx*self.nb_S + nu*self.nb_U - 1,:] = slg.norm(np.concatenate((self.X[:,:self.N - max(nx,nu)*d],
                                                  self.U[:self.N - max(nx,nu)*d,np.newaxis].T),axis=0))
        
#         self.X_lift[nx*self.nb_S + nu*self.nb_U ,:] = np.ones(shape=[1,self.N - max(nx,nu)*d])
            
        # Defining the inputs of the regression problem
        self.X_lift_plus = self.X_lift[:,1:]
        self.X_lift = self.X_lift[:,:self.X_lift.shape[1]-1]
        
    def EDMD(self):
        """
        Extended dynamics mode decomposition
        """
        self.lift_dim = self.X_lift.shape[0]
        XU = np.concatenate((self.X_lift,self.U[:self.X_lift.shape[1],np.newaxis].T),axis=0)
        AB = self.X_lift_plus@slg.pinv2(XU)
        self.A = AB[:,:AB.shape[1]-1]
        self.B = AB[:,AB.shape[1]-1]
        self.B = self.B[:,np.newaxis]
        self.C = np.zeros(shape=[self.nb_S,self.lift_dim])
        self.C[:,:self.nb_S] = np.eye(self.nb_S)
        
        self.loss = np.sum(np.square((self.X_lift_plus - AB@XU).T@(self.X_lift_plus - AB@XU)))
        
    def Simulate(self, X, U):
        N = X.shape[1]
        X0 = X[:,0]
        Y0 = pinv(self.C)@X0
#         N_ = np.eye(self.lift_dim) - pinv(self.C) @ self.C # Nullspace projection operator
#         Y0 = Y0 + N_ @ (self.X_lift[:,0] - Y0) # Corresponding position in subspace
        #######################################
        self.X_sim_lift = np.empty(shape=[self.lift_dim,N])
        self.X_sim_lift[:,0] = Y0
        for i in range(N-1):
            self.X_sim_lift[:,i+1] = self.A@self.X_sim_lift[:,i] + self.B[:,np.newaxis].T*U[i]
        self.X_sim = self.C@self.X_sim_lift
              
    def trajLoss(self, X, Xp):
        """
        Sum of square errors of the trajectory w.r.t. reference trajectory
        """
        self.traj_loss = np.sqrt(np.sum(np.square((X - Xp).T@(X - Xp))))
        
    def ConstructLQR(self, x_std, u_std, dt, ref, *args):
        self.u_std, self.x_std = u_std, x_std
        if len(args) == 2:
            reference = args[0]
            custom_precision = args[1]
            N = reference.shape[1]
        else:
            N = self.N
            reference = np.zeros([N,self.nb_S])
            reference[:,0] = ref 
            
        # Build LQR problem instance
        self.LQR = pbd.LQR(self.A, self.B, nb_dim=self.lift_dim, dt=dt, horizon=len(ref))

        self.LQR.z = (pinv(self.C)@reference.T).T

        # Trajectory timing (1 via-point = 1 time-step, should be as long as the horizon)
        seq_tracking = 1*[0]

        for i in range(1,N):
            seq_tracking += (1)*[i]

        self.LQR.seq_xi = seq_tracking

        # Control precision 
        self.LQR.gmm_u = u_std

        # Tracking precision
        x_std = 1e6 # Importance of tracking the position
        Q_tracking = np.empty(shape=[N,self.lift_dim,self.lift_dim])

        for i in range(N):
            if len(args) == 2:
                cost = custom_precision[:,i]
            else:
                cost = np.array([x_std,0])
            Q_tracking[i,:,:] = self.C.T@np.diag(cost)@self.C # Put zero velocity precision

        self.LQR.Q = Q_tracking
        self.LQR.ricatti()
        self.K_lift = self.LQR.K
        self.K = self.K_lift@pinv(self.C)
        
    def LQR_simulate(self, X0):
        N = np.eye(self.lift_dim) - pinv(self.C)@self.C # Nullspace projection operator
        X0_lift = np.zeros((1,self.lift_dim))
        X0_lift[0,:] = pinv(self.C)@X0 + N@(self.X_lift[:,0] - pinv(self.C)@X0)
        ys, self.LQR_U = self.LQR.make_rollout(X0_lift)
        ys_mean = np.mean(ys, axis=0)
        xs = self.C@ys_mean.T # Map back to original space
        xs = xs.T
        self.LQR_X = xs

    def LQR_cost(self, X, U, ref, *args):
        if len(args) == 1:
            N = args[0]
        else:
            N = self.N

        Qu = np.diag(np.ones(N-1)*self.u_std)
        Q_tracking_modified = self.LQR.Q[:,0:self.nb_S-1,0:self.nb_S-1]

        cost_X = 0
        for i in range(N):
            cost_X += (X[0,i] - ref[i]).T*Q_tracking_modified[i,0,0]*(X[0,i] - ref[i])

        self.J = U.T@Qu@U + cost_X
        self.xQx, self.uRu = cost_X, U.T@Qu@U
        return self.J
    
    def RMSE(self, X_pred, X_true):
        delta_X_norm = lg.norm(X_true - X_pred, ord=2, axis=0)
        X_true_norm = lg.norm(X_true, ord=2, axis=0)
        return 100*delta_X_norm.sum()/X_true_norm.sum()
            
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
        
def plot_robot(xs, color='k', xlim=None,ylim=None, **kwargs):

	l = plt.plot(xs[0,:], xs[1,:], marker='o', color=color, lw=10, mfc='w', solid_capstyle='round',
			 **kwargs)

	plt.axes().set_aspect('equal')

	if xlim is not None: plt.xlim(xlim)
	if ylim is not None: plt.ylim(ylim)

	return l
