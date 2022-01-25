import numpy as np
from numpy.linalg import inv, lstsq, pinv
from numpy import linalg as lg, random as rnd
from scipy import linalg as slg
from scipy.linalg import svd
import pbdlib as pbd
from pbdlib import LQR
from pbdlib.utils import get_canonical
import matplotlib.pyplot as plt
import torch
import tensorly as tl
tl.set_backend('pytorch')

def RK4(func, X0, u, t, type=None):
    """
    Runge Kutta 4 integrator.
    """
    if type == 'controller-step-by-step':
        dt = t
        if len(X0.shape) == 1:
            X = np.empty([X0.shape[0]])
        else:
            X = np.empty([X0.shape[0],X0.shape[1]])
    
    if type == 'controller':
        for i in range(nt-1):
            k1 = func(X[:,i], u(X, i))
            k2 = func(X[:,i] + dt/2. * k1, u(X, i))
            k3 = func(X[:,i] + dt/2. * k2, u(X, i))
            k4 = func(X[:,i] + dt    * k3, u(X, i))
            X[:,i+1] = X[:,i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    elif type == 'controller-step-by-step':
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
    def __init__(self, mass, length, time, X0=None, dt=1e-2, nu=.5, k=.5):
        self.m = mass
        self.l = length
        self.g = 9.81
        self.T = time
        self.nu, self.k = nu, k
        self.M = self.l**2*self.m
        self.N = len(self.T)
        self.X, self.X0 = np.empty(shape=[2,self.N]), X0
        self.X[:,0], self.X_ref2 = X0, np.empty(shape=[self.N,2])
        self.U, self.U2 = np.empty(self.N-1), np.empty(self.N-1)
        self.dt = dt
        self.x = None
        
    ref, dX, dU = None, None, None
    
    def Physics(self, x, tau):
        x1 = x[0] 
        x2 = x[1] 
        dx1dt = x2 
        dx2dt = 1/(self.M)* ( -self.m*self.g*self.l*np.cos(x1) - 
                                         self.k*x1 - 
                                         self.nu*x2 +
                                         tau
                            )
        return np.array([dx1dt,dx2dt])
    
    def CTC(self, qddot_desired, u, gravity=True, stiffness=True, friction=True, inertia=True):
        tau = self.M* ( qddot_desired  -
                        u ) * inertia + self.m*self.g*np.cos(self.x[0])*gravity + (
                        self.k*self.x[0]*stiffness + 
                        self.nu*self.x[1]*friction )
        return tau
    
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
    
class DoublePendulum:
    """
    - Instanciantes 2D pendulum object
    - Defines its dynamics and properties
    """
    def __init__(self, param):
        self.l1, self.l2 = param['l1'], param['l2']
        self.m1, self.m2 = param['m1'], param['m2']
        self.T, self.dt = param['time'], param['dt']
        self.N = len(self.T)
        self.g = 9.81
        self.x = None
        
    def Physics(self, x, tau=np.zeros([2])):
        q1, q1_dot = x[0,0], x[0,1]
        q2, q2_dot = x[1,0], x[1,1]
        # Mass matrix computation
        M_11 = self.m1*self.l1**2/3 + self.m2*self.l1**2 + self.m2*self.l2**2/3 + self.m2*self.l1*self.l2*np.cos(q2)
        M_12 = self.m2*self.l2**2/3 + self.m2*self.l1*self.l2/2*np.cos(q2)
        M_21 = M_12
        M_22 = self.m2*self.l2**2/3
        self.M = np.array([[M_11,M_12],[M_21,M_22]])
        # Stiffness matrix computation
        N_1 = - self.m2*self.l1*self.l2/2*np.sin(q2)*q2_dot**2 - self.m2*self.l1*self.l2*np.sin(q2)*q1_dot*q2_dot + self.m1*self.g*self.l1/2*np.cos(q1) + self.m2*self.g*self.l2/2*np.cos(q1-q2) + self.m2*self.g*self.l1*np.cos(q1)
        N_2 = self.m2*self.l1*self.l2/2*np.sin(q2)*q1_dot**2 + self.m2*self.g*self.l2/2*np.cos(q1-q2)
        self.Nq = np.array([[N_1],[N_2]])
        # Joint torque vector
        self.tau = tau
        # Dynamics formulation
        q_ddot = inv(self.M) @ (self.tau - self.Nq[:,0])
        
        return np.array([[q1_dot,q_ddot[0]],
                         [q2_dot,q_ddot[1]]])
    
    def CTC(self, q_ddot_desired, u):
        tau = self.M @ ( q_ddot_desired  + u ) + self.Nq[:,0]
        return tau
    
class Robot:
    """
    - Creates random trajectories for each joints while taking joint limits into account
    - Converts robot dynamics data into state-space form
    """
    def __init__(self, n_joints, nu, dt, N, robot=None):
        self.nb_joints = n_joints
        self.nb_S = n_joints*2
        self.nb_U = nu
        self.N = N
        self.dt = dt
        self.T = np.arange(0,N*dt,dt)
        self.time = np.arange(0,N)
        if robot is not None:
            self.robot = robot # PyBullet instance
        
    def hasCollisionHappened(self, traj, margin):
        table_height = .9 + .9*margin # To take bad control into account
        
        for i in range(traj.shape[1]):
            self.robot.default_q = traj[:,i]
            self.robot.reset_q()
            end_effector_pos = self.robot.x[2]
            
            if end_effector_pos < table_height: return True
        
        return False
        
    def drawTrajectories(self, nb_traj, q_limits, specs):
        self.nb_traj = nb_traj
        phi_low = specs['phi']['low']
        phi_high = specs['phi']['high']
        A_low = specs['A']['low']
        A_high = specs['A']['high']
        f_low = specs['f']['low']
        f_high = specs['f']['high']
        nb_parts = specs['nb_parts']
        margin, safety = specs['margin'], specs['safety']
        q_min, q_max = q_limits[:,0], q_limits[:,1]
        
        # LQR initialisation (double-integrator)
        A,B = get_canonical(self.nb_joints,nb_deriv=2,dt=self.dt)
        lqr = LQR(A, B, dt=self.dt, horizon=self.N)
        lqr.gmm_u = -6.
        temp = np.zeros(self.nb_S) 
        temp[0:self.nb_S//2] = 1e3 # Only precision on position
        lqr.Q = np.diag(temp)
        
        self.ref = np.zeros(shape=[nb_traj,self.nb_S,self.N])
        self.desired_ddq = np.zeros(shape=[nb_traj,self.nb_S//2,self.N-1])
        for j in range(self.nb_traj):
            x_train = 0
            collision_happened = True
                
#             while(collision_happened):
                
            for i in range(0, self.nb_S//2):

                # Draw random sinusoid according to specifications
                phi = rnd.uniform(low=phi_low, high=phi_high, size=nb_parts)
                f = rnd.uniform(low=f_low, high=f_high, size=nb_parts)*self.dt
                A = rnd.uniform(low=A_low, high=A_high, size=nb_parts)

                # Construct reference within joints bounds
                for nb in range(nb_parts):
                    x_train = x_train + A[nb]*np.sin(2*np.pi*f[nb]*self.time + phi[nb])

                if q_max[i] == -q_min[i]:
                    x_train = safety*x_train/max(x_train)*q_max[i]
                elif q_max[i] > abs(q_min[i]):
                    x_train += -(min(x_train) - safety*q_min[i]) # Shift upwards
                    if max(x_train) >= q_max[i]:
                        x_train = safety*x_train/max(x_train)*q_max[i]
                else: 
                    x_train -= max(x_train) - safety*q_max[i] # Shift downwards
                    if min(x_train) <= q_min[i]:
                        x_train = safety*x_train/abs(min(x_train))*abs(q_min[i])

                self.ref[j,i,:] = x_train  
                        
#                 collision_happened = self.isCollisionHappened(self.ref[j,:7,:], margin)
                    
            # Perform LQR of double integrator
            z = self.ref[j,:,:].T
            lqr.z = z
            lqr.ricatti()
            xs, us = lqr.get_seq(z[0,:])
            self.desired_ddq[j,:,:] = us.T
        
    def toStateSpace(self, q, dq, u):
        if len(q.shape) == 2:
            self.X = torch.empty([self.nb_traj,self.nb_S,q.shape[1]])
            
            self.X[:,0,:] = torch.tensor(q)
            self.X[:,1,:] = torch.tensor(dq)
            self.U = torch.tensor(u)
        else:
            self.X = torch.empty([self.nb_traj,self.nb_S,q.shape[2]])

            self.X[:,0:np.int(self.nb_S/2),:] = torch.tensor(q)
            self.X[:,np.int(self.nb_S/2):,:] = torch.tensor(dq)
            self.U = torch.tensor(u)

            self.dX = torch.diff(self.X)
            self.dU = torch.diff(self.U)
        
class HAVOK:
    """
    Hankel Alternative View Of Koopman
    Step 1 : learn SVD time-embedded coordinate system
    Step 2 : learn linear DMD model of the dynamics within this coordinate system
    Step 3 : plan LQR gains for controlling original non-linear system in a linear fashion
    """
    def __init__(self, X=None, U=None, **kwargs):
        if X is not None:
            self.X = X
            self.U = U
            self.N = X.shape[1] # Total number of points
            self.nb_S = X.shape[0] # Number of states
            self.kwargs = kwargs
            if 'learnOnDiff' in kwargs: self.N += 1 # Learn on differences
            if 'nb_U' in kwargs: self.nb_U = kwargs['nb_U']
            if 'nb_U_ex' in kwargs: self.nb_U_ex = kwargs['nb_U_ex']
            self.nb_U_c = self.nb_U - self.nb_U_ex
            self.to_numpy = False
        
    def HANKEL(self, horizon, delay_spacing=None):
        self.n_h = horizon # Number of points in one trajectory
        self.spacing = delay_spacing
        if delay_spacing is not None: s = delay_spacing
        else: s = 1
        #########################################################################
        self.H = torch.empty([self.nb_S*self.n_h,self.N-self.n_h*s])
        for i in range(self.n_h):
            self.H[self.nb_S*i:self.nb_S*(i+1),:] = self.X[:,s*i:self.N - self.n_h*s + s*i]
#         self.Un = np.empty(shape=[self.nb_U*self.n_h,self.N-self.n_h*s])
#         for i in range(self.n_h):
#             self.Un[self.nb_S*i,:] = np.zeros(shape=[1,self.N-self.n_h*s])
#             self.Un[self.nb_S*i + 1,:] = self.U[s*i:self.N - self.n_h*s + s*i]
#         for i in range(self.n_h):
#             self.Un[self.nb_U*i,:] = self.U[s*i:self.N - self.n_h*s + s*i]
            
    def SVD(self, tau):
        # Perform SVD ###########################################################
        self.tau = tau # Number of embedded delays desired
#         self.u, self.s, self.vh = torch.linalg.svd(self.H)
#         self.sigma = self.s
#         self.v = self.vh.T
        # Restrict to desired subspace ##########################################
#         self.u, self.s, self.v = self.u[:,:self.tau], self.s[:self.tau], self.v[:,:self.tau]
        self.u, self.s, self.v = tl.partial_svd(self.H, n_eigenvecs=self.tau)
        self.Y = self.v
        self.C = self.u[:self.nb_S,:] @ torch.diag(self.s) 
        self.pinvC = torch.linalg.pinv(self.C).to(torch.float)
        # Project u from R^n to R^r using SVD modes projection matrix ###########
#         self.S = self.u@np.diag(self.s) # Projection matrix from R^r to R^n
#         self.P = inv(self.S.T@self.S)@self.S.T
#         self.Ur = self.P@self.Un # Input matrix evolving within subspace
#         self.Cu = self.U[:self.N - self.n_h*self.spacing]@self.Ur.T@inv(self.Ur@self.Ur.T)
#         self.Cu = self.Cu[:,np.newaxis].T
        
    def LS(self, p, rcond=None):
        Y_cut = self.Y[:,:-1]
        
        if self.kwargs['mode'] == 'prediction' or self.kwargs['mode'] == 'ELQR form':
            if self.nb_U == 1:
                self.YU = torch.cat((Y_cut,self.U[:Y_cut.shape[1],None].T), axis=0)
            else:
                self.YU = torch.cat((Y_cut,
                                     self.U[:,:Y_cut.shape[1]]),axis=0)
            Y = self.Y[:,1:]
            AB, self.res, _, _ = torch.linalg.lstsq(self.YU.T.to(torch.float),Y.T,rcond)
            AB = AB.T
            if self.kwargs['mode'] == 'ELQR form':
                self.A, self.Bd = AB[:,:self.tau], AB[:,self.tau:self.tau+self.nb_S+self.nb_U_c]
                self.B = AB[:,self.tau+self.nb_S+self.nb_U_c:] # Input matrix
            else:
                self.A, self.B = AB[:,:self.tau], AB[:,self.tau:]
        elif self.kwargs['mode'] == 'not dynamical':
            Y = self.Y[:,1:]
            self.B, _, _, _ = torch.linalg.lstlq(self.U.T.to(torch.float),Y.T,rcond)
        
    def Simulate(self, X0, horizon, **kwargs):
        if self.to_numpy is True:
            self.toTorch()
        if 'U' in kwargs: U = kwargs['U']
        else: U = self.U
        if 'mode' in kwargs:
            if kwargs['mode'] == 'step-wise':
                self.kwargs['mode'] = 'step-wise'
        N = horizon
        self.Y_prediction = torch.empty([self.tau,N])
        
        if X0 is not None:
            X0 = X0.to(torch.float)
            U = U.to(torch.float)
            Y0 = self.pinvC @ X0
            N_ = torch.eye(self.tau) - self.pinvC @ self.C # Nullspace projection operator
            Y0 = Y0 + N_ @ (self.Y[:,0] - Y0).to(torch.float) # Corresponding position in subspace
            self.Y_prediction[:,0] = Y0
        
        if self.kwargs['mode'] == 'prediction' or self.kwargs['mode'] == 'ELQR form' or self.kwargs['mode'] == 'step-wise':
            for i in range(N-1):
                if self.kwargs['mode'] == 'prediction': # Standard model with a state vector and an input
                    if self.nb_U == 1:
                        self.Y_prediction[:,i+1] = self.A @ self.Y_prediction[:,i] + self.B[:,None].T*U[i]
                    else:
                        self.Y_prediction[:,i+1] = self.A @ self.Y_prediction[:,i] + self.B @ U[:,i]
                elif self.kwargs['mode'] == 'ELQR form': # Includes exogenous input Ud
                    self.Y_prediction[:,i+1] = self.A @ self.Y_prediction[:,i] + torch.cat((self.Bd[:],
                                                                                            self.B),axis=1) @ U[:,i]
                elif self.kwargs['mode'] == 'step-wise': # only predicts one step forward, hence should perform better
                    self.Y_prediction[:,i+1] = self.A @ self.pinvC @ kwargs['delta_X'][:,i] + torch.cat((self.Bd[:],
                                                                                                         self.B),axis=1) @ U[:,i]
        elif self.kwargs['mode'] == 'prediction_i': # Not recursive
            self.Y_prediction = self.A @ torch.cat((Y0[:,None].to(torch.float),
                                            kwargs['delta_U_i'].to(torch.float)),axis=0)+ self.B @ U
            self.X_prediction = self.C @ self.Y_prediction
            return self.X_traj
        elif self.kwargs['mode'] == 'not dynamical': # Does not updates recursively state-wise
            for i in range(N-1):
                self.Y_prediction[:,i+1] = self.B @ U[:,i]
                
        self.X_prediction = self.C @ self.Y_prediction # Project trajectory back to original space
        
    def TrajError(self,X):
        self.traj_error = torch.sqrt(np.sum(np.square(X - self.X_traj),axis=1))
        
    def LS_residuals(self, AB):
        self.residuals = (AB @ self.YU - self.Y[:,1:self.Y.shape[1]]) @ (AB @ self.YU - self.Y[:,1:self.Y.shape[1]]).T
        
    def toNumpy(self):
        if not self.to_numpy:
            if self.kwargs['mode'] == 'prediction':
                self.A, self.B, self.C, self.pinvC = self.A.numpy(), self.B.numpy(), self.C.numpy(), self.pinvC.numpy()
            else:
                self.A, self.B, self.Bd, self.C, self.pinvC = self.A.numpy(), self.B.numpy(), self.Bd.numpy(), self.C.numpy(), self.pinvC.numpy()
            self.to_numpy = True
    
    def toTorch(self):
        if self.to_numpy:
            if self.kwargs['mode'] == 'prediction':
                self.A, self.B, self.C, self.pinvC = torch.tensor(self.A), torch.tensor(self.B), torch.tensor(self.C), torch.tensor(self.pinvC)
            else:
                self.A, self.B, self.Bd, self.C, self.pinvC = torch.tensor(self.A), torch.tensor(self.B), torch.tensor(self.Bd), torch.tensor(self.C), torch.tensor(self.pinvC)
            self.to_numpy = False
        
    def ConstructLQR(self, x_std, u_std, dt, horizon, ref=None, **kwargs):
        self.u_std, self.x_std = u_std, x_std
        if self.to_numpy is False:
            self.toNumpy()
        
        if kwargs['mode'] == 'via-points': # Via-points
            reference = ref
            precision = kwargs["precision"]
            N = horizon
        elif kwargs['mode'] == 'ELQR':
            N = horizon-1 # Because of the ELQR formulation
            reference = np.zeros([N,self.nb_S]).T # Drive the residual dynamics to 0
        else:
            N = horizon
            reference = np.zeros([N,self.nb_S])
            if len(ref) == 1:
                reference[:,0] = ref 
            else:
                reference = ref
            
        # Build LQR problem instance
        self.LQR = pbd.LQR(self.A, self.B, nb_dim=self.tau, dt=dt, horizon=N)

        self.LQR.z = (self.pinvC@reference).T
#         self.LQR.z = np.zeros([N,self.tau + 1])

        # Trajectory timing (1 via-point = 1 time-step, should be as long as the horizon)
        seq_tracking = 1*[0]

        for i in range(1,N):
            seq_tracking += (1)*[i]

        self.LQR.seq_xi = seq_tracking

        # Control precision 
        self.LQR.gmm_u = u_std

        # Tracking precision
        Q_tracking = np.empty([N,self.tau,self.tau])

        for i in range(N):
            if kwargs['mode'] == 'via-points':
                cost = precision[:,i]
            else:
                cost = x_std
            Q_tracking[i,:,:] = self.C.T@np.diag(cost)@self.C # Put zero velocity precision

        self.LQR.Q = Q_tracking
        if kwargs['mode'] == 'ELQR':
            self.LQR.ricatti(mode='ELQR', u_d=kwargs['u_d'], Bd=kwargs['Bd'])
        else:
            self.LQR.ricatti()
        
    def LQR_simulate(self, X0, **kwargs):
        if self.to_numpy is False:
            self.toNumpy()
        N = np.eye(self.tau) - self.pinvC@self.C # Nullspace projection operator
        Y0 = np.zeros((1,self.tau))
        Y0[0,:] = self.pinvC@X0 + N@(self.Y[:,0].numpy() - self.pinvC@X0)
        if kwargs['mode'] == 'ELQR':
            ys, self.LQR_U = self.LQR.make_rollout_w_dist(Y0, u_d=kwargs['u_d'], Bd=kwargs['Bd'])
        else:
            ys, self.LQR_U = self.LQR.make_rollout(Y0)
        xs = self.C@ys[0,:,:].T # Map back to original space
        self.LQR_X = xs
        
        
    def LQR_cost(self, X, U, ref, horizon=None):
        if horizon is not None: N = horizon
        else: N = self.N
            
        Qu = torch.diag(torch.ones(N-1)*self.u_std)
        Q_tracking_modified = self.LQR.Q[:,0:self.nb_S-1,0:self.nb_S-1]
        
        cost_X = 0
        for i in range(N):
            cost_X += (X[0,i] - ref[i]).T*Q_tracking_modified[i,0,0]*(X[0,i] - ref[i])
        
        self.J = U.T@Qu@U + cost_X
        self.xQx, self.uRu = cost_X, U.T@Qu@U
        return self.J
    
    def RMSE(self, X_pred, X_true, **kwargs):
        if isinstance(X_pred, np.ndarray): X_pred = torch.tensor(X_pred)
        if isinstance(X_true, np.ndarray): X_true = torch.tensor(X_true)
        if 'regulation' in kwargs: 
            X_true_norm = torch.ones(1)
            X_true = torch.zeros([X_pred.shape[0],X_pred.shape[1]])
        delta_X_norm = torch.linalg.norm(X_true - X_pred, ord=2, dim=0)
        X_true_norm = torch.linalg.norm(X_true, ord=2, dim=0)
        return (100*delta_X_norm.sum()/X_pred.shape[1]).numpy()
    
    def toCuda(self, X):
        if torch.cuda.is_available():
            return X.to(torch.device('cuda'))
        else:
            return X
    
    def toCPU(self, X):
        return X.to(torch.device('cpu'))
    
class SLFC:
    """
    Subspace learning for control
    """
    def __init__(self, X, U, **kwargs):
        self.X = X
        self.U = U
        self.N = X.shape[1] # Total number of points
        self.kwargs = kwargs
        if 'learnOnDiff' in kwargs: self.N += 1 # Learn on differences
        self.nb_S = X.shape[0] # Number of states
        if len(self.U.shape) == 1: self.nb_U = 1
        else: self.nb_U = self.U.shape[0] # Number of control inputs
        
    def delayEmbeddings(self, nx, nu, d):
        self.X_lift = torch.empty([self.nb_S*nx + self.nb_U*nu - 1 + 1,
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
            self.X_lift[nx*self.nb_S + self.nb_U*i:nx*self.nb_S + self.nb_U*(i+1),:] = self.U[:,d*i:self.N - max(nx,nu)*d + d*i]
            
        # Observable : norm of the input vector
        self.X_lift[nx*self.nb_S + nu*self.nb_U - 1,:] = torch.norm(torch.cat((self.X[:,:self.N - max(nx,nu)*d],
                                                  self.U[:,:self.N - max(nx,nu)*d]),axis=0),p=2)
        
#         self.X_lift[nx*self.nb_S + nu*self.nb_U ,:] = np.ones(shape=[1,self.N - max(nx,nu)*d])
            
        # Defining the inputs of the regression problem
        self.X_lift_plus = self.X_lift[:,1:]
        self.X_lift = self.X_lift[:,:self.X_lift.shape[1]-1]
        
    def EDMD(self):
        """
        Extended dynamics mode decomposition
        """
        self.lift_dim = self.X_lift.shape[0]
        N = self.X_lift.shape[1]
        if 'mode' in self.kwargs:
            if self.kwargs['mode'] == 'prediction':
                XU = torch.cat((self.X_lift,
                                self.U[:,:N]), # Contains X^m, delta_U and U^m
                               axis=0).to(torch.float)
                AB = self.X_lift_plus@torch.linalg.pinv(XU)
                self.A = AB[:,:AB.shape[0]] # Autonomous system
                self.B = AB[:,AB.shape[0]:] # Forcing terms
        
        self.C = torch.zeros([self.nb_S,self.lift_dim])
        self.C[:,:self.nb_S] = torch.eye(self.nb_S)
        self.pinvC = torch.linalg.pinv(self.C)
        
    def Simulate(self, X0, U, **kwargs):
        N = self.N
        Y0 = self.pinvC@X0
        N_ = torch.eye(self.lift_dim) - self.pinvC @ self.C # Nullspace projection operator
        Y0 = Y0 + N_ @ (self.X_lift[:,0] - Y0) # Corresponding position in subspace
        #######################################
        self.X_sim_lift = torch.empty(size=[self.lift_dim,N])
        self.X_sim_lift[:,0] = Y0
        if 'mode' in self.kwargs:
            if self.kwargs['mode'] == 'prediction':
                for i in range(N-1):
                    self.X_sim_lift[:,i+1] = self.A@self.X_sim_lift[:,i] + self.B@U[:,i]
        else:
            for i in range(N-1):
                self.X_sim_lift[:,i+1] = self.A@self.X_sim_lift[:,i] + self.B[:,None].T*U[i]
        self.X_sim = self.C@self.X_sim_lift
        
    def ConstructLQR(self, x_std, u_std, dt, ref, **kwargs):
        self.A, self.B, self.C, self.pinvC = self.A.numpy(), self.B.numpy(), self.C.numpy(), self.pinvC.numpy()
        self.u_std, self.x_std = u_std, x_std
        
        if len(kwargs) == 1: # Via-points
            reference = ref
            precision = kwargs["precision"]
            N = ref.shape[0]
        else:
            N = self.N
            reference = np.zeros([N,self.nb_S])
            if len(ref) == 1:
                reference[:,0] = ref 
            else:
                reference = ref
            
        # Build LQR problem instance
        self.LQR = pbd.LQR(self.A, self.B, nb_dim=self.lift_dim, dt=dt, horizon=N)

        self.LQR.z = (self.pinvC@reference).T

        # Trajectory timing (1 via-point = 1 time-step, should be as long as the horizon)
        seq_tracking = 1*[0]

        for i in range(1,N):
            seq_tracking += (1)*[i]

        self.LQR.seq_xi = seq_tracking

        # Control precision 
        self.LQR.gmm_u = u_std

        # Tracking precision
        Q_tracking = np.empty(shape=[N,self.lift_dim,self.lift_dim])

        for i in range(N):
            if len(kwargs) == 2:
                cost = custom_precision[:,i]
            else:
                cost = x_std
            Q_tracking[i,:,:] = self.C.T@np.diag(cost)@self.C # Put zero velocity precision

        self.LQR.Q = Q_tracking
        self.LQR.ricatti()
        self.K_lift = self.LQR.K
        self.K = self.K_lift@self.pinvC
        
    def LQR_simulate(self, X0):
        N = np.eye(self.lift_dim) - self.pinvC@self.C # Nullspace projection operator
        X0_lift = np.zeros((1,self.lift_dim))
        X0_lift[0,:] = self.pinvC@X0 + N@(self.X_lift[:,0].numpy() - self.pinvC@X0)
        ys, self.LQR_U = self.LQR.make_rollout(X0_lift)
        xs = self.C@ys[0,:,:].T # Map back to original space
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
    
    def RMSE(self, X_pred, X_true, **kwargs):
        delta_X_norm = torch.linalg.norm(X_true - X_pred, ord=2, dim=0)
        X_true_norm = torch.linalg.norm(X_true, ord=2, dim=0)
        if 'regulation' in kwargs: X_true_norm = np.ones(1)
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
