# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:40:18 2021

@author: cedri

HAVOK analysis of a 1D pendulum controlled. Aim : perform LQR upward stabilisation
using the learned linear model.
"""

# Import stuff
import numpy as np
from numpy import random as rnd
from numpy.linalg import pinv
from numpy import linalg as lg
from scipy import linalg as slg
from importlib import reload
import seaborn as snb
import matplotlib.pyplot as plt
import Utils.Functions 
from Utils.Functions import *
import pbdlib as pbd
from scipy.linalg import hankel
plt.style.use("default")
plt.rcParams['font.size'] = 13
plt.rcParams['axes.linewidth'] = 1.5
np.set_printoptions(precision=5, suppress=False)

"""
Synthetize training and testing data by controlling the pendulum along some 
trajectories
"""

# Pendulum instance
dt = 1e-2
time = np.arange(0,20,dt)

# Controller design
K, Ki, Kd, t_max = 10, 50, 0, 12

# Reference design
freq1, freq2 = .25, .35
T_ref1, T_ref2 = 1/freq1/dt, 1/freq2/dt
amplitude1, amplitude2 = 40, 50 # deg
ref1 = sineReference(np.int(30/dt), dt, T_ref1, amplitude1, 50)
ref1 = varSineRef([80,110,200,50],[2,2,2,2],[1/2,1/4,1/6,1/8],dt)
time1 = np.arange(0,len(ref1)*dt,dt)
pend1 = SimplePendulum(mass=1, length=1, time=time1, X0=np.array([deg2rad(0),0]), dt=dt)
pend2 = SimplePendulum(mass=1, length=1, time=time1, X0=np.array([deg2rad(0),0]), dt=dt)
pend1.ref = ref1
# pend1.ref = sineReference(pend1.N, pend1.dt, [], [], offset=60, varying=True, freq_vector=[.4], amp_vector=[30])
# pend1.ref, precision = squareReference(pend1.N, pend1.T_ref, [deg2rad(0),deg2rad(90)], np.int(pend1.T_ref/10), 0)
pend2.ref = sineReference(pend2.N, pend1.dt, T_ref2, amplitude2, 120)
# pend2.ref, precision = squareReference(pend1.N, pend1.T_ref, [deg2rad(50),deg2rad(130)], np.int(pend1.T_ref/10), 0)

# Simulation

integral = 0
for i in range(pend1.N-1):
    pend1.U[i] = PID(pend1.X, i, pend1.ref[i], 15, 5, 20, pend1.dt, 30, limit=30, integral=integral)[0] 
    pend1.U[i] = rnd.normal(0,2,1)
    pend1.X[:,i+1] = RK4(pend1.dynamics, pend1.X[:,i], pend1.U[i], pend1.dt, type='controller-step-by-step')
    pend2.U[i] = PID(pend2.X, i, pend2.ref[i], 70, 1, 28, pend2.dt, 20, limit=20)[0]
    pend2.U[i] = rnd.normal(0,1,1)
    pend2.X[:,i+1] = RK4(pend2.dynamics, pend2.X[:,i], pend2.U[i], pend2.dt, type='controller-step-by-step')
    
# Plots
plt.rcParams['font.size'] = 20
fig, ax = plt.subplots(3, 2, 
                       constrained_layout = True, figsize=(30,10))

ax[0,0].plot(pend1.T, rad2deg(pend1.X[0,:]), label=r"$\theta (t)$")
# ax[0,0].plot(pend1.T, rad2deg(np.array(pend1.ref)[:pend1.N]), label=r"$reference \ r_1$")
ax[0,1].plot(pend2.T, rad2deg(pend2.X[0,:]), label=r"$\theta (t)$")
# ax[0,1].plot(pend2.T, rad2deg(np.array(pend2.ref)), label=r"$reference \ r_2$")
ax[0,0].set_ylabel('Position [deg]')
ax[0,0].grid(), ax[0,1].grid()
ax[0,0].legend(), ax[0,1].legend()
ax[0,0].set_title('Reference for training'), ax[0,1].set_title('Reference for testing')

ax[1,0].plot(pend1.T, rad2deg(pend1.X[1,:]), label=r"$\dot{\theta} (t)$")
ax[1,1].plot(pend2.T, rad2deg(pend2.X[1,:]), label=r"$\dot{\theta} (t)$")
ax[1,0].set_ylabel('Angular speed [deg/s]')
ax[1,0].grid(), ax[1,1].grid()
ax[1,0].legend(), ax[1,1].legend()

ax[2,0].plot(pend1.T[:pend1.N-1], pend1.U, label='u(t)')
ax[2,1].plot(pend2.T[:pend1.N-1], pend2.U, label='u(t)')
ax[2,0].set_xlabel('Time [s]'), ax[2,0].set_ylabel('Torque [Nm]'), ax[2,1].set_xlabel('Time [s]')
ax[2,0].grid(), ax[2,1].grid()
ax[2,0].legend(), ax[2,1].legend()

fig.savefig('Pend_id_sine.svg',format='svg',dpi=600)

"""
Learn the linear dynamics using the HAVOK method along with DMDc
"""

# In the following, pend1 --> training of A & B
#                   pend2 --> testing (trajectory simulation)

pend = pend1
horizon = np.int(len(pend.T)*.6)
model1 = HAVOK(pend.X, pend1.U)
X0 = [deg2rad(60),0]

nb_delay = [8,10,15,20,30]
nb_delay = [5,10,15,30]
nb_plots = len(nb_delay)
res = np.empty(shape=[2,nb_plots])

plt.rcParams['font.size'] = 15
fig, ax = plt.subplots(2, 2, constrained_layout = True, figsize=(18,7))

pend = pend2

ax[0,0].plot(pend.T, rad2deg(pend.X[0,:]), label=r"$\theta (t)$")
ax[0,0].set_xlabel('Time [s]'), ax[0,0].set_ylabel('Position [deg]')
ax[0,0].grid()

ax[0,1].plot(pend.T, rad2deg(pend.X[1,:]), label=r"$\dot{\theta} (t)$")
ax[0,1].set_xlabel('Time [s]'), ax[0,1].set_ylabel('Angular speed [deg]')
ax[0,1].grid()

ax[1,0].plot(pend.T[:len(pend.T)-1], pend.U, label='Input')
ax[1,0].set_xlabel('Time [s]'), ax[1,0].set_ylabel('Torque [Nm]')
ax[1,0].grid(), ax[1,0].legend(), ax[1,0].set_title('Original control input')

for i in range(nb_plots):
    tau = nb_delay[i]
    model1.HANKEL(horizon)
    model1.SVD(tau)
    model1.LS()
    model1.Simulate(pend.X0, U_testing=pend.U)
    
    # Test mapping A and B to original space and then simulate
    # A_prime = model1.U_map.T@model1.A@model1.U_map # DMDc
    # A_bar = model1.C@model1.A@model1.C.T # Input mapping
    # print(A_prime)
    # print(A_bar)
    
    ax[0,0].plot(pend.T[:len(pend.T)], rad2deg(model1.X_traj[0,:]), label='r = '+str(tau))
    ax[0,0].set_xlabel('Time [s]'), ax[0,0].set_ylabel('Position [deg]')
    ax[0,0].grid()
    ax[0,0].legend(bbox_to_anchor=(.9, 1.3),ncol=nb_plots,fontsize=12)

    ax[0,1].plot(pend.T[:len(pend.T)], rad2deg(model1.X_traj[1,:]), label='r = '+str(tau))
    ax[0,1].set_xlabel('Time [s]'), ax[0,1].set_ylabel('Angular speed [deg/s]')
    ax[0,1].grid()
    ax[0,1].legend(bbox_to_anchor=(.9, 1.35),ncol=nb_plots,fontsize=12)
    
    res[:,i] = model1.residuals
    
ax[1,1].plot(nb_delay, np.log(res[0,:]),'-o', markersize=10, color='b', label='Position')
ax[1,1].plot(nb_delay, np.log(res[1,:]),'-o', markersize=10, color='r', label='Velocity')
ax[1,1].set_xlabel(r'Subspace dimension $r$'), ax[1,1].set_ylabel('Log(precision)'), ax[1,1].legend()
ax[1,1].grid(), ax[1,1].set_title('Least square log residuals')
ax[0,0].grid(), ax[0,1].grid()

fig.savefig('Images/Pend_DMD_sine.svg',format='svg',dpi=600)

"""
LQR control rollout
"""

tau = 15
# Training
pend=pend2
model1 = HAVOK(pend.X, pend.U)
model1.HANKEL(horizon)
model1.SVD(tau)
model1.LS()
# Testing
pend=pend2
model1.Simulate(pend.X0, U_testing=pend.U)
model1.ConstructLQR(x_std=1e6, u_std=2., dt=pend.dt, ref=pend.ref)
model1.LQR_simulate(pend.X0)

# Plot
plt.rcParams['font.size'] = 20
fig,ax = plt.subplots(2, 2, constrained_layout = True, figsize=(18,7))

ax[0,0].plot(pend.T, rad2deg(np.array(pend.ref)), label="Reference $r_2$")
ax[0,0].plot(pend.T, rad2deg(model1.LQR_X[:,0]), label=r"$\theta_{LQR} (t)$")
ax[0,0].set_xlabel('Time [s]'), ax[0,0].set_ylabel('Position [deg]')
ax[0,0].grid()
ax[0,0].legend()

ax[0,1].plot(pend.T, rad2deg(model1.LQR_X[:,1]), label=r"$\dot{\theta}_{LQR} (t)$")
ax[0,1].set_xlabel('Time [s]'), ax[0,1].set_ylabel('Angular speed [deg/s]')
ax[0,1].grid()
ax[0,1].legend()

ax[1,0].plot(pend.T[:model1.N-1], model1.LQR_U[0,:,0], label='$u_{LQR}(t)$')
ax[1,0].set_xlabel('Time [s]'), ax[1,0].set_ylabel('Torque [Nm]')
ax[1,0].grid()
ax[1,0].legend()

ax[1,1].set_title('$K_{LQR}$')
figA = snb.heatmap(np.array(model1.LQR._K)[:,0,:].T,cmap='Reds')
ax[1,1].set_xlabel('Horizon'), ax[1,1].set_ylabel(r'Subspace coordinate $r$')

