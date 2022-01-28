# Subspace learning for robot application

In this repo, you will find many notebooks providing insights about what was tested/explored, as well as the library developed to code the algorithm and its different functionalities.

## Repo layout

Everything lays within the ```LQR_simulation``` folder. 

## Notebooks

The number before each notebook name states at which week of the semester a certain notebook was written. Their content is expressed by their names, but will also be explained just below :

* ```Pendulum simulation``` : first notebook with linear identification using no time-delays
* ```PendulumDelayEmbeddings``` : same as the first one, but with delays this time. However, this is not HAVOK, but rather something in the same vein as Section 2.4 from the report.
* ```KoopmanDelayEmbeddingsAssesment``` : same as the preceeding notebook, but with different types of references, although still not random
* ```5_Havok_pendulum1``` : First try about the standard HAVOK algorithm with constant sinusoid reference (no randomness)
* ```6_HAVOK_TSVD``` : Different methods for finding the optimal delay. Truncated SVD from a paper from _Gavish & Donoho_ was first used, and then a mixed-integer convex program for SVD truncation was developed.
* ```6_Testing_1``` : First try at using time-varying sinusoids, but still no randomness. A script was developed to vary their amplitude and frequency, and the main goal of this notebook consists in finding a way to test the performance of the algorithm. Before that, only training performance was considered. Also, pendulum swing ups were tested.
* ```7_HAVOK_DMDc``` : Script for performing DMD with control along with plots
* ```8_Delay_embedded_simulation``` : Tests for implementing a smoothly varying random sinusoid as a sum of random sinusoids
* ```8_Better_identification``` : First time using HAVOK on the random smooth sinusoid as well as development of a hyper-parameter routine using some sort of grid-search methods.
* ```9_Adding_noise``` : Testing on the algorithm robustness to noise. Was not continued though since SVD inherently provides robut^stness against Gaussian distributed noise.
* ```9_ Midterm``` : notebook used for generating the results showed at the midterm presentation. Within it, the complete grid search alorithm as well as LQR in subspace are present.
* ```9_hyperopt``` : After the midterm, following the advice to maybe switch to Bayesian optimisation for hyperparameter tuning, this notebook allowed to test the use of the hyperopt bayesian optimisation software library. This notebook is thus essentially the same as ```9_Midterm```, but it runs with the hyperopt routine and allows to have better results.
* ```10_SLFC``` : _Subspace Learning For Control_ Direct implementation of what is presented in Section 2.4 of the report, combined with what can be found in [20] or [21] (see refs. of the report). The reason for that is that nobdy in the robotics community uses HAVOK, but some people rather like to use what's in this notebook for state-space augmentation purposes, using time-delay embeddings. Hence, this offers a comparison between both algorithms. However, this algorithm does not directly aims at performing system identification but rather increase the state-space dimension before applying observables to this augmented state-space. Thus, its performance is poorer than HAVOK when not used in the context of state observables.
* ```11_Panda_robot_identification``` : First try at using the HAVOK algorithm from the notebook ```9_hyperopt```, but with a more efficient bayesian optimisation library, namely _Scikit-optimise_. The LQR control seems to work a little bit, but cannot be applied on the real system because it still suffers from the model quality
* ```12_Panda_robot_identification``` : Enhances the code from previous notebook, using another optimisation routine borrowed from GpyOpt from Sheffied because it truly leverages GPR for the search space.
* ```12_Panda_robot_identification_SLFC``` : Same as the preceeding notebook, but tested on the SLFC algorithm, presented in the 10th notebook.
* ```13_Residual_Panda_dynamics_learning``` : Since the LQR framework did not end up to succeed with the Panda robot, a first shot was given at trying to learn residual dynamics instead. This version is not the final one which is explained in the report though.
* ```14_Friction_learning``` : First tests using the final routine leveraging ELQR, as presented in the report.
* ```15_Uncertain_pendulum``` : Notebook providing the methodology to learn residual dynamics for a simple pendulum perturbed by viscous friction and how to suppress it with ELQR.
* ```16_Friction_learning``` : Final notebook for ELQR residual dynamics control for the Panda robot
* ```17_Panda_robot_learning``` : Final notebook for full model learning for Panda robot and generation of plots for the report and presentation
* ```19_2DPendulum``` : Attempts to control the residual dynamics of a double pendulum, using the same methodology as before
* ```19p_2DPendulum_control``` : Attempts to learn a complete model for the 2D pendulum as well as for controlling it using LQR within subspace

## Libraries developped

All useful functions, classes and methods used for this work can be found within the script file ```Functions.py```, which can be found within the ```LQR_simulation\Utils``` folder.

The main classes and their most important methods are listed below :

* ```SimplePendulum``` : instantiates a simple pendulum object, to which are linked the following methods :
    * ```Physics``` : pendulum dynamics, to be used along with a Runge Kutta integrator
    * ```CTC``` : _Computed Torque Control_ routine for performing feedback linearization onto the pendulum. Some terms can be desactivated using the boolean arguments.
* ```DoublePendulum``` : instantiates a 2D pendulum objects and its properties. 
    * ```Physics``` : as for the 1D pendulum, defines the 2D pendulum dynamics for numerical integration
    * ```CTC``` : again, _Computed Torque Control_ ...
* ```Robot``` : Class for defining objects containing all required information to simulate any robot (or pendulum). These information are typically the time-scale, trajectories, reference trajectories and all data for performing learning.
    * ```drawTrajectories``` : routine for generating pseudo-random trajectories according to the specifications of the dictionary ```specs```. A double integrator used along with a LQR allows to define the required accelerations.
    * ```toStateSPace``` : stores data necessary for learning later on.
* ```HAVOK``` : **most important class in this project** : implements the HAVOK algorithm and all methods necessary.
    * ```HANKEL``` : builds Hankel matrix
    * ```SVD``` : Performs SVD according to specified delay, defines transformations
    * ```LS``` : Least squares (or DMDc) routine for learning A and B.
    * ```Simulate``` : recursively simulates a learnt model within the Koopman subspace and transforms it back to the original space
    * ```ConstructLQR``` : builds LQR or ELQR objects and solves the respective problems
    * ```LQR_simulate``` : LQR or ELQR simulation within subspace
    * ```RMSE``` : _Root mean squared error_ for comparing different models
* ```SLFC``` : implements the _Subspace Learning For Control_ algorithm from the papers [20] or [21] of the report and makes use of the theory in Section 2.4 of the report.
    * ```delayEmbeddings``` : creates pseudo-Hankel matrix
    * ```EDMD``` : _Extended dynamic mode decomposition_
    * ```Simulate``` : simulates model within learnt subspace
    * ... all other routines perform the same operations as in the ```HAVOK``` classes, but adapted to this algorithm.
* ```DelayedLeastSquare``` : implements what is presented in the first 6 notebooks, i.e. learning with time delays, but without the HAVOK framework around it as well as a first version of HAVOK with SVD truncation
    * ```matricesConstruction``` : construct pseudo-Hankel matrix
    * ```truncate``` : performs SVD optimal truncation
    * ```solve``` : solves least square problem
* ```poglqr``` : software package for LQR developed at the IDIAP, but slightly modified for including ELQR control

## Software packages required

* ```numpy```
* ```scipy```
* ```seaborn```
* ```pbdlib```
* ```pybullet```
* ```joblib```
* ```pandapybullet```
* ```torch```
* ```time```
* ```GPy```
* ```GPyOpt```
