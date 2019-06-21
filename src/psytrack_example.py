# code for messing with psytrack

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from psytrack.runSim import generateSim
from psytrack.runSim import recoverSim
from psytrack.hyperOpt import hyperOpt
import psy_tools as ps

K = 3 # number of parameter classes
N = 4000 # number of flashes
seed = 17 # random seed for reproduction

# Generate Data, the inputs are defined within the function
simDataVB = ps.generateSim_VB(K,N,seed=seed)

# Plot time-varying weights
plt.figure(figsize=(10,5))
plt.plot(simDataVB['W'][:,0])
plt.plot(simDataVB['W'][:,1])
plt.plot(simDataVB['W'][:,2])
plt.xlabel('Trial #'); plt.ylabel('Weight')

# Plot Input Drives
fig,ax =plt.subplots(4,figsize=(10,8))
ax[0].plot(simDataVB['X'][:,0],color='tab:blue')
ax[0].set_xlabel('Trial #'); ax[0].set_ylabel('Random Guessing Drive')
ax[0].set_xlim([0,100])
ax[1].plot(simDataVB['X'][:,1],color='tab:orange')
ax[1].set_xlabel('Trial #'); ax[1].set_ylabel('Timing Drive')
ax[1].set_xlim([0,100])
ax[2].plot(simDataVB['X'][:,2],color='tab:green')
ax[2].set_xlabel('Trial #'); ax[2].set_ylabel('Task Drive')
ax[2].set_xlim([0,100])
ax[3].plot(simDataVB['all_Y'][0][:],'ko',alpha=0.5)
ax[3].set_xlim([0,100])
ax[3].set_xlabel('Trial #'); ax[3].set_ylabel('Lick (1) or Not (2)')

# Recover the weights already knowing the hyperparameters
recVB = recoverSim(simDataVB)
plt.figure(figsize=(10,5))
plt.plot(simDataVB['W'], linestyle="-", alpha=0.5)
plt.plot(recVB['wMode'].T, linestyle="--", lw=3, color="black")
plt.xlabel("Trial #"); plt.ylabel("Weight");

# Recover the weights without already knowing the weights
# How many types of weights are we fitting?
weights = { 'random': 1,
            'timing': 1,
            'task': 1}

# What hyperparameters exist?
hyper = {   'sigInit': 2**4., # prior over initial weights, make it huge
            'sigma': [2**-4.]*K, #each weight has same initial sigma
            'sigDay': None }

optList = ['sigma'] # which hyper parameters to optimize

# Create data structure
inputDict = {   'random': simDataVB['X'][:,0][:,np.newaxis],
                'timing': simDataVB['X'][:,1][:,np.newaxis],
                'task': simDataVB['X'][:,2][:,np.newaxis]}

D = {   'y': simDataVB['all_Y'][0],
        'inputs': inputDict}


hyp, evd, wMode, hess = hyperOpt(D, hyper,weights, optList)
plt.plot(wMode.T, linestyle="-", lw=3, color="red")











