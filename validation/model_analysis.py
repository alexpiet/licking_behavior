import pickle
from alex_utils import load
import matplotlib.pyplot as plt
import fit_tools
plt.ion()
import numpy as np

filepath = '/Users/alex.piet/glm_fits/'

experiment_ids = [837729902, 838849930,836910438,840705705,840157581,841601446,840702910,841948542,841951447,842513687,842973730,843519218,846490568,847125577,848697604] 
dt = 0.01

for ids in experiment_ids:
    res = load(filepath+'fitglm_'+str(ids))
    print(np.sum(res.jac == 0))


plt.close('all')
fig,ax = plt.subplots(3,2)
for ids in experiment_ids:
    res = load(filepath+'fitglm_'+str(ids))
    if np.sum(res.x == 0) <= 15:
        mean_lick = res.x[0]
        mean_lick_nl = np.exp(mean_lick)
        post_lick = res.x[1:11]
        post_lick_filter=fit_tools.build_filter(post_lick,np.arange(dt,.21,dt),0.025)
        post_lick_nl = np.exp(post_lick_filter)
        run_speed = res.x[11:17]
        run_speed_nl = np.exp(run_speed)
        reward    = res.x[17:57]
        reward_filter=fit_tools.build_filter(reward,np.arange(dt,4,dt),0.1)
        reward_nl = np.exp(reward_filter)
        flash     = res.x[57:72]
        flash_filter = fit_tools.build_filter(flash, np.arange(dt,0.7,dt), 0.025)
        flash_nl = np.exp(flash_filter)
        change_flash  = res.x[72:87]
        change_flash_filter = fit_tools.build_filter(change_flash,np.arange(dt,0.7,dt),0.025)
        change_flash_nl = np.exp(change_flash_filter)
        ax[0,0].plot(np.arange(dt,0.21,dt),post_lick_nl,'k-',alpha=.3)
        ax[1,0].plot(np.arange(dt,dt*7,dt),run_speed_nl,'k-',alpha=.3)
        if max(reward_nl) < 500:
            ax[2,0].plot(np.arange(dt,4,dt),reward_nl,'k-',alpha=.3)
        ax[0,1].plot(np.arange(dt,.7,dt),flash_nl,'k-',alpha=.3)
        ax[1,1].plot(np.arange(dt,.7,dt),change_flash_nl,'k-',alpha=.3)
        ax[2,1].plot(mean_lick_nl*100,0,'ko',alpha = .3)
    

plt.tight_layout()
ax[0,0].title.set_text('post lick')
ax[0,0].set_xlabel('Time (s)')
ax[1,0].title.set_text('running speed')
ax[1,0].set_xlabel('Time (s)')
ax[1,0].set_ylim(ymin=0)
ax[2,0].title.set_text('reward')
ax[2,0].set_ylim(ymin=0)
ax[2,0].set_xlabel('Time (s)')
ax[0,1].title.set_text('flash')
ax[0,1].set_ylim(ymin=0)
ax[0,1].set_xlabel('Time (s)')
ax[1,1].title.set_text('change flash')
ax[1,1].set_ylim(ymin=0)
ax[1,1].set_xlabel('Time (s)')
ax[2,1].title.set_text('mean lick rate')
ax[2,1].set_xlim(xmin=0)
ax[2,1].set_xlabel('licks/s')

