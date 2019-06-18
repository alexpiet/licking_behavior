import matplotlib.pyplot as plt
import numpy as np
import fit_tools

def plot_components(params, durations, sigmas, events,xlims,mean_lick):
    licks = events[3]
    fig,ax = plt.subplots(len(params)+1,1)
    fig.set_size_inches(10,14) 
    stop_time = 600
    dt = 0.01
    time_vec = np.arange(dt,6.01,dt)
    base = np.ones(np.shape(time_vec))
    base = base*np.exp(mean_lick)
    ax[0].plot(time_vec,base,'k-',linewidth=5)     
    ax[0].set_ylabel('P(lick/$\Delta t$)',fontsize=24)
    ax[0].xaxis.set_tick_params(labelsize=24)
    ax[0].yaxis.set_tick_params(labelsize=24) 
    ax[0].set_xlim(0,xlims)
    ax[0].set_ylim(bottom=0,top=np.exp(mean_lick)*2)
    for i in range(1, len(params)):
        f = params[i]
        e = events[i]
        d = durations[i]
        s = sigmas[i]

        linear_response = compute_linear_response(f,d,e,dt,s,stop_time)
        nonlinear = np.exp(linear_response)
        base = base*nonlinear
        ax[i].plot(time_vec,nonlinear,'k-',linewidth=5)     
        ax[i].plot(np.array(e)*dt,np.zeros(np.shape(e)), 'ro',markersize=8)  
        ax[i].set_ylabel('gain',fontsize=24)
        ax[i].xaxis.set_tick_params(labelsize=24)
        ax[i].yaxis.set_tick_params(labelsize=24) 
        ax[i].set_xlim(0,xlims)
    ax[5].plot(time_vec, base, 'k-', linewidth=5)
    ax[5].set_ylabel('P(Lick/$\Delta t$)',fontsize=24)
    ax[5].xaxis.set_tick_params(labelsize=24)
    ax[5].yaxis.set_tick_params(labelsize=24) 
    ax[5].set_xlim(0,xlims)
    plt.tight_layout()
    lick_ps = base[licks]
    print(np.mean(lick_ps))
    non_lick_ps = (np.sum(base) - np.sum(lick_ps))/(len(base) - len(lick_ps))
    print(non_lick_ps)

def compute_linear_response(params, durations, events, dt, sigma, stop_time):
    filter_time_vec =np.arange(dt, durations,dt)
    my_filter = fit_tools.build_filter(params, filter_time_vec, sigma)
    base = np.zeros((stop_time+len(my_filter)+1,))
    for i in events:
        base[int(i)+1:int(i)+1+len(my_filter)] += my_filter
    base = base[0:stop_time]
    return base


