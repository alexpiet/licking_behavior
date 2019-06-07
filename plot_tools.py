import matplotlib.pyplot as plt
import numpy as np
import fit_tools

def plot_components(params, durations, sigmas, events):
    fig,ax = plt.subplots(4,1)
    for i in range(0, len(params)):
        f = params[i]
        e = events[i]
        d = durations[i]
        s = sigmas[i]
        stop_time = 600
        dt = 0.01
        linear_response = compute_linear_response(f,d,e,dt,s,stop_time)
        nonlinear = np.exp(linear_response)
        time_vec = np.arange(dt,6.01,dt)
        ax[i].plot(time_vec,nonlinear,'k-')     
        ax[i].plot(np.array(e)*dt,np.zeros(np.shape(e)), 'ro')  
        ax[i].set_ylabel('gain') 
    plt.tight_layout()

def compute_linear_response(params, durations, events, dt, sigma, stop_time):
    filter_time_vec =np.arange(dt, durations,dt)
    my_filter = fit_tools.build_filter(params, filter_time_vec, sigma)
    base = np.zeros((stop_time+len(my_filter)+1,))
    for i in events:
        base[int(i)+1:int(i)+1+len(my_filter)] += my_filter
    base = base[0:stop_time]
    return base


