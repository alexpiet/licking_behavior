import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
plt.ion() # makes non-blocking figures
import fit_tools

# Define which experiment id you want
experiment_id = 715887471

# Get the data
data = fit_tools.get_data(experiment_id, save_dir='./example_data')
licks = data['lick_timestamps']
running_timestamps = data['running_timestamps']

# get start/stop time for session
start_time = 1
dt = 0.01 # 10msec timesteps
stop_time = int(np.round(running_timestamps[-1],2)*(1/dt))
licks = licks[licks < stop_time/100]
licks = np.round(licks,2)
licksdt = np.round(licks*(1/dt))
time_vec = np.arange(0,stop_time/100.0,dt)


### Make mean lick rate example
# simple model with only mean lick rate
nll, latent = fit_tools.licking_model([-.5], licksdt, stop_time, post_lick=False,running_speed=False)

# Wrapper function for optimization that only takes one input
def mean_wrapper_func(mean_lick_rate):
    #return mean_lick_model(mean_lick_rate,licksdt,stop_time)[0]
    return fit_tools.licking_model(mean_lick_rate, licksdt, stop_time, post_lick=False,running_speed=False)[0]

# optimize
inital_param = 0
res_mean = minimize(mean_wrapper_func, inital_param)

# We get a sensible result!
Average_probability_of_lick = np.exp(res_mean.x)[0]
sanity_check = len(licks)/(stop_time + 0.000001)

def wrapper(mean_lick_rate):
    return fit_tools.licking_model(mean_lick_rate, licksdt, stop_time, post_lick=False,running_speed=False)

res_mean = fit_tools.evaluate_model(res_mean,wrapper, licksdt, stop_time)
fit_tools.compare_model(res_mean.latent, time_vec, licks, stop_time)

### Make Post lick filter example
nll, latent = fit_tools.licking_model(np.concatenate(([-.5],np.zeros((10,)))), licksdt, stop_time, post_lick=True,running_speed=False)

# Wrapper function for optimization that only takes one input
def post_lick_wrapper_func(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=True,running_speed=False)[0]

# optimize
inital_param = np.concatenate(([-.5],np.zeros((10,))))
res_post_lick = minimize(post_lick_wrapper_func, inital_param)

def wrapper(params):
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=True,running_speed=False)

res_post_lick = fit_tools.evaluate_model(res_post_lick,wrapper, licksdt, stop_time)
fit_tools.compare_model(res_post_lick.latent, time_vec, licks, stop_time)
fit_tools.build_filter(res_post_lick.x[1:], np.arange(dt,.21,dt), 0.025, plot_filters=True)






###############
# set up basic model, which has a constant lick rate
def mean_lick_model(mean_lick_rate,licksdt, stop_time):
    base = np.ones((stop_time,))*mean_lick_rate
    latent = np.exp(base)
    return loglikelihood(licksdt,latent), latent

# compute the negative log likelihood of poisson observations, given a latent vector
def loglikelihood(licksdt, latent):
    NLL = -sum(np.log(latent)[licksdt.astype(int)]) + sum(latent)
    return NLL

# Wrapper function for optimization that only takes one input
def mean_wrapper_func(mean_lick_rate):
    return mean_lick_model(mean_lick_rate,licksdt,stop_time)[0]

# optimize
res_mean = minimize(mean_wrapper_func, 1)

# We get a sensible result!
Average_probability_of_lick = np.exp(res_mean.x)[0]
sanity_check = len(licks)/(stop_time + 0.000001)

# evaluate fit by plotting prediction and lick times
def compare_model(latent, time_vec, licks, stop_time):
    fig,ax  = plt.subplots()    
    plt.plot(time_vec,latent,'b',label='model')
    plt.vlines(licks,0, 1, alpha = 0.3, label='licks')
    plt.ylim([0, 1])
    plt.xlim(600,660)
    plt.legend(loc=9 )
    plt.xlabel('time (s)')
    plt.ylabel('Licking Probability')
    plt.tight_layout()
    return fig, ax

# BIC = log(#num-data-points)*#num-params - 2*log(L)
#     = log(x)*k + 2*NLL
def compute_bic(nll, num_params, num_data_points):
    return np.log(num_data_points)*num_params + 2*nll

# make a figure
res_mean.nll, res_mean.latent = mean_lick_model(res_mean.x, licksdt, stop_time)
res_mean.BIC = compute_bic(res_mean.nll, len(res_mean.x), len(res_mean.latent))
compare_model(res_mean.latent, time_vec, licks, stop_time)


### Lets improve our model with a post-licking filter
def mean_post_lick_model(params, licksdt,stop_time):
    mean_lick_rate = params[0]
    base = np.ones((stop_time,))*mean_lick_rate
    post_lick_filter = params[1:]
    post_lick = np.zeros((stop_time+len(post_lick_filter)+1,))
    for i in licksdt:
        post_lick[int(i)+1:int(i)+1+len(post_lick_filter)] +=post_lick_filter
    post_lick = post_lick[0:stop_time]
    latent = np.exp(base+post_lick)
    return loglikelihood(licksdt,latent), latent

def post_lick_wrapper_func(params):
    return mean_post_lick_model(params,licksdt,stop_time)[0]

# optimize
res_post_lick = minimize(post_lick_wrapper_func, np.ones(21,))
res_post_lick.nll,res_post_lick.latent = mean_post_lick_model(res_post_lick.x, licksdt,stop_time)
res_post_lick.BIC = compute_bic(res_post_lick.nll, len(res_post_lick.x), len(res_post_lick.latent))
compare_model(res_post_lick.latent, time_vec, licks, stop_time)

if res_post_lick.BIC < res_mean.BIC:
    print('BIC favors the post-lick filter')

# But how long of a filter should we use? We can do model optimization to find out
models = []
models.append(res_mean)
keep_going = True
current_val = 1
while keep_going:
    res = minimize(post_lick_wrapper_func, np.ones(1+current_val,))
    res.nll,res.latent = mean_post_lick_model(res.x, licksdt,stop_time)
    res.BIC = compute_bic(res.nll, len(res.x), len(res.latent))   
    models.append(res)
    if models[current_val].BIC < models[current_val - 1].BIC:
        print('BIC favors extending the model '+ str(current_val)+" "+str(res.BIC))    
    else:
        print('BIC does not favor extending the model '+ str(current_val)+" "+str(res.BIC))   
    current_val += 1
    if current_val > 100:
        keep_going = False



# And the winner is!
BIC = []
for res in models:
    BIC.append(res.BIC)

tvec = np.arange(0,len(BIC)*dt,dt)
plt.plot(tvec,BIC-BIC[0],'ro')
plt.plot(tvec, np.zeros(np.shape(tvec)), 'k--', alpha=0.3)
plt.xlabel('Length of Post Lick Filter (s)')
plt.ylabel('Training Set BIC')
plt.tight_layout()
plt.ylim(ymax = 500)
plt.ylim(ymin = np.min(BIC-BIC[0])-500)

filters = []
plt.figure()
for res in models:
    if len(res.x) > 12:
        filters.append(res.x[1:])
        plt.plot(tvec[0:len(res.x[1:])+1], np.exp(res.x[0:])*dt,'k-',alpha = 0.3)

plt.ylabel('Licking Probability')
plt.xlabel('Time from lick (s)')
plt.tight_layout()

# So this is nice, but it takes forever with a long filter. It also fits noise (the filter is bumpy)
# So instead, lets fit with a basis function to parameterize the filter



# puts len(params) gaussian bumps equally spaced across time_vec
# each gaussian is weighted by params, and is truncated outside of time_vec
def build_filter(params,filter_time_vec, sigma, plot_filters=False):
    def gaussian_template(mu,sigma):
        return (1/(np.sqrt(2*3.14*sigma**2)))*np.exp(-(filter_time_vec-mu)**2/(2*sigma**2))
    numparams = len(params)
    mean = (filter_time_vec[-1] - filter_time_vec[0])/(numparams-1)
    base = np.zeros(np.shape(filter_time_vec)) 
    if plot_filters:
        plt.figure()
    for i in range(0,len(params)):
        base += params[i]*gaussian_template(mean*i,sigma)    
        if plot_filters:
            plt.plot(filter_time_vec, params[i]*gaussian_template(mean*i,sigma))
    if plot_filters:
        plt.plot(filter_time_vec,base, 'k')
    return base

filter_time_vec = np.arange(dt,.21,dt)
# plot an example filter
build_filter([2.75,-2,-2,-2,-2,3,3,3,.1], filter_time_vec, 0.025, plot_filters=True)

def basis_post_lick_model(params, licksdt,stop_time,sigma):
    mean_lick_rate = params[0]
    base = np.ones((stop_time,))*mean_lick_rate
    filter_time_vec = np.arange(dt,.21,dt)
    post_lick_filter = build_filter(params[1:],filter_time_vec,sigma)
    post_lick = np.zeros((stop_time+len(post_lick_filter)+1,))
    for i in licksdt:
        post_lick[int(i)+1:int(i)+1+len(post_lick_filter)] +=post_lick_filter
    post_lick = post_lick[0:stop_time]
    latent = np.exp(base+post_lick)
    return loglikelihood(licksdt,latent), latent

def basis_post_lick_wrapper_func(params):
    return basis_post_lick_model(params,licksdt,stop_time,0.025)[0]

# optimize
init = [-.5, -.5, 0.01, 0.02, -.5]
res_basis = minimize(basis_post_lick_wrapper_func, init)


res_basis.nll,res_basis.latent = basis_post_lick_model(res_basis.x, licksdt,stop_time,0.025)
res_basis.BIC = compute_bic(res_basis.nll, len(res_basis.x), len(res_basis.latent))
compare_model(res_basis.latent, time_vec, licks, stop_time)
build_filter(res_basis.x[1:], filter_time_vec, 0.025, plot_filters=True)
plt.figure()
plt.plot(filter_time_vec, np.exp(build_filter(res_basis.x[1:], filter_time_vec, 0.025, plot_filters=False)))







