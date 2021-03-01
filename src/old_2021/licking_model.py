import os
import sys
import numpy as onp
import time
import datetime
import h5py
from collections import OrderedDict
from scipy.optimize import minimize
from functools import partial
import matplotlib as mpl
from matplotlib import pyplot as plt
import fit_tools
import pickle

USEJAX=True

if USEJAX: 
    print("Using JAX")
    import jax.numpy as np
    from jax import grad, jacfwd, jacrev, lax, hessian
    from jax.experimental.optimizers import adam
    from jax import jit
    from jax.config import config
    #  config.update('jax_disable_jit', True)
else:
    print("Using autograd")
    import autograd.numpy as np
    from autograd import grad
    from autograd.scipy import signal
    jit = lambda x: x

import copy
import itertools

def boxoff(ax, keep="left", yaxis=True):
    """
    Hide axis lines, except left and bottom.
    You can specify which axes to keep: 'left' (default), 'right', 'none'.
    """
    ax.spines["top"].set_visible(False)
    xtlines = ax.get_xticklines()
    ytlines = ax.get_yticklines()
    if keep == "left":
        ax.spines["right"].set_visible(False)
    elif keep == "right":
        ax.spines["left"].set_visible(False)
    elif keep == "none":
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        for t in xtlines + ytlines:
            t.set_visible(False)
    for t in xtlines[1::2] + ytlines[1::2]:
        t.set_visible(False)
    if not yaxis:
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ytlines = ax.get_yticklines()
        for t in ytlines:
            t.set_visible(False)

def _loglikelihood(licksdt, latent, n_total_bins=None):
    '''
    Compute the negative log likelihood of poisson observations, given a latent vector

    Args:
        licksdt: a vector of len(time_bins) with 1 if the mouse licked
                 at in that bin
        latent: a vector of the estimated lick rate in each time bin
        params: a vector of the parameters for the model
        l2: amplitude of L2 regularization penalty
    
    Returns: NLL of the model
    '''
    if n_total_bins is None:
        n_total_bins = latent.shape[0]
    # If there are any zeros in the latent model, have to add "machine tolerance"
    latent += onp.finfo(np.float32).eps
    LL = np.dot(np.log(latent), licksdt) - np.sum(latent) #Proportional! ignoring the k!
    LL = LL * (n_total_bins / latent.shape[0]) # normalize by total number of bins
    return LL

loglikelihood = jit(_loglikelihood)

def _negative_log_evidence(licksdt,
                           latent,
                           n_total_bins,
                           params,l2=0):
    LL = loglikelihood(licksdt, latent, n_total_bins)
    prior = l2*np.sum(np.array(params)**2)
    log_evidence = LL - prior
    return -1 * log_evidence

negative_log_evidence = jit(_negative_log_evidence)

class Model(object):

    def __init__(self, dt, licks, name=None,
                 l2=0, verbose=False, n_splits=2,
                 cross_validate=False):

        # TODO: Can we use licks as 0/1 vec instead of inds?
        '''
        Args:
            licks: a vector of lick times in dt-index points
        '''

        self.dt = dt
        self.licks = licks
        self.filters = OrderedDict()
        self.latent = None
        self.NLL = None
        self.BIC = None
        self.l2=l2
        self.verbose=verbose

        # Initial param guess for mean rate
        self.mean_rate_param = -0.5
        self.num_time_bins = len(licks)

        # Construct train/test splits at the beginning

        if name is None:
            self.name='test'
        else:
            self.name=name

        self.n_splits = n_splits
        self.splits = self.get_data_splits(n_splits=self.n_splits)
        self.training_split = onp.sum(self.splits[:-1, :], axis=0).astype(bool)
        self.test_split = self.splits[-1, :]

        # Testing
        #  self.splits = self.get_data_splits(n_splits=2)
        #  self.training_split = self.splits[0, :]
        #  self.test_split = self.splits[1, :]


    def initialize_filters(self):
        self.mean_rate_param = -0.5
        for filter_name, filter_obj in self.filters.items():
            filter_obj.initialize_params()

    def add_filter(self, filter_name, filter):
        '''
        Add a filter to the model. 

        Args: 
            filter_name (str): The filter's name
            filter (instance of Filter, GaussianBasisFilter, etc.)
        '''
        self.filters[filter_name] = filter

    def set_filter_params(self, flat_params):
        '''
        Break up a flat array of params and set them for each filter in the model.
        '''
        self.mean_rate_param = flat_params[0] # The first param is always the mean.
        flat_params = flat_params[1:]
        param_start = 0
        for filter_name, filter in self.filters.items():
            num_params = filter.num_params
            filter.set_params(flat_params[param_start:param_start+num_params])
            param_start += num_params
        if not param_start == len(flat_params):
            raise ValueError("We didn't use all of the params when setting")

    def get_filter_params(self):
        '''
        Take params from each filter out into a flat array. 
        '''
        paramlist = [np.array([self.mean_rate_param])] # First param is always the mean rate
        for filter_name, filter in self.filters.items():
            paramlist.append(filter.params)
        return np.concatenate(paramlist)

    # TODO target for JIT
    def calculate_latent(self):
        '''
        Filters own their params and data, so we just call the linear_output
        method on each filter and add up the result
        '''

        base = np.zeros(self.num_time_bins)
        base += self.mean_rate_param # Add in the mean rate

        for filter_name, filter in self.filters.items():
            base += filter.linear_output()
        latent = np.exp(np.clip(base, -88, 88))
        return latent

    def get_data_splits(self, n_splits=6):
        '''
        We should use 4 of these for training, one for early stopping, and one for test
        Returns:
            bool of shape (nSplits, nBins): Whether each bin is included in that split
        '''
        n_bins = self.licks.shape[0]
        split_id_each_bin = onp.random.randint(n_splits, size=n_bins)
        split_id_this_row = onp.repeat(onp.arange(n_splits)[:,np.newaxis], repeats=n_bins, axis=1)
        split_id_this_col = onp.repeat(split_id_each_bin[np.newaxis, :], repeats=n_splits, axis=0)
        return split_id_this_row == split_id_this_col

    def get_licks_and_latent(self, bins_to_use=None):
        '''
        Give me the licks in those bins, plus the latent predictions in the bins
        '''
        # Some kind of train inds
        if bins_to_use is None: 
            bins_to_use = np.ones(len(self.licks)).astype(bool)
        licks_to_use = self.licks[bins_to_use]
        latent_to_use = self.calculate_latent()[bins_to_use]
        return licks_to_use, latent_to_use

    def nle(self, split=None):
        if split is None:
            split = self.training_split
        licks_to_use, latent_to_use = self.get_licks_and_latent(split)
        n_total_bins = self.licks.shape[0]
        NLE = negative_log_evidence(licks_to_use,
                                    latent_to_use,
                                    n_total_bins,
                                    self.get_filter_params(),
                                    self.l2)
        return NLE

    def nll(self, split=None):
        '''
        Return the log-liklihood of the data given the model
        '''
        if split is None:
            split = self.test_split
        licks_to_use, latent_to_use = self.get_licks_and_latent(split)
        n_total_bins = self.licks.shape[0]
        NLL = float(-1 * loglikelihood(licks_to_use, latent_to_use, n_total_bins))
        return NLL

    # Function to minimize
    def objective(self, params, split):
        self.set_filter_params(params)
        return self.nle(split)
    
    def fit(self, training_split=None):

        if training_split is None: 
            training_split = self.training_split #Use default setting if we aren't crossvalidating

        params = self.get_filter_params()

        sys.stdout.write("Fitting model with {} params\n".format(len(params)))
        start_time = time.time()

        def print_NLE_callback(xk):
            '''
            A callback for printing info about each iteration.
        
            Args:
                xk: This is the vector of current params (this is how the 
                    callable is executed per the minimize docs). We don't 
                    use it in this func though.
            '''
            sys.stdout.flush() # This and the \r make it keep writing the same line
            sys.stdout.write('\r')
            #  NLL, latent = self.calculate_latent()
            NLL = self.nll()
            self.iteration+=1
            sys.stdout.write("Iteration: {} NLE: {}".format(self.iteration, NLL))
            sys.stdout.flush()
        
        kwargs = {}
        if self.verbose:
            self.iteration=0
            kwargs.update({'callback':print_NLE_callback})
        cost_func = partial(self.objective, split=training_split)
        g_jit = jit(grad(cost_func))
        res = minimize(cost_func, params, jac=g_jit, **kwargs)

        #Want to use the adam optimizer from jax? We should try.

        elapsed_time = time.time() - start_time
        sys.stdout.write('\n')
        sys.stdout.write("Done! Elapsed time: {:02f} sec".format(time.time()-start_time))

        # Set the final version of the params for the filters
        self.set_filter_params(res.x)
        self.res = res

    #TODO need a better name for this
    def dropout_analysis(self, cache_dir=None):
        nll = self.nll()
        dropout_nll_percent_change = {}
        dropout_nll = {}
        if len(self.filters)<1:
            # TODO: Should we define this? Like, drop out the baseline rate?
            print("no filters to drop out")
            pass
        else:
            for ind_filter, filter_to_drop in enumerate(self.filters.keys()):
            # for filters_to_use in itertools.combinations(filter_names,

                #Make a copy of the model
                sub_model = copy.deepcopy(self)

                #Remove the filter
                print("\nRemoving filter: {}\n".format(filter_to_drop))
                del(sub_model.filters[filter_to_drop])

                sub_model.fit()
                sub_nll = sub_model.nll()
                dropout_nll[filter_to_drop] = sub_nll
                dropout_nll_percent_change[filter_to_drop] = 100*((nll-sub_nll)/nll)

        print("\nDone with dropout analysis")
        self.dropout_nll = dropout_nll
        self.dropout_nll_percent_change = dropout_nll_percent_change

        if cache_dir is not None:
            fn = "{}_dropout_percent_change".format(self.name)
            np.savez(os.path.join(cache_dir, fn), **self.dropout_percent_change)

    ## Metric functions need to take nonlinear filter and time vector for now.
    @staticmethod
    def time_to_peak(nonlinear_filt, filter_time_vec):
        return filter_time_vec[np.argmax(nonlinear_filt)]

    @staticmethod
    def max_gain(nonlinear_filt, filter_time_vec):
        return np.max(nonlinear_filt)

    def filter_metrics(self):
        '''
        Return summary metrics about each filter.
        '''
        metric_funcs = {"time_to_peak": self.time_to_peak,
                        "max_gain":self.max_gain}
        
        metric_output = {}
        for ind_filter, (filter_name, filter_obj) in enumerate(self.filters.items()):
            metric_output[filter_name] = {}
            linear_filt = filter_obj.build_filter()
            nonlinear_filt = np.exp(linear_filt)
            filter_time_vec = filter_obj.filter_time_vec
            
            for metric_name, metric_func in metric_funcs.items():
                metric_output[filter_name].update({metric_name:metric_func(nonlinear_filt,
                                                                           filter_time_vec)})
        return metric_output

    def eval(self):
        '''
        Evaluate the model, updating the model res with BIC

        '''
        self.res = evaluate_model(self.res, self.calculate_latent,
                                  self.licksdt, self.stop_time)

    def plot_filters(self, show=True, save_dir=None):
        #plt.clf()
        n_filters = len(self.filters)
        for ind_filter, (filter_name, filter_obj) in enumerate(self.filters.items()):
            linear_filt = filter_obj.build_filter()
            plt.subplot(2, (n_filters/2)+1, ind_filter+1)
            plt.plot(filter_obj.filter_time_vec, 
                     np.exp(linear_filt),
                     'k-')
            plt.title(filter_name)
            plt.xlabel('time (s)')
            ax = plt.gca()
            boxoff(ax)
        plt.tight_layout()
        if save_dir is not None:
            fig_name = "{}_nonlinear_filters.pdf".format(self.name)
            plt.savefig(os.path.join(save_dir, fig_name))
        if show:
            plt.show()

    def save(self, output_dir, fn=None, verbose=True):
        if fn==None:
            today = datetime.datetime.now().strftime("%Y%m%d")
            fn = "model_{}_fit_{}.pkl".format(self.name, 
                                              today)
        full_path = os.path.join(output_dir, fn)
        if verbose:
            print("Saving model to: {}".format(full_path))
        pickle_model(self, full_path)

    def save_filter_params(self, output_dir, fn=None):
        if fn==None:
            fn = "model_{}_{}_saved_params.npz".format(self.name, 
                                                       len(self.get_filter_params()))

        param_dict = {filt_name:filt.params for filt_name, filt in self.filters.items()}
        full_path = os.path.join(output_dir, fn)
        np.savez(full_path,
                 mean_rate=self.mean_rate_param,
                 **param_dict)

    def set_filter_params_from_file(self, save_fn):
        # TODO: some smarter checking here. Up to us to not break it.
        param_dict = np.load(save_fn)
        self.mean_rate_param = param_dict['mean_rate']
        for filter_name, filt in self.filters.items():
            filt.set_params(param_dict[filter_name])


# Can we do the convolve func with a vec of 0/1 instead of rolling our own?
# so linear_whatever funcs can use the same thing.

class Filter(object):

    def __init__(self, num_params, data, initial_params=None):
        '''
        Base class for filter objects

        Args:
            num_params (int): Number of filter parameters
            data (np.array): The data relevant to the filter. Always has to
                             be num_bins in length. For filters that operate
                             on discrete events, pass an array with ones at
                             time bin indices where the event happened, and 
                             zero otherwise.
            initial_params (np.array): Initial parameter guesses. 
        '''
        self.num_params = num_params
        self.data = data

        if initial_params is not None:
            self.set_params(initial_params)
        else:
            self.initialize_params()

    def set_params(self, params):
        if not len(params) == self.num_params:
            raise ValueError(("Trying to give {} params to the {} filter"
                              " which takes {} params".format(len(params),
                                                              self.name,
                                                              self.num_params)))
        else:
            self.params = params

    def initialize_params(self):
        '''
        Init all params to zero by default
        '''
        self.params = np.zeros(self.num_params)

    def linear_output(self):
        '''
        This base class just convolves the filter params with the data.
        Cuts the output to be the same length as the data
        '''
        output = np.convolve(self.data, self.params)[:self.stop_time]

        # Shift prediction forward by one time bin
        #  output = np.r_[0, output[:-1]]
        output = np.concatenate([0, output[:-1]])

        return output


class GaussianBasisFilter(object):
    def __init__(self, num_params, data, dt, duration, sigma):
        '''
        A filter implemented as the sum of a number of gaussians. 

        Param value controls the amplitude of each gaussian. 
        Uniformly spaced over the duration of the filter.
        
        Args:
            num_params (int): Number of gaussians to use
            data (np.array): See documentation for data in Filter
            dt (float): Time in seconds per time bin
            duration (float): filter duration in seconds
            sigma (float): Std for each gaussian.
        '''
        self.name=''
        self.num_params = num_params
        self.data = data
        #  self.set_params(initial_params)
        self.initialize_params()

        #  if initial_params is not None:
        #      self.set_params(initial_params)
        #  else:

        self.duration = duration
        self.dt = dt
        self.sigma = sigma
        self.filter_time_vec = np.arange(dt, duration, dt)


    def set_params(self, params):
        if not len(params) == self.num_params:
            raise ValueError(("Trying to give {} params to the a filter"
                              " which takes {} params".format(len(params),
                                                              self.num_params)))
        else:
            self.params = params

    def initialize_params(self):
        '''
        Init all params to zero by default
        '''
        self.params = np.zeros(self.num_params)



    def build_filter(self):
        '''
        This replicates the function in fit_tools just to make this self-contained.
        '''

        def gaussian_template(x, mu, sigma):
            return (1 / (np.sqrt(2 * 3.14 * sigma ** 2))) * \
                   np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        mean = (self.filter_time_vec[-1] - self.filter_time_vec[0]) / \
               (self.num_params-1)

        # Empty array to save each gaussian
        basis_funcs = np.empty((self.num_params, len(self.filter_time_vec)))

        # Zero filter array to start with
        filter = np.zeros(np.shape(self.filter_time_vec)) 

        # Add each gaussian to the filter
        for ind_param in range(0, len(self.params)):
            this_gaussian = self.params[ind_param] * \
                            gaussian_template(self.filter_time_vec,
                                              mean * ind_param,
                                              self.sigma)    
            filter += this_gaussian

            # Save this gaussian
            # TODO: Not working with autograd
            # basis_funcs[ind_param, :] = this_gaussian

        return filter

    def linear_output(self):
        filt = self.build_filter()
        output = linear_output_external(self.data, filt)
        return output


if USEJAX:
    def _linear_output_external_xla(data, filt):
        datalen = len(data)
        rhs = filt[::-1].reshape(1, 1, -1, 1)
        lhs = data.astype(np.float32).reshape(1, 1, -1, 1)
        window_strides = np.array([1, 1])
        padding = 'SAME'
        output = lax.conv_general_dilated(lhs, rhs, window_strides, padding).ravel()[:datalen]
        output = np.concatenate([np.zeros((len(filt)//2)+1), output[:-1]])[:datalen]
        return output
    linear_output_external = jit(_linear_output_external_xla)
else:
    def _linear_output_external(data, filt):
        datalen = len(data)
        output = signal.convolve(data, filt)[:datalen]
        output = np.concatenate([np.array([0]), output[:-1]])
        return output
    linear_output_external = _linear_output_external

class MixedGaussianBasisFilter(GaussianBasisFilter):
    '''
    Gaussian basis filter with more basis funcs clustered near zero,
    and then wider basis funcs farther away.

    Args:
        num_params_narrow (int): Number of narrow basis funcs
        num_params_wide (int): Number of wide basis funcs
        duration_total (float): Total duration of the filter
        boundary (float): time boundary between narrow and wide filters
        sigma_narrow (float): Sigma for narrow basis funcs
        sigma_wide (float): Sigma for wide basis funcs
    '''
    def __init__(self,
                 data, dt,
                 num_params_narrow, num_params_wide,
                 duration_total, boundary,
                 sigma_narrow, sigma_wide):

        self.num_params = num_params_narrow + num_params_wide
        self.num_params_narrow = num_params_narrow
        self.num_params_wide = num_params_wide
        self.data = data
        self.initialize_params()

        if boundary > duration_total:
            raise ValueError("Boundary must be within total duration")

        self.duration = duration_total
        self.boundary = boundary

        self.dt = dt
        self.sigma_narrow = sigma_narrow
        self.sigma_wide = sigma_wide
        self.filter_time_vec = np.arange(dt, duration_total, dt)

    def build_filter(self, return_basis_funcs=False):

        def gaussian_template(x, mu, sigma):
            return (1 / (np.sqrt(2 * 3.14 * sigma ** 2))) * \
                   np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        #TODO: Should we not overlap on the boundary point? 
        filter_means_narrow = np.linspace(0,
                                          self.boundary,
                                          self.num_params_narrow)

        filter_means_wide = np.linspace(self.boundary,
                                        self.duration,
                                        self.num_params_wide)

        all_means = np.concatenate([filter_means_narrow, filter_means_wide])
        all_sigmas = np.concatenate([np.full(filter_means_narrow.shape, self.sigma_narrow),
                                     np.full(filter_means_wide.shape, self.sigma_wide)])

        # Empty array to save each gaussian
        basis_funcs = np.empty((self.num_params, len(self.filter_time_vec)))

        # Zero filter array to start with
        filter = np.zeros(np.shape(self.filter_time_vec)) 

        # Add each gaussian to the filter
        for ind_param in range(0, len(self.params)):
            this_gaussian = self.params[ind_param] * \
                            gaussian_template(self.filter_time_vec,
                                              all_means[ind_param],
                                              all_sigmas[ind_param])    
            filter += this_gaussian

            # Save this gaussian
            # TODO: Not working with autograd
            if return_basis_funcs:
                basis_funcs[ind_param, :] = this_gaussian

        if return_basis_funcs:
            return filter, basis_funcs
        else:
            return filter

def bin_data(data, dt, time_start=None, time_end=None):

    lick_timestamps = data['lick_timestamps']
    running_timestamps = data['running_timestamps']
    running_speed = data['running_speed']
    reward_timestamps = data['reward_timestamps']
    flash_timestamps = data['stim_on_timestamps']

    if time_start is None:
        time_start = 0
    if time_end is None:
        time_end = running_timestamps[-1]+0.00001

    change_flash_timestamps = fit_tools.extract_change_flashes(data)

    running_speed = running_speed[(running_timestamps >= time_start) & \
                                  (running_timestamps < time_end)]

    # Normalize running speed
    running_speed = running_speed / running_speed.max()

    running_timestamps = running_timestamps[(running_timestamps >= time_start) & \
                                            (running_timestamps < time_end)]


    reward_timestamps = reward_timestamps[(reward_timestamps >= time_start) & \
                                          (reward_timestamps < time_end)]


    lick_timestamps = lick_timestamps[(lick_timestamps >= time_start) & \
                                      (lick_timestamps < time_end)]

    flash_timestamps = flash_timestamps[(flash_timestamps >= time_start) & \
                                        (flash_timestamps < time_end)]

    change_flash_timestamps = change_flash_timestamps[
        (change_flash_timestamps > time_start) & (change_flash_timestamps < time_end)
    ]

    running_acceleration = fit_tools.compute_running_acceleration(running_speed)


    for arr in [running_timestamps, reward_timestamps, lick_timestamps,
                flash_timestamps, change_flash_timestamps]:
        arr -= running_timestamps[0]

    timebase = onp.arange(0, time_end-time_start, dt)
    edges = onp.arange(0, time_end-time_start+dt, dt)

    licks_vec, _ = onp.histogram(lick_timestamps, bins=edges)
    rewards_vec, _ = onp.histogram(reward_timestamps, bins=edges)
    flashes_vec, _ = onp.histogram(flash_timestamps, bins=edges)
    change_flashes_vec, _ = onp.histogram(change_flash_timestamps, bins=edges)

    return (licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
            running_speed, running_timestamps, running_acceleration,
            timebase, time_start, time_end)

# Funcs for saving and loading models with pickle
def pickle_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def unpickle_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
        return model

if __name__ == "__main__":

    dt = 0.01

    experiment_id = 715887471
    data = fit_tools.get_data(experiment_id, save_dir='../example_data')

    (licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
     running_speed, running_timestamps, running_acceleration, timebase,
     time_start, time_end) = bin_data(data, dt, time_start=300, time_end=1000)

    #  (licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
    #   running_speed, running_timestamps, running_acceleration, timebase,
    #   time_start, time_end) = bin_data(data, dt)

    case=9
    if case==0:
        # Model with just mean rate param
        model = Model(dt=0.01,
                      licks=licks_vec,
                      verbose=True)
        model.fit()

    elif case==1:

        model = Model(dt=0.01,
                      licks=licks_vec,
                      verbose=True)

        post_lick_filter = GaussianBasisFilter(num_params = 10,
                                               data = licks_vec,
                                               dt = model.dt,
                                               duration = 0.21,
                                               sigma = 0.025)
        model.add_filter('post_lick', post_lick_filter)

        #print(grad(model.set_params_and_calculate_nll)(model.get_filter_params()))

        model.fit()


    elif case==2:

        model = Model(dt=0.01,
                      licks=licks_vec, 
                      verbose=True,
                      l2=0.1)
        post_lick_filter = GaussianBasisFilter(num_params = 10,
                                               data = licks_vec.astype(np.float),
                                               dt = model.dt,
                                               duration = 0.21,
                                               sigma = 0.025)
        model.add_filter('post_lick', post_lick_filter)

        reward_filter = GaussianBasisFilter(num_params = 10,
                                            data = rewards_vec.astype(np.float),
                                            dt = model.dt,
                                            duration = 4,
                                            sigma = 0.50)
        model.add_filter('reward', reward_filter)

        model.fit()

    elif case==3:

        model = Model(dt=0.01,
                      licks=licks_vec,
                      verbose=True)
        post_lick_filter = GaussianBasisFilter(num_params = 10,
                                               data = licks_vec,
                                               dt = model.dt,
                                               duration = 0.21,
                                               sigma = 0.025)
        #  model.add_filter('post_lick', post_lick_filter)

        flash_filter = GaussianBasisFilter(num_params = 15,
                                            data = flashes_vec,
                                            dt = model.dt,
                                            duration = 0.76,
                                            sigma = 0.05)
        model.add_filter('flash', flash_filter)

        reward_filter = GaussianBasisFilter(num_params = 20,
                                            data = rewards_vec,
                                            dt = model.dt,
                                            duration = 4,
                                            sigma = 0.25)
        model.add_filter('reward', reward_filter)

        model.fit()

    elif case==4:

        model = Model(dt=0.01,
                      licks=licks_vec)

        change_filter = GaussianBasisFilter(num_params = 30,
                                            data = change_flashes_vec,
                                            dt = model.dt,
                                            duration = 1.6,
                                            sigma = 0.05)
        model.add_filter('change_flash', change_filter)

        post_lick_filter = GaussianBasisFilter(num_params = 10,
                                               data = licks_vec,
                                               dt = model.dt,
                                               duration = 0.21,
                                               sigma = 0.025)
        model.add_filter('post_lick', post_lick_filter)


        model.fit()

        # running_speed_filter = Filter(num_params = 6,
        #                               data = running_speed)
        # model.add_filter('running_speed', running_speed_filter)

    elif case==5:
        import filters
        import importlib; importlib.reload(filters)

        model = Model(dt=0.01,
                      licks=licks_vec, 
                      verbose=True,
                      l2=0.5)

        post_lick_filter = GaussianBasisFilter(data = licks_vec,
                                               dt = model.dt,
                                               **filters.post_lick)
        model.add_filter('post_lick', post_lick_filter)

        reward_filter = GaussianBasisFilter(data = rewards_vec,
                                            dt = model.dt,
                                            **filters.reward)
        model.add_filter('reward', reward_filter)

        flash_filter = GaussianBasisFilter(data = flashes_vec,
                                           dt = model.dt,
                                           **filters.flash)
        model.add_filter('flash', flash_filter)

        change_filter = GaussianBasisFilter(data = change_flashes_vec,
                                            dt = model.dt,
                                            **filters.change)
        model.add_filter('change_flash', change_filter)

        running_speed_filter= GaussianBasisFilter(data = running_speed,
                                                  dt = model.dt,
                                                  **filters.running_speed)
        model.add_filter('running_speed', running_speed_filter)

        acceleration_filter= GaussianBasisFilter(data = running_acceleration,
                                                 dt = model.dt,
                                                 **filters.acceleration)
        model.add_filter('acceleration', acceleration_filter)

        model.fit()

    elif case==6:

        import filters
        import importlib; importlib.reload(filters)

        param_save_fn = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/model_test_71_saved_params.npz'

        model = Model(dt=0.01,
                      licks=licks_vec, 
                      verbose=True,
                      l2=0.5)

        post_lick_filter = GaussianBasisFilter(data = licks_vec,
                                               dt = model.dt,
                                               **filters.post_lick)
        model.add_filter('post_lick', post_lick_filter)

        reward_filter = GaussianBasisFilter(data = rewards_vec,
                                            dt = model.dt,
                                            **filters.reward)
        model.add_filter('reward', reward_filter)

        flash_filter = GaussianBasisFilter(data = flashes_vec,
                                           dt = model.dt,
                                           **filters.flash)
        model.add_filter('flash', flash_filter)

        change_filter = GaussianBasisFilter(data = change_flashes_vec,
                                            dt = model.dt,
                                            **filters.change)
        model.add_filter('change_flash', change_filter)

        running_speed_filter= GaussianBasisFilter(data = running_speed,
                                                  dt = model.dt,
                                                  **filters.running_speed)
        model.add_filter('running_speed', running_speed_filter)

        acceleration_filter= GaussianBasisFilter(data = running_acceleration,
                                                 dt = model.dt,
                                                 **filters.acceleration)
        model.add_filter('acceleration', acceleration_filter)
        # model.set_filter_params_from_file(param_save_fn)
        model.fit()

    elif case==7:

        import filters
        import importlib; importlib.reload(filters)

        model = Model(dt=0.01,
                      licks=licks_vec, 
                      verbose=True,
                      name='{}'.format(experiment_id),
                      l2=0.5)

        #  post_lick_filter = mo.GaussianBasisFilter(data = licks_vec,
        #                                         dt = model.dt,
        #                                         **filters.post_lick)
        #  model.add_filter('post_lick', post_lick_filter)

        long_lick_filter = GaussianBasisFilter(data = licks_vec,
                                               dt = model.dt,
                                               **filters.long_lick)
        model.add_filter('post_lick', long_lick_filter)

        reward_filter = GaussianBasisFilter(data = rewards_vec,
                                            dt = model.dt,
                                            **filters.reward)
        model.add_filter('reward', reward_filter)

        #  long_reward_filter = mo.GaussianBasisFilter(data = rewards_vec,
        #                                      dt = model.dt,
        #                                      **filters.long_reward)
        #  model.add_filter('long_reward', long_reward_filter)

        flash_filter = GaussianBasisFilter(data = flashes_vec,
                                           dt = model.dt,
                                           **filters.flash)
        model.add_filter('flash', flash_filter)

        change_filter = GaussianBasisFilter(data = change_flashes_vec,
                                            dt = model.dt,
                                            **filters.change)
        model.add_filter('change_flash', change_filter)

        model.fit()

    elif case==8:

        model = Model(dt=0.01,
                      verbose=True,
                      licks=licks_vec)

        change_filter = GaussianBasisFilter(num_params = 30,
                                            data = change_flashes_vec,
                                            dt = model.dt,
                                            duration = 1.6,
                                            sigma = 0.05)
        model.add_filter('change_flash', change_filter)

        post_lick_filter = GaussianBasisFilter(num_params = 10,
                                               data = licks_vec,
                                               dt = model.dt,
                                               duration = 0.21,
                                               sigma = 0.025)
        model.add_filter('post_lick', post_lick_filter)

        model.fit()
        nll = model.nll()
        dropout_nll = model.dropout_analysis()

    elif case==9:
        # 3-fold cv
        import filters
        from copy import deepcopy
        import importlib; importlib.reload(filters)


        l2=0.1
        model = Model(dt=0.01,
                      licks=licks_vec,
                      verbose=True,
                      name='{}'.format(l2),
                      l2=l2,
                      n_splits=3)

        long_lick_filter = GaussianBasisFilter(data = licks_vec,
                                               dt = model.dt,
                                               **filters.long_lick)
        model.add_filter('post_lick', long_lick_filter)

        reward_filter = GaussianBasisFilter(data = rewards_vec,
                                            dt = model.dt,
                                            **filters.reward)
        model.add_filter('reward', reward_filter)

        flash_filter = GaussianBasisFilter(data = flashes_vec,
                                           dt = model.dt,
                                           **filters.flash)
        model.add_filter('flash', flash_filter)

        change_filter = GaussianBasisFilter(data = change_flashes_vec,
                                            dt = model.dt,
                                            **filters.change)
        model.add_filter('change_flash', change_filter)

        split_inds = np.arange(model.n_splits)
        models_for_cv = [copy.deepcopy(model) for ind in split_inds]
        training_set_nll = []
        test_set_nll = []
        for test_split_ind in split_inds:
            print("test fold: {}".format(test_split_ind))
            this_model = models_for_cv[test_split_ind]
            test_split = this_model.splits[test_split_ind, :]
            training_split_inds = onp.logical_not(split_inds == test_split_ind)
            training_split = onp.sum(model.splits[training_split_inds, :],
                                     axis=0).astype(bool)
            this_model.fit(training_split=training_split)
            training_set_nll.append(this_model.nll(split=training_split))
            test_set_nll.append(this_model.nll(split=test_split))


    elif case==10:
        import filters
        from copy import deepcopy
        import importlib; importlib.reload(filters)

        # do some cross validation
        l2_to_try = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

        model_list = [Model(dt=0.01,
                            licks=licks_vec,
                            verbose=True,
                            name='{}'.format(l2),
                            l2=l2)
                      for l2 in l2_to_try]

        for model, l2 in zip(model_list, l2_to_try): 

            print("Fitting model with l2: {}".format(l2))
            long_lick_filter = GaussianBasisFilter(data = licks_vec,
                                                   dt = model.dt,
                                                   **filters.long_lick)
            model.add_filter('post_lick', long_lick_filter)

            reward_filter = GaussianBasisFilter(data = rewards_vec,
                                                dt = model.dt,
                                                **filters.reward)
            model.add_filter('reward', reward_filter)

            flash_filter = GaussianBasisFilter(data = flashes_vec,
                                               dt = model.dt,
                                               **filters.flash)
            model.add_filter('flash', flash_filter)

            change_filter = GaussianBasisFilter(data = change_flashes_vec,
                                                dt = model.dt,
                                                **filters.change)
            model.add_filter('change_flash', change_filter)

            model.fit()
            print("\n\n")




'''
def save_model_params(model):
    # db_group = h5py.require_group("/")
    f = h5py.File('model_test.h5', 'a')
    model_grp = f.create_group("model")

    model_attrs = list(model.__dict__.keys())
    model_attrs.remove('filters') # We will save filters separately
    for attr in model_attrs:
        data = getattr(model, attr)
        if not isinstance(data, np.ndarray):
            data = np.array([data])
        dset_attr = model_grp.create_dataset(attr, data, dtype=data.dtype)

    for filter_name, filter in model.filters.items():
        filter_grp = f.create_group(filter_name)

        filter_attrs = list(filter.__dict__.keys())
        for attr in filter_attrs:
            data = getattr(filter, attr)
            dset_attr = filter_grp.create_dataset(attr, data, dtype=data.dtype)

'''