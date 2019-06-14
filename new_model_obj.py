import sys
import numpy as np
import time
import fit_tools
from collections import OrderedDict
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import numba

#  @numba.jit
def loglikelihood(licks_vector, latent,
                  params,l2=0):
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
    # If there are any zeros in the latent model, have to add "machine tolerance"
    latent[latent==0] += np.finfo(float).eps

    # Get the indices of bins with licks
    licksdt = np.flatnonzero(licks_vector)

    NLL = -sum(np.log(latent)[licksdt.astype(int)]) + sum(latent) + l2*np.sum(np.array(params)**2)
    return NLL

class Model(object):

    def __init__(self, dt, licks, l2=0, verbose=False):

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

    def add_filter(self, filter_name, filter):
        '''
        Add a filter to the model. 

        Args: 
            filter_name (str): The filter's name
            filter (instance of Filter, GaussianBasisFilter, etc.)
        '''
        self.filters[filter_name] = filter

    #  @numba.jit
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

    #  @numba.jit
    def calculate_latent(self):
        '''
        Filters own their params and data, so we just call the linear_output
        method on each filter and add up the result
        '''

        base = np.zeros(self.num_time_bins)
        base += self.mean_rate_param # Add in the mean rate

        for filter_name, filter in self.filters.items():
            base += filter.linear_output()

        latent = np.exp(np.clip(base, -700, 700))
        NLL = loglikelihood(self.licks,
                            latent,
                            self.get_filter_params(),
                            self.l2)
        return NLL, latent
    
    def fit(self):

        params = self.get_filter_params()

        sys.stdout.write("Fitting model with {} params\n".format(len(params)))

        # Func to minimize
        def wrapper_func(params):
            self.set_filter_params(params)
            return self.calculate_latent()[0]

        start_time = time.time()
        # TODO: Make this async?

        def print_NLL_callback(xk):
            '''
            A callback for printing info about each iteration.

            Args:
                xk: This is the vector of current params (this is how the 
                    callable is executed per the minimize docs). We don't 
                    use it in this func though.
            '''
            sys.stdout.flush() # This and the \r make it keep writing the same line
            sys.stdout.write('\r')
            NLL, latent = self.calculate_latent()
            self.iteration+=1
            sys.stdout.write("Iteration: {} NLL: {}".format(self.iteration, NLL))
            sys.stdout.flush()

        kwargs = {}
        if self.verbose:
            self.iteration=0
            kwargs.update({'callback':print_NLL_callback})
        res = minimize(wrapper_func, params, **kwargs)
        elapsed_time = time.time() - start_time
        sys.stdout.write('\n')
        sys.stdout.write("Done! Elapsed time: {:02f} sec".format(time.time()-start_time))

        # Set the final version of the params for the filters
        self.set_filter_params(res.x)
        self.res = res

    def eval(self):
        '''
        Evaluate the model, updating the model res with BIC

        '''
        self.res = evaluate_model(self.res, self.calculate_latent,
                                  self.licksdt, self.stop_time)

    def plot_filters(self):
        plt.clf()
        n_filters = len(self.filters)
        for ind_filter, (filter_name, filter_obj) in enumerate(self.filters.items()):
            linear_filt, basis = filter_obj.build_filter()
            plt.subplot(2, (n_filters/2)+1, ind_filter+1)
            plt.plot(filter_obj.filter_time_vec, 
                     np.exp(linear_filt),
                     'k-')
            plt.title(filter_name)
        plt.show()


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
        Doesn't shift the output by default.
        '''
        output = np.convolve(self.data, self.params)[:self.stop_time]

        # Shift prediction forward by one time bin
        output = np.r_[0, output[:-1]]

        return output


spec = [
    ('num_params', numba.int32),
    ('data', numba.float64[:]), ('params', numba.float64[:]),
    ('duration', numba.float32),
    ('dt', numba.float32),
    ('sigma', numba.float32),
    ('filter_time_vec', numba.float64[:]),
    ('initial_params', numba.float64[:]),
    ('x', numba.float64[:]),
    ('mu', numba.float32),
    ('sigma', numba.float32),
    ('mean', numba.float32),
]
#  @numba.jitclass(spec)
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
            basis_funcs[ind_param, :] = this_gaussian

        return filter, basis_funcs

    def linear_output(self):
        filter, _ = self.build_filter()
        output = np.convolve(self.data, filter)[:len(self.data)]

        # Shift prediction forward by one time bin
        output = np.r_[0, output[:-1]]

        return output

def bin_data(data, dt, time_start=None, time_end=None):

    lick_timestamps = data['lick_timestamps']
    running_timestamps = data['running_timestamps']
    running_speed = data['running_speed']
    reward_timestamps = data['reward_timestamps']
    flash_timestamps = data['stim_on_timestamps']

    if time_start is None:
        time_start = 0
    if time_end is None:
        time_end = running_timestamps[-1]

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

    timebase = np.arange(0, time_end-time_start, dt)
    edges = np.arange(0, time_end-time_start+dt, dt)

    licks_vec, _ = np.histogram(lick_timestamps, bins=edges)
    rewards_vec, _ = np.histogram(reward_timestamps, bins=edges)
    flashes_vec, _ = np.histogram(flash_timestamps, bins=edges)
    change_flashes_vec, _ = np.histogram(change_flash_timestamps, bins=edges)

    return (licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
            running_speed, running_timestamps, running_acceleration,
            timebase, time_start, time_end)

if __name__ == "__main__":

    dt = 0.01

    experiment_id = 715887471
    data = fit_tools.get_data(experiment_id, save_dir='./example_data')

    (licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
     running_speed, running_timestamps, running_acceleration, timebase,
     time_start, time_end) = bin_data(data, dt, time_start=300, time_end=1000)

    #  (licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
    #   running_speed, running_timestamps, running_acceleration, timebase,
    #   time_start, time_end) = bin_data(data, dt)

    case=5
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
                                                  **filters.running)
        model.add_filter('running_speed', running_speed_filter)

        acceleration_filter= GaussianBasisFilter(data = running_acceleration,
                                                 dt = model.dt,
                                                 **filters.acceleration)
        model.add_filter('acceleration', acceleration_filter)


        model.fit()

