from collections import OrderedDict

class Model(object):

    def __init__(dt, licks):

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

    def add_filter(filter_name, filter):
        '''
        Add a filter to the model. 

        Args: 
            filter_name (str): The filter's name
            filter (instance of Filter, GaussianBasisFilter, etc.)
        '''
        self.filters[filter_name] = filter

    def initial_params(self):
        pass

    def set_filter_params(self, flat_params):
        '''
        Break up a flat array of params and set them for each filter in the model.
        '''
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
        paramlist = []
        for filter_name, filter in self.filters.items():
            paramlist.append(filter.params)
        return np.concatenate(paramlist)

    def calculate_latent(self):
        '''
        Filters own their params and data, so we just call the linear_output
        method on each filter and add up the result
        '''

        base = np.zeros(self.num_time_bins)
        for filter_name, filter in self.filters.items():
            base += filter.linear_output()

        latent = np.exp(np.clip(base, -700, 700))
        NLL = loglikelihood(self.licks, latent, self.get_filter_params, self.l2)

        return NLL, latent
    
    def fit(self):
        print("Fitting model with {} params".format(len(self.initial_params)))

        # Func to minimize
        def wrapper_func(params):
            self.set_filter_params(params)
            return self.calculate_latent()[0]

        start_time = time.time()
        # TODO: Make this async?
        res = minimize(wrapper_func, self.initial_params)
        elapsed_time = time.time() - start_time
        print("Done! Elapsed time: {:02f} sec".format(time.time()-start_time))

        # Set the final version of the params for the filters
        self.set_filter_params(res.x)
        self.res = res

    def eval(self):
        '''
        Evaluate the model, updating the model res with BIC

        '''
        self.res = evaluate_model(self.res, self.calculate_latent,
                                  self.licksdt, self.stop_time)

# Can we do the convolve func with a vec of 0/1 instead of rolling our own?
# so linear_whatever funcs can use the same thing.

class Filter(object):

    def __init__(num_params, data, initial_params=None):
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


    def linear_output(self):
        '''
        This base class just convolves the filter params with the data.
        Cuts the output to be the same length as the data
        Doesn't shift the output by default.
        '''
        output = np.convolve(self.data, self.params)[:self.stop_time]
        return output


class GaussianBasisFilter(Filter):
    def __init__(num_params, data, dt, duration, sigma):
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
        # Run the Filter init
        super(GaussianBasisFilter, self).__init__(num_params, data)

        self.duration = duration
        self.dt = dt
        self.sigma = sigma
        self.filter_time_vec = np.arange(dt, duration, dt)

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
        output = np.convolve(self.data, filter)[:self.stop_time]
        return output


if __name__ == "__main__":

    licks = np.array([0, 0, 0, 1, 0, 0])

    # TODO: Default mean rate filter? 
    model = Model(dt=0.01,
                  licks=licks)

    post_lick_filter = GaussianBasisFilter(num_params = 10,
                                           data = licks,
                                           dt = model.dt,
                                           duration = 0.21,
                                           sigma = 0.025)
    model.add_filter('post_lick', post_lick_filter)

    # running_speed_filter = Filter(num_params = 6,
    #                               data = running_speed)
    # model.add_filter('running_speed', running_speed_filter)

    model.fit()





