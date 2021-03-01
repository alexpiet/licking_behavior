import autograd.numpy as np
from autograd.core import primitive
from autograd import grad

def outside_func(x1, x2):
    return np.multiply(x1, x2)

def outside_func_2(x1, x2):
    return x1 * x2

print(grad(outside_func)(1.0, 2.0))
print(grad(outside_func_2)(1.0, 2.0))

class A(object):
    def __init__(self):
        self.x1 = 1.0
        self.x2 = 2.0

    def inside_func(self, x0, x1):
        return x0 * x1

a = A()
print(grad(a.inside_func)(1.0, 2.0))

class B(object):
    def __init__(self, x0, x1):
        self.x0 = x0
        self.x1 = x1

    def hidden_func(self, X):
        return X**2

    def inside_func(self, X):
        return np.sum(self.hidden_func(X))

b = B(1.0, 2.0)
print(grad(b.inside_func)(np.array([0., 1., 2.])))


#### Hacking on the real funcs now. 

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

    def __init__(self, dt, licks, name=None, l2=0, verbose=False):

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

        if name is None:
            self.name='test'
        else:
            self.name=name

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

m = Model(
print(grad
