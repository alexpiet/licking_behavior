import theano
import theano.tensor as T
from theano.tensor.signal.conv import conv2d
from matplotlib import pyplot as plt
from scipy import signal
import sys; sys.path.append('/home/nick/tmp')
from licking_behavior.src import fit_tools
from licking_behavior.src import licking_model as mo
import numpy as np

### fake some data
#  rate = 0.2
#  np.random.seed(123)
#  num_time_bins = 1000
#  licks = (np.random.rand(num_time_bins) < rate).astype(int)
#  licks_float = licks.astype(float)

# Use real data!!
dt = 0.01
experiment_id = 715887471
data = fit_tools.get_data(experiment_id, save_dir='../../example_data')

#  (licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
#   running_speed, running_timestamps, running_acceleration, timebase,
#   time_start, time_end) = mo.bin_data(data, dt, time_start=300, time_end=1000)

(licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
 running_speed, running_timestamps, running_acceleration, timebase,
 time_start, time_end) = mo.bin_data(data, dt)

licks = licks_vec.astype(int)
licks_float = licks_vec.astype(float)
num_time_bins = licks_vec.shape[0]


conv2d = T.signal.conv.conv2d
def conv1d(x, y):
    veclen = x.shape[1]
    return conv2d(x[:,np.newaxis], y[:,np.newaxis],
                  image_shape=(1, veclen), border_mode='full').ravel()

#  conv1d_expr = conv2d(x, y, image_shape=(1, veclen), border_mode='full')
#  conv1d = theano.function([x, y], outputs=conv1d_expr)

mean_rate_param = theano.shared(np.array(-0.5), 'mean_rate_param')
params = theano.shared(np.zeros(201), 'params')
num_bins_t = T.constant(num_time_bins, 'num_time_bins')

licks_vector = theano.shared(licks, 'licks_vec')
licks_vector_float = theano.shared(licks_float, 'licks_vec_float')

def latent(params, licks):
    
    mean_rate_param = params[0]
    post_lick_params = params[1:]

    mean_rate_component = T.zeros(num_bins_t) + mean_rate_param
    post_lick_component = conv1d(licks_vector_float, post_lick_params)[:num_time_bins]

    post_lick_component_rolled = T.concatenate([T.zeros(1),
                                                 post_lick_component[:-1]])
    
    component_list = []
    component_list.append(mean_rate_component)
    component_list.append(post_lick_component_rolled)
    #  
    component_sum = T.sum(T.stack(component_list, axis=1), axis=1)
    #  return T.exp(T.clip(mean_rate_component, -700, 700))
    return T.exp(T.clip(component_sum, -700, 700))


l2 = 0.5
learning_rate = 1e-4

nll_expr = (-1 * T.sum(T.log(latent(params, licks_vector))[T.flatnonzero(licks_vector).astype('int64')]))+ T.sum(latent(params, licks_vector)) + l2*T.sum(params**2)

gradients = T.grad(nll_expr, params)

param_update = params - (learning_rate * gradients)
updates = [(params, param_update)]

nll = theano.function([], nll_expr, allow_input_downcast=True, updates=updates)

for i in range(1000):
    output = nll()
    print(output)
    #  print(mean_rate_param.get_value())
    print('')

print("Recovered mean rate: {}".format(np.exp(params.get_value()[0])))
print("True mean rate: {}".format(licks.sum() / len(licks)))






