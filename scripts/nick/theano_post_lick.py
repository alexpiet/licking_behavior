import theano
import theano.tensor as T
from theano.tensor.signal.conv import conv2d
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np

### fake some data
rate = 0.2
np.random.seed(123)
num_time_bins = 1000
licks = (np.random.rand(num_time_bins) < rate).astype(int)

# 1d convolution
conv2d = T.signal.conv.conv2d
x = T.dmatrix('x')
y = T.dmatrix('y')
veclen = x.shape[1]
conv1d_expr = conv2d(x, y, image_shape=(1, veclen), border_mode='full')
conv1d = theano.function([x, y], outputs=conv1d_expr)

# calculating latent rate for mean-rate model
mean_rate_param = theano.shared(np.array(-0.5), 'mean_rate_param')
#  mean_rate_param = T.scalar('mean_rate_param')

num_bins_t = T.constant(num_time_bins, 'num_time_bins')
#  latent_expr = T.exp(T.clip((T.zeros(num_bins_t) + mean_rate_param), -700, 700))
#  latent = theano.function([], latent_expr)

# loglikelihood
licks_vector = theano.shared(licks, 'licks_vec')

#  nll_expr = -1 * T.sum(T.log(latent()+1e-99)[T.flatnonzero(licks_vector).astype('int64')]) + T.sum(latent())

# Reconstructing this to take the param directly instead

def latent(mean_rate_param):
    
    mean_rate_component = T.zeros(num_bins_t) + mean_rate_param
    some_other_component = T.zeros(num_bins_t)

    component_list = []
    component_list.append(mean_rate_component)
    component_list.append(some_other_component)

    component_sum = T.sum(T.stack(component_list, axis=1), axis=1)
    #  return T.exp(T.clip(mean_rate_component, -700, 700))
    return T.exp(T.clip(component_sum, -700, 700))


#TODO: Now just need to define latent(params) and code up some more params

nll_expr = (-1 * T.sum(T.log(latent(mean_rate_param))[T.flatnonzero(licks_vector).astype('int64')]))+ T.sum(latent(mean_rate_param))

gradients = T.grad(nll_expr, [mean_rate_param])

rate_update = mean_rate_param - (1e-3 * gradients[0])
updates = [(mean_rate_param, rate_update)]

nll = theano.function([], nll_expr, allow_input_downcast=True, updates=updates)

for i in range(100):
    output = nll()
    print(output)
    print(mean_rate_param.get_value())
    print('')

print("Recovered mean rate: {}".format(np.exp(mean_rate_param.get_value())))
print("True mean rate: {}".format(licks.sum() / len(licks)))




