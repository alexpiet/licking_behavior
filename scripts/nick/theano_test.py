import theano
import theano.tensor as T
from theano.tensor.signal.conv import conv2d
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np

# need to make the theano 1d convolution function
conv2d = T.signal.conv.conv2d
x = T.dmatrix('x')
y = T.dmatrix('y')
veclen = x.shape[1]
conv1d_expr = conv2d(x, y, image_shape=(1, veclen), border_mode='full')
conv1d = theano.function([x, y], outputs=conv1d_expr)


np.random.seed(1)
impulse = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
timebase = np.zeros(100)
event_inds = np.random.randint(0, 100, size=20)
timebase[event_inds] += 1

output = conv1d(timebase[:,np.newaxis], impulse[:,np.newaxis]).ravel()
np_output = np.convolve(timebase, impulse)

plt.clf()
plt.plot(output)
plt.plot(np_output+0.1)
plt.show()


