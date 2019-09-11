from matplotlib import pyplot as plt
from scipy import signal
import numpy as np

np.random.seed(1)

impulse = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

timebase = np.zeros(100)
event_inds = np.random.randint(0, 100, size=20)

timebase[event_inds] += 1

result = np.convolve(timebase, impulse)
s_result = signal.convolve(timebase, impulse)

def convolve_with_events(event_vec, impulse):
    output = np.zeros(event_vec.shape[0] + impulse.shape[0])
    event_inds = np.flatnonzero(event_vec)
    for i in event_inds:
        output[i:i+len(impulse)] += impulse
    output = output[0:event_vec.shape[0]]
    return output

# Can't assign in place. So what can we do? 
def convolve_with_events_2(event_vec, impulse):
    output = np.zeros(event_vec.shape[0] + impulse.shape[0])
    event_inds = np.flatnonzero(event_vec)

    intermediate = np.concatenate([impulse, np.zeros(event_vec.shape[0])])
    intermediate_2 = np.tile(intermediate, (len(event_inds), 1))

    roll_arr = event_inds

    A = intermediate_2
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]
    # Use always a negative shift, so that column_indices are valid.
    # (could also use module operation)
    roll_arr[roll_arr < 0] += A.shape[1]
    column_indices = column_indices - roll_arr[:,np.newaxis]
    result = A[rows, column_indices]
    
    output = np.sum(result, axis=0)
    return output


my_result = convolve_with_events(timebase, impulse)
wont_work = convolve_with_events_2(timebase, impulse)

plt.clf()
plt.plot(timebase)
plt.plot(result[:len(timebase)])
plt.plot(s_result[:len(timebase)])
plt.plot(wont_work[:len(timebase)])
plt.plot(my_result)
plt.show()

