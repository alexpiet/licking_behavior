import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def on_key_press(event):
    '''
    Method to listen for keypresses and flip to next/previous view
    '''
    xStep = 1
    if event.key=='<' or event.key==',' or event.key=='left':
        
        xMin -= xStep
        xMax -= xStep
        
        redraw(ax, xMin, xMax)

    elif event.key=='>' or event.key=='.' or event.key=='right':
        xMin -= xStep
        xMax -= xStep

        redraw(ax, xMin, xMax)

def redraw(ax, xMin, xMax):
    ax.set_xlim([xmin, xMax])
    plt.draw()

fig = plt.figure()
ax = plt.subplot(111)
kpid = fig.canvas.mpl_connect('key_press_event', on_key_press)

xMin = 0
xMax = 20
ax.plot(np.arange(20))
ax.set_xlim([xMin, xMax])
plt.draw()
plt.show()


 
class FlipThrough(object):
    def __init__(self, plotter, data):
        '''
        Allow flipping through data plots.
        Args:
            plotter (function): function that plots data.
            data (list): list of tuples containing the parameters for the plotter function.
        '''
        self.data = data
        self.counter = 0
        self.maxDataInd = len(data)-1
        self.fig = plt.gcf()
        self.redraw() # Plot data
        self.kpid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def redraw(self):
        self.fig.clf()
        if isinstance(self.data[self.counter], tuple):
            # FIXME: this will fail if the function requires a tuple as input
            self.plotter(*self.data[self.counter])
        else:
            self.plotter(self.data[self.counter])
        plt.suptitle('{}/{}: Press < or > to flip through data'.format(self.counter+1,
                                                                       self.maxDataInd+1))
        self.fig.show()

    def on_key_press(self, event):
        '''
        Method to listen for keypresses and flip to next/previous view
        '''
        if event.key=='<' or event.key==',' or event.key=='left':
            if self.counter>0:
                self.counter -= 1
            else:
                self.counter = self.maxDataInd
            self.redraw()
        elif event.key=='>' or event.key=='.' or event.key=='right':
            if self.counter<self.maxDataInd:
                self.counter += 1
            else:
                self.counter = 0
            self.redraw()
