import psy_tools as ps
import psy_timing_tools as pt
import psy_metrics_tools as pm
import matplotlib.pyplot as plt
import psy_cluster as pc
from alex_utils import *
from importlib import reload
plt.ion()

# Build timing regressor
dir="/home/alex.piet/codebase/behavior/psy_fits_v4/"
all_weights = ps.plot_session_summary_weights(ps.get_session_ids(),return_weights=True,directory=dir)
w = np.vstack(all_weights)
timing = w[:,3:]
mean_timing = np.mean(timing[1:,:],0)
mean_timing = np.concatenate([mean_timing[0:1], mean_timing[2:], mean_timing[1:2]])

def sigmoid(x,a,b,c,d):
    y = d+(a-d)/(1+(x/c)**b)
    return y

from scipy.optimize import curve_fit
x = np.arange(1,len(mean_timing)+1)
y = mean_timing
popt,pcov = curve_fit(sigmoid, x,y,p0=[0,1,1,-3.5])

plt.figure()
plt.plot(x,mean_timing,'ko')
plt.plot(x,sigmoid(x,popt[0],popt[1],popt[2],popt[3]),'r')
plt.plot(x,sigmoid(x,0,popt[1],popt[2],popt[3]-popt[0]),'b')
plt.gca().axhline(0,color='k',linestyle='--')
plt.ylabel('Regressor Amplitude')
plt.xlabel('Flashes since last lick')

#dex = (np.max(timing,1)<5)&(np.min(timing,1) > -7)
#x_all = np.tile(np.array([1,10,2,3,4,5,6,7,8,9]),(np.shape(timing[dex,:])[0],1)).reshape(-1)
#y_all = timing[dex,:].reshape(-1)
#all_popt,all_pcov = curve_fit(sigmoid, x_all,y_all,p0=[0,1,1,-3.5])
#plt.plot(x,sigmoid(x,all_popt[0],all_popt[1],all_popt[2],all_popt[3]),'g')
#plt.plot(x,sigmoid(x,0,all_popt[1],all_popt[2],all_popt[3]-all_popt[0]),'g')
#plt.plot(x,sigmoid(x,0,-5,4,-2),'g')

df = ps.get_all_timing_index(ps.get_session_ids(),dir)
b=[]
c= []
index = []
timing_index = []
for i in np.arange(0,np.shape(timing)[0]):
    try:
        session = ps.get_data(df.iloc[i].id.astype(int))
        y = timing[i,:]
        x = np.array([1,10,2,3,4,5,6,7,8,9])
        x_popt,x_pcov = curve_fit(sigmoid, x,y,p0=[0,1,1,-3.5])
        if (len(session.rewards) > 50) & (x_popt[2] > 0.001):
            b.append(x_popt[1])
            c.append(x_popt[2])
            index.append(df.iloc[i]['Timing/Task Index'])
            timing_index.append(df.iloc[i]['timingdex'])
    except:
        print('crash')
        pass



plt.figure()
plt.plot(np.abs(np.array(b)),c,'ko')
plt.plot(np.abs(popt[1]),popt[2],'ro')
plt.gca().axvline(1,alpha=0.3)
plt.xlim(0,20)
plt.ylim(bottom=0)
plt.ylabel('Intercept')
plt.xlabel('Slope Parameter')



plt.figure()
plt.plot(timing_index,c,'ko')
plt.ylabel('Intercept')
plt.xlabel('Timing Dropout')

