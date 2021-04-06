import matplotlib.pyplot as plt


def RT_by_engagement(ophys,version=None):
    plt.figure()
    plt.hist(ophys['RT_engaged'], color='k',alpha=.5,label='Engaged')
    plt.hist(ophys['RT_disengaged'], color='r',alpha=.5,label='Disengaged')
    plt.ylabel('count')
    plt.xlabel('Session average RT')



