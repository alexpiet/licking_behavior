import psy_tools as ps
import matplotlib.pyplot as plt
plt.ion()

IDS = [ 787498309, 796105823, 783927872,
        832117336, 842975542, 848006184,
        862023618, 867337243] 

experiment_id = IDS[5]
session = ps.get_data(experiment_id)
psydata = ps.format_session(session)
hyp, evd, wMode, hess, credibleInt,weights = ps.fit_weights(psydata)
ypred = ps.compute_ypred(psydata, wMode,weights)
ps.plot_weights(session,wMode, weights,psydata,errorbar=credibleInt, ypred = ypred)

ps.check_lick_alignment(session,psydata)

## TODO
# add rolling hit rate, false alarm, correct reject, miss
# (1) scatter plot of trials types needs to filter out consumption flashes
# calculate drop-out Delta-evidence for each feature

# make fake data with different strategies: 
#   change bias/task ratio
#   bias(-5:1:5) X task(-5:1:5)
# do bootstrapping to recover fitted params 
# examine effects of hyper-params
# should we regress separately on hits/miss vs CRs/FA?

# SDK GITHUB ISSUES
# should it be possible to be a CR and aborted trials?
# should it be possible to have aborted trials with no licks?
# reward times is an array, shouldnt it be a float?
# from stimulus table, I should be able to know what trial I was a part of
# from trial table, I should be able to know what stimuli were part of my trial

from psytrack.helper.helperFunctions import read_input
g = read_input(psydata, weights)
gw = np.sum(g*wMode.T,axis=1)
pR = 1/(1+np.exp(-gw))


