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
# calculate drop-out Delta-evidence for each feature
# fit full training

# make fake data with different strategies: 
#   change bias/task ratio
#   bias(-5:1:5) X task(-5:1:5)
# do bootstrapping to recover fitted params 
# examine effects of hyper-params
# should we regress separately on hits/miss vs CRs/FA?

# my issues
# figure out how to import alex_utils
# make whos() that shows size of all dictionary elements
# Document that the aborted classification misses trials with dropped frames

# SDK GITHUB ISSUES
# CR and aborted trials
# aborted trials with no licks
# reward times is an array, shouldnt it be a float?
# should trial start/stop times be aligned to flash times?
# should trial stop = trial start +1?
# from stimulus table, I should be able to know what trial I was a part of
# from trial table, I should be able to know what stimuli were part of my trial


