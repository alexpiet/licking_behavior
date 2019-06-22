import psy_tools as ps

#experiment_id = 787498309
experiment_id = 842975542
session = ps.get_data(experiment_id)
psydata = ps.format_session(session)

hyp, evd, wMode, hess = ps.fit_weights(psydata)
plt.figure(figsize=(10,5))
plt.plot(wMode[0,:], linestyle="-", lw=3, color="blue")
plt.plot(wMode[1,:], linestyle="-", lw=3, color="red")
plt.plot([0,5000], [0,0], 'k--')



