import psy_general_tools as pgt
import matplotlib.pyplot as plt
import seaborn as sns
plt.ion()

ids = pgt.get_active_ids()
counts = []
numtrials = []
triallength = []
for id in ids:
    print(id)
    session = pgt.get_data(id)
    session.trials['pre-post'] = session.trials['initial_image_name'].values + session.trials['change_image_name'].values
    unique = session.trials['pre-post'].unique()
    numtrials.append(np.max(session.trials.index.values))
    triallength.append(session.trials['trial_length'].values)
    for transition in unique:
        count = len(session.trials[session.trials['pre-post'] == transition])
        counts.append(count)

sns.set_context('notebook', font_scale=1, rc={'lines.markeredgewidth': 2})
sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False, 'xtick.bottom': False, 'ytick.left': False,})

plt.figure()
hcounts, edges =  np.histogram(counts,np.array(range(0,8))+0.5)
centers = edges[0:-1] + np.diff(edges)/2
plt.bar(centers, hcounts/np.sum(hcounts),color='k',width=.9)
plt.ylabel('% percentage of pairs')
plt.xlabel('Image pair repetitions per session')
plt.xticks([1,2,3,4,5,6,7])
plt.title('Active sessions')
plt.tight_layout()

plt.figure()
plt.hist(numtrials,20,color='k')
plt.ylabel('# of sessions')
plt.xlabel('Trials per session')
plt.tight_layout()


plt.figure()
plt.hist(np.concatenate(triallength),50)
plt.ylabel('count')
plt.xlabel('Trial Duration (s)')
plt.tight_layout()




