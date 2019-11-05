import psy_tools as ps

ids = ps.get_active_ids()
counts = []
for id in ids:
    print(id)
    session = ps.get_data(id)
    session.trials['pre-post'] = session.trials['initial_image_name'].values + session.trials['change_image_name'].values
    unique = session.trials['pre-post'].unique()
    for transition in unique:
        count = len(session.trials[session.trials['pre-post'] == transition])
        counts.append(count)


plt.figure()
plt.hist(counts,np.array(range(0,8))+0.5)
plt.ylabel('count')
plt.xlabel('Repetitions per Session')
plt.xticks([1,2,3,4,5,6,7])
plt.title('Active sessions')
plt.tight_layout()


