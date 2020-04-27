##########################
# Check if all fits have completed:

VERSION = '10'
train_id_file = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/licking_behavior/scripts/psy_training_ids_v'+VERSION+'.txt'
id_file = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/licking_behavior/scripts/psy_ids_v'+VERSION+'.txt'
directory = '/home/alex.piet/codebase/behavior/psy_fits_v'+VERSION+'/'

with open(train_id_file) as f:
    string_ids = f.read()

string_ids = string_ids.split('\n')
train_session_ids = []
for s in string_ids[0:-1]:
    train_session_ids.append(int(float(s)))

with open(id_file) as f:
    string_ids = f.read()

string_ids = string_ids.split('\n')
ophys_session_ids = []
for s in string_ids[0:-1]:
    ophys_session_ids.append(int(float(s)))

import os
def check_sessions(ids,dirc,TRAIN=False):
    print('The following sessions need to be fit')
    count = 0
    for id in ids:
        if TRAIN:
            filename = dirc+str(id)+'_training.pkl'
        else:
            filename = dirc+str(id)+".pkl"
        if not os.path.isfile(filename):
            print(id)
            count +=1
    return count
    
print('TRAINING')
num_train = check_sessions(train_session_ids,directory, TRAIN=True)
print('\n\n\nOPHYS')
num_ophys = check_sessions(ophys_session_ids,directory)

print('\n\nTRAINING: '+str(num_train))
print('OPHYS: '+str(num_ophys))

