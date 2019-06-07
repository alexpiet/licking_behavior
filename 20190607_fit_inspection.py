import os
import fit_tools as ft
import importlib
import sys
import matplotlib.pyplot as plt
importlib.reload(ft)

index = int(sys.argv[1])
fit_path = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/cluster_jobs'
experiment_ids = [837729902, 838849930,836910438,840705705,840157581,841601446,840702910,841948542,841951447,842513687,842973730,843519218,846490568,847125577,848697604] # start jobs, one for each of these experiment IDS
id = experiment_ids[index]

Fn = 'glm_model_vba_v2_'+str(id)+'.pkl'

full_path = os.path.join(fit_path, Fn)
print(str(id))
model = ft.Model.from_file_rebuild(full_path)
model.plot_all_filters()

