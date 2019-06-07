import os
import fit_tools as ft
import importlib
importlib.reload(ft)

fit_path = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/cluster_jobs'

Fn = 'glm_model_vba_v2_836910438.pkl'

full_path = os.path.join(fit_path, Fn)

model = ft.Model.from_file_rebuild(full_path)
