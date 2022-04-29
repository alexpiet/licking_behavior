import psy_tools as ps
import psy_general_tools as pgt
import psy_timing_tools as pt
import psy_metrics_tools as pm
 
## Basic SDK
###########################################################################################
bsid = 914705301
training = pgt.get_training_manifest()

test = training.drop_duplicates(keep='first',subset=['session_type'])
test = test[test.session_type.str.startswith('TRAINING')]
test = test.sort_values(by=['session_type'])


## PCA
###########################################################################################
drop_dex,drop_var = ps.PCA_dropout(ids,pgt.get_mice_ids(),version)
weight_dex  = ps.PCA_weights(ids,pgt.get_mice_ids(),version)
ps.PCA_analysis(ids, pgt.get_mice_ids(),version)

## Clustering
###########################################################################################
# Get unified clusters
ps.build_all_clusters(pgt.get_active_ids(), save_results=True)

