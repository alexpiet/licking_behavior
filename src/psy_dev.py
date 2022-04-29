
## PCA
###########################################################################################
drop_dex,drop_var = ps.PCA_dropout(ids,pgt.get_mice_ids(),version)
weight_dex  = ps.PCA_weights(ids,pgt.get_mice_ids(),version)
ps.PCA_analysis(ids, pgt.get_mice_ids(),version)

