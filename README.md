# licking_behavior
Analysis of mouse licking behavior during visually guided behavior

Initial Project Outline: https://docs.google.com/document/d/1Skvk_tj9a2nwtIRatTJ3y-bC0qrIAdU7wKAFgkQsnoE/edit

This repo consists of two models of mouse behavior. 

1. Poisson GLM that characterizes the licking probability within 10msec time bins by learning temporal filters that map external events onto licking probability.

2. A time-varying logistic regression model that learns the probability of licking on a flash by flash basis, using weights that vary over time by following random walk priors. 


## Fitting the Time Varying Regression model
> import src/psy_tools as ps  
> for ID in IDS:  
>    ps.process_session(ID)  
>    ps.plot_fit(ID)  
> ps.plot_session_summary(IDS)

## Integrating the Time Varying Regression Model clustering with the flash_response_df
> import src/psy_tools as ps
> import src/psy_sdk_tools as psd
> from allensdk.brain_observatory.behavior.swdb import behavior_project_cache as bpc
> import allensdk.brain_observatory.behavior.swdb.utilities as tools
> cache = bpc.BehaviorProjectCache(cache_json)
> fit = ps.load_fit(id)
> session = cache.get_session(id)
> cdf = psd.get_joint_table(fit,session)

