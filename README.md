# licking_behavior
Analysis of mouse licking behavior during visually guided behavior. Primarily, this repo develops a time-varying logistic regression model that learns the probability of licking on a flash by flash basis, using weights that vary over time by following random walk priors. 

## Time Varying Regression Model

The model predicts the probability of the mouse starting a licking bout on each image presentation. Its described as the sum of several time-varying strategies. 

- Bias, is a strategy that wants to lick on every image
- Visual/Task0, is a strategy that only wants to lick on the image-changes
- Timing1D, is a strategy that wants to lick every 4-5 images after the end of the last licking bout
- Omission0, is a strategy that wants to lick on every omission
- Omission1, is a strategy that wants to lick on the image after every omission

### Check if the Time Varying Regression model has already been fit to a session
> import src/psy_tools as ps  
> ps.check_session(id)

### Fitting the Time Varying Regression model
> import src/psy_tools as ps  
> for ID in IDS:  
>    ps.process_session(ID)  
>    ps.plot_fit(ID)  
> ps.plot_session_summary(IDS)

### Integrating the Time Varying Regression Model clustering with the flash_response_df
> import src/psy_tools as ps  
> import src/psy_sdk_tools as psd  
> from allensdk.brain_observatory.behavior.swdb import behavior_project_cache as bpc  
> import allensdk.brain_observatory.behavior.swdb.utilities as tools  
> cache = bpc.BehaviorProjectCache(cache_json)  
> fit = ps.load_fit(id)  
> session = cache.get_session(id)  
> cdf = psd.get_joint_table(fit,session)  

### Diagram of information flow
![code_diagram](https://user-images.githubusercontent.com/7605170/175404261-4565ab0a-2c82-4215-9840-dffb2b736883.png)

