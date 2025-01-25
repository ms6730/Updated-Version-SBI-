import os
import sys      # I added this library, since I am using one of its function for the json path
import pickle
import numpy as np
import pandas as pd
import subsettools
from sbi.inference import SNPE
from sbi.utils import get_density_thresholder, RestrictedPrior
from pf_ens_functions import get_parflow_output_nc
import torch
import json
import random
import matplotlib.pyplot as plt

#read in variables from the json file
json_path = sys.argv[1]                 # I decided to use this function to get the json path needed for this script
with open(json_path, 'r') as file:
    settings = json.load(file)
    
base_dir = settings['base_dir']
grid = settings['grid']
huc = settings['huc']
temporal_resolution = settings['temporal_resolution']
runname=settings['runname']
variable_list = settings['variable_list']
num_sims = settings['num_sims']
ens_num=settings['ens_num']
num_samples = settings['num_samples']
quantile = settings['quantile']
obsv_path=settings['observation_path']
seed=settings['random_seed']
metadata_path=f'{base_dir}/outputs/{runname}/streamflow_daily_metadf.csv'
orig_vals_path = f"{base_dir}/{runname}_filtered_orig_vals.csv"
filtered_df=pd.read_csv(orig_vals_path)

#set the random seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


ij_bounds, mask = subsettools.define_huc_domain([huc], grid)

# Evaluate
for sim in range(0,num_sims):
    
    parflow_output_dir=f"{base_dir}/outputs/{runname}_{ens_num}_{sim}"
    nc_path = f"{parflow_output_dir}/{runname}_{sim}.nc"
    write_path=f"{parflow_output_dir}/{variable_list[0]}_{temporal_resolution}_pfsim.csv"
    
    for variable in variable_list:  
        # Get ParFlow outputs matching site locations
        parflow_data_df = get_parflow_output_nc(nc_path, metadata_path, variable, write_path)

##### SBI #####

# try loading existing inference structure
# if not there, create new one from prior
try:
    with open(f"{base_dir}/{runname}_inference.pkl", "rb") as fp:
        inference=pickle.load(fp)
except FileNotFoundError:
    with open(f"{base_dir}/{runname}_prior.pkl", "rb") as fp:
        prior = pickle.load(fp)
    inference = SNPE(prior=prior)

# get parameters for last ensemble run
theta_sim = pd.read_csv(f"{base_dir}/{runname}_parameters_ens{ens_num}.csv")
noise_param_col = theta_sim['noise_param']
noise_param = torch.tensor(noise_param_col.values, dtype=torch.float)
theta_sim = torch.tensor(theta_sim.values, dtype=torch.float)

# create 1D torch tensors for observed and simulated outputs
sim_data = []

for i in range(0,num_sims):
    sim_df = pd.read_csv(f'{base_dir}/outputs/{runname}_{ens_num}_{i}/streamflow_daily_pfsim.csv').drop('date', axis=1)
    sim_df = sim_df[5:]#dropping first 5 days from evaluation for spinup
    if i == 0:
        obsv_df = pd.read_csv(obsv_path).drop('date', axis=1)
        obsv_df = obsv_df[5:]#dropping first 5 days
        common_columns = sim_df.columns.intersection(obsv_df.columns)
        obsv_df = obsv_df[common_columns]
        obsv_tensor = torch.tensor(obsv_df.values, dtype=torch.float)
        obsv_flat = torch.flatten(obsv_tensor)
        x_obs = obsv_flat.unsqueeze(0)               # this function has been used since it is more simpler 

    sim_df = sim_df[common_columns]
    sim_tensor = torch.tensor(sim_df.values, dtype=torch.float)
    sim_flat = torch.flatten(sim_tensor)
    current_noise_param = noise_param[i]        # there is bugs here in this part of the code 
                                                # after checking the noise_parameters values from the csv files and assessing the size of 
                                                # the noise parameters which consists of 100 values where one value of noise per simulation
                                                # to be able to determine the noise parameter value for this current simulation, we were able to
                                                # access it using `i` which represents the data for the current simulation 
                                                # the error that was before adding this is mismatcg in sizes where the size of the simulated                                                        # data is around 2160 and the size of the noise parameters were 100 which corresponds to the
                                                # total number of simulations 
    sim_flat += torch.randn(sim_flat.shape) * (sim_flat * current_noise_param)  # this line has been also modified
    sim_data.append(sim_flat)

x_sim = torch.stack(sim_data, dim=0)

# update posterior with new simulations
_ = inference.append_simulations(theta_sim, x_sim).train(force_first_round_loss=True)
posterior = inference.build_posterior().set_default_x(x_obs)

#make plots from sampling the proposal 
plots_dir = f'{base_dir}/plots'                      # I added these two lines to make the code more flexible, since in my case I did not have this                                                      # folder and it let to an error when executing the code 
os.makedirs(plots_dir, exist_ok=True)                # this line is to ensure that the folder exists and in case it does not exist, it will create 
                                                     # the directory 
samples = posterior.sample((1000,))
num_params = filtered_df.shape[1]                    # this line has been modified where the num_params should be equal to filtered_df.shape[1]
                                                     # instead of samples.shape[1] since the samples that does comes from the theta_sim file
                                                     # have one addtional column which is for the noise_parameter 
for i in range(num_params):
    plt.figure(figsize=(8, 6))
    plt.hist(samples[:, i].numpy(), bins=30, density=True, alpha=0.6, color='b')
    plt.axvline(x=filtered_df.iloc[0, i], color='r', linestyle='--', label='True Value')
    plt.title(f'Density Plot for Parameter {i}')
    plt.xlabel(f'Parameter {i}')
    plt.ylabel('Density')
    plt.savefig(f'{base_dir}/plots/param{i}_posterior_density_ens{ens_num}.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()
    
# update inference and posterior
filename = f"{base_dir}/{runname}_inference_{ens_num}.pkl"
with open(filename, "wb") as fp:
    pickle.dump(inference, fp)
print("pickled inference")
filename = f"{base_dir}/{runname}_posterior_{ens_num}.pkl"
with open(filename, "wb") as fp:
    pickle.dump(posterior, fp)
print("pickled posterior")

#updating the ensemble number in the json file 
next_ens = ens_num+1
settings['ens_num']=next_ens
with open(json_path, 'w') as file:
    json.dump(settings, file, indent=4) 
