# SBI-ML-Updated-Version

### Create Virtual Environment 
- Open the terminal on Verdee  
- `conda --version`  to check the version of the conda   
- `conda create -n sbi_env` to create new environment named sbi_env  
-  `conda init bash` to ensure that the conda command is available in your terminal without requiring you to manually source Conda's activation script  
- Close the terminal and open new terminal to refresh it  
- `conda activate sbi_env` to activate the new environment created
- Save `requirement.txt` in your working directory
- Install all dependencies specified in `requirement.txt` file using `pip install -r requirements.txt`
- Verify the installation  using `pip list`
- After installing the packages from requirement.txt file you still need to install the packages from conda 
- `conda install -c conda-forge python=3.10 pip`
- `conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia`
- Verify the installation using `conda list` command


### **`setup_experiment.py`**
This is the first script that you need to run. It is the baseline simulation that will be considered as the truth. A baseline simulation refers to the initial simulation which uses assumed values for input parameters (e.g: Manning's coefficient, terrain slope ...). The baseline simulation involves one single run unlike ensemble simulation which involves multiple runs.
#### Modifications that needs to be done before running `setup_experiment.sh`
- Open `settings.JSON` file 
  - Basic directory in `settings.JSON` needs to be changed 
  -  Save your changes 
- Open `setup_experiment.sh`:
  - Change the path of the output
  - Change the path of the error
  - Change the mail user
  - Change the `JSON PATH` 
  - Change the name of the environment in `conda activate sbi_new` , include the name of the environment that you created to run this project 
  - Save your changes
- Open `setup_experiment.py`:
  - Change the path in **line 15**
  - Save your changes
#### To run `setup_experiment.py`, you need to run ` setup_experiment.sh`
- open your verdee terminal 
  - locate `setup_experiment.sh`
  - use this command `ls -l` to display detailed information about the files. It displays information about the permission as well ad about the type of the file  
  - write this command on your terminal `chmod +x setup_experiment.sh` used to change the permissions of a file to make it executable
  - use this command `ls -l` to display detailed information about the files and make sure that it has become executable
  - make sure that you are inside the directory where the setup_experiment.sh is uploaded 
  - write `sbatch setup_experiment.sh` on your terminal to start by the execution of the script
#### Running ` setup_experiment.py` for comparative analysis
This part has nothing to do with running the other scripts. It can be done after running all your scripts since it is related to a part in `analyse_ensembles.ipynb`
To run it : 
- Open ` new_press.json` 
- Change the `base_dir` in the `new_press_json`
- Save your changes
- Open `setup_experiment.sh`:
  - Change the json file to the **`new_press JSON file`** and modify the path
  - Save your changes 
- Open your verdee terminal:
  - Locate `setup_experiment.sh`
  - Write `sbatch setup_experiment.sh` on your terminal to start by the execution of the script


### `create_ensemble.py`
This is the second step that needs to be done after running `setup_experiment.py`. This script is not running anything. It is just sample from the distribution, and pass them as new row. It is important to note that that it is just gets the original mamnning's map that we have then it will go through.    
**Remark:** Try to add `print` statment when you want to debug
#### Modifications that needs to be done before running `create_ensemble.py`
- Open `setup_experiment.sh`:
  - Change the path of the output
  - Change the path of the error
  - Change the mail user
  - Change the `JSON PATH` 
  - Change the name of the environment in `conda activate sbi_new` , include the name of the environment that you created to run this project 
  - Save your changes
#### To run `create_ensemble.py`, you need to run `create_ensemble.sh`
- Open `create_ensemble.sh`:
  - Change the path of the output
  - Change the path of the error
  - Change the mail user
  - Change the json path 
  - Change the name of the environment in `conda activate sbi_new` , include the name of the environment that you created to run this project 
  - Save your changes
  - Open your verdee terminal
  - Write on you terminal the following command: `sbatch create_ensemble.sh`


### ` run_ensemble.py`
This is the third step that needs to be done after running `create_ensemble.py`. This script executes multiple simulation runs for the ensemble created when running `create_ensemble.py`.
**Remark:** Try to add `print` statment when you want to debug
#### Modifications that needs to be done before running `run_ensemble.py`
- Open `run_ensemble.sh`:
  - Change the path of the output
  - Change the path of the error
  - Change the mail user
  - Change the `JSON PATH` 
  - Change the name of the environment in `conda activate sbi_new` , include the name of the environment that you created to run this project 
  - Save your changes
#### Running `run_ensemble.py`, you need to run `run_ensemble.sh`
- Open `run_ensemble.sh`:
  - Change the path of the output
  - Change the path of the error
  - Change the mail user
  - Change the json path 
  - Change the name of the environment in `conda activate sbi_new` , include the name of the environment that you created to run this project 
  - Save your changes
  - Open your verdee terminal
  - Write on you terminal the following command: `sbatch run_ensemble.sh`


### ` evaluate_ensemble.py`
This is the fourth step that needs to be done after running `run_ensemble.py`. This script is responsible for executing the posterior distribution and generating the density of the 10,000 samples.
**Remark:** Try to add `print` statment when you want to debug
#### Running `run_ensemble.py`, you need to run `run_ensemble.sh`
- Open `run_ensemble.sh`:
  - Change the path of the output
  - Change the path of the error
  - Change the mail user
  - Change the json path 
  - Change the name of the environment in `conda activate sbi_new` , include the name of the environment that you created to run this project 
  - Save your changes
  - Open your verdee terminal
  - Write on you terminal the following command: `sbatch evaluate_ensemble.sh`


### `pf_ens_functions.py`
This script has nothing to do with running the other scripts. However,  it is used to provide functions that will be used in the other scripts. 


### `analyze_ensemble.ipynb`
This script has nothing to do with running the other scripts. However, it is used to analyze the posterior after running SBI, which is done in the `evaluate_ensemble.py`script. By using this script, you can evaluate the posterior and determine whether you need to repeat steps 2, 3, and 4 to generate a new posterior and move closer to the true values, which, in our case, are derived from the baseline run of PARFLOW. It is the key point that tells you whether you still need to iterate or not.























