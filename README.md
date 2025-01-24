# SBI-ML-Updated-Version

### Create Virtual Environment 
- Open the terminal on Verdee  
- `conda --version`  to check the version of the conda   
- `conda create -n sbi_env` to create new environment named sbi_env  
-  `conda init bash` to ensure that the conda command is available in your terminal without requiring you to manually source Conda's activation script  
- Close terminal and open new terminal to refresh it  
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

### Modifications that needs to be done before running `setup_experiment.py`
- Open `settings.JSON` file 
- Basic directory in `settings.JSON` needs to be changed 
- Open `setup_experiment.sh`:
  - Change the path of the output
  -  Change the path of the error
  - Change the mail user
  - Change the `JSON PATH` 
  - Change the name of the environment in `conda activate sbi_new` , include the name of the environment that you created to run this project 
- Open `setup_experiment.py`:
  - Change the path in **line 15**


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
- Open `setup_experiment.sh`:
  change the json file to the **`new_press JSON file`** and modify the path
- Open your verdee terminal 
- Locate `setup_experiment.sh`
- Write `sbatch setup_experiment.sh` on your terminal to start by the execution of the script

### `create_ensemble.py`
#### Running `create_ensemble.py`, you need to run `create_ensemble.sh`
- Open `create_ensemble.sh`
- Change the path of the output
- Change the path of the error
- Change the mail user
- Change the json path 
- Change the name of the environment in `conda activate sbi_new` , include the name of the environment that you created to run this project 


