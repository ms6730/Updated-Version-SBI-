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
