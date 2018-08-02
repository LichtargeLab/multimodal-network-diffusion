Multimodal Network Diffusion
==============================

Repository for *Multimodal Network Diffusion Predicts Future Disease-Gene-Chemical Associations*.  
If you have any questions or comments, feel free to contact Chih-Hsu Lin (chihhsul[at]bcm[dot]edu).

--------
## Content
 - [Download code](#download-code)
 - [Installation and download network data](#installation-and-download-network-data)
 - [Run tutorial](#run-tutorial)
 - [Run experiments](#run-experiments)
 - [Project organization](#project-organization)
 - [Links to download data](#links-to-download-data)

--------
## Download code
```bash
git clone https://github.com/ChihHsuLin/multimodal-network-diffusion
```

--------
## Installation and download network data

### Requirements
- [Anaconda](https://www.anaconda.com/) or [MiniConda](https://conda.io/miniconda.html)
- For Mac: [Julia 0.5.2](https://julialang.org/downloads/oldreleases.html)
- For Linux: [Julia 0.6.3](https://julialang.org/downloads/)

- Add Julia path in system variable `PATH` 
Mac commands are below. You might need to change the Julia path if you install in a different location.
```bash
echo "export PATH=\"/Applications/Julia-0.5.app/Contents/Resources/julia/bin:\$PATH\"" >> "$HOME/.bash_profile" 
source ~/.bash_profile
```

### Install and download network data
```bash
cd multimodal-network-diffusion
./install.sh
```
Mapped network data and the mapping file will be automatically downloaded and extracted to `./data/mapped_networks/`.

--------
## Run tutorial


### 1. Open Jupyter notebook in `./notebooks/`
```bash
cd notebooks/
jupyter notebook
```
### 2. Click on `Prospective_example.ipynb`
### 3. Choose the kernel of the `Diffusion2018` environment
#### (1) Choose `Python [Diffusion2018]`  when being asked to choose kernel
#### or (2) (In the menu bar of the newly opened page) Kernel -> Change kernel -> Python [Diffusion2018]
*The kernel name could be `Python [Diffusion2018]` or `Python [conda env:Diffusion2018]` in our tests.*
### 4. Run each cell

### Q: If you don't see `Python [Diffusion2018]` in Jupyter Notebook 
### A: Enable environment for Jupyter Notebook
#### 1. Activate environment (skip it if you already activated)
```bash
source activate Diffusion2018
```
#### 2. Register the environment for Jupyter Notebook
```bash
python -m ipykernel install --user --name Diffusion2018 --display-name "Python [Diffusion2018]"
```
### Q: If you need to run Jupyter Notebook through ssh
### A: Please follow the instructions [here](http://www.blopig.com/blog/2018/03/running-jupyter-notebook-on-a-remote-server-via-ssh/)

--------
## Run experiments
### 0. Recommended resource
#### Memory >=256GB is recommended for running large networks.

### 1. (Optional) Download precomputed results of algorithms
```bash
./download_precomp.sh
```
Enter `y` or `n` for downloading experiments of interests:

    Please choose experiments of interests:
    10-fold cross-validation (275 GB after compression) [y]/n ? 
    Leave-one-mode-out (9.4 GB after compression) [y]/n ? 
    Time-stamped (22 GB after compression) [y]/n ? 
    Prospective (21.6 GB after compression) [y]/n ? 

### 2. Activate environment
```bash
source activate Diffusion2018
```
If it's activated, you will see `(Diffusion2018)`  at the beginning of your command prompt  

### 3. Example of running 10-fold cross-validation of 1-mode networks
Experiment parameter: 1mode:
```bash
cd src
python runKfoldCV.py --e 1mode # Could be one of {'1mode','3mode','6mode'}
```
Experiment parameter options: 
**1mode**: 1-mode networks; **3mode**: 3-mode networks; **6mode**: 6-mode networks  

### 4. Example of running leave-one-mode-out experiment of disease-gene prediction
```bash
cd src
python runLeaveAModeOut.py --e DG # Could be one of {'DG','DC','GC'}
```
Experiment parameter options: 
**DG**: disease-gene; **DC**: disease-chemical; **GC**: gene-chemical  

### 5. Example of running time-stamped experiment of disease-chemical prediction
```bash
cd src
python runTimeStamped.py --e DC # Could be one of {'DG','DC','GC'}
```
Experiment parameter options: 
**DG**: disease-gene; **DC**: disease-chemical; **GC**: gene-chemical  

### 6. Example of running prospective experiment of disease-gene prediction
```bash
cd src
python runProspective.py --e DG # Could be one of {'DG','DC','GC'}
```
Experiment parameter options: 
**DG**: disease-gene; **DC**: disease-chemical; **GC**: gene-chemical  


--------

Project organization
------------
    multimodal-network-diffusion/
    ├── README.md                       <- This document
    ├── install.sh                      <- The script to set up environment and download data.
    ├── download_precomp.sh             <- The script to download precomputed data
    ├── Diffusion2018.yml               <- Conda environment file
    ├── pyjulia-master_20180601.zip     <- PyJulia downloaded on 2018/06/01.
    │
    ├── src/                            <- Source code for use in this project.
    │   ├── runKfoldCV.py               <- The script to run 10-fold cross-validation.
    │   ├── runLeaveAModeOut.py         <- The script to run leave-one-mode-out experiments.
    │   ├── runTimeStamped.py           <- The script to run time-stamped experiments.
    │   ├── runProspective.py           <- The script to run prospective experiments.
    │   ├── paths.py                    <- The script to load environment variables.
    │   │
    │   ├── install/                    <- The scripts to test installation.
    │   │
    │   ├── Networks/                   <- The classes for manipulating graph data.
    │   │
    │   ├── Algorithms/                 <- The classes of different algorithms.
    │   │
    │   └── Validation_Methods/         <- The classes of different validation experiments.
    │
    └── notebooks/                      <- Jupyter notebooks
        ├── Prospective_example.ipynb   <- Example to run a prospective experiment
        └── Prospective_example.html    <- The html of expected results of the example.


--------
## Links to download data
1. Mapped networks, mapping files, saved PubMed searching results, and predictions from prospective experiments (could be downloaded through `install.sh`):  
[data.tar.gz](http://static.lichtargelab.org/Diffusion2018/data.tar.gz) (md5: 827e732c6480b2ffcc19ee16721869d9; 75 MB after compression)
2. Precomputed algorithm results of 10-fold cross-validation (could be downloaded through `download_precomp.sh`):  
[10_fold.tar.gz](http://static.lichtargelab.org/Diffusion2018/10_fold.tar.gz) (md5: a9b538536c0b59f3a3860ac77fffa755; 275 GB after compression)
3. Precomputed algorithm results of leave-one-mode-out experiments (could be downloaded through `download_precomp.sh`):  
[lomo.tar.gz](http://static.lichtargelab.org/Diffusion2018/lomo.tar.gz) (md5: f996d985a4fa9c6baf02ff91f2021cf2; 9.4 GB after compression)
4. Precomputed algorithm results of time_stamped experiments (could be downloaded through `download_precomp.sh`):  
[time_stamped.tar.gz](http://static.lichtargelab.org/Diffusion2018/time_stamped.tar.gz) (md5: 16a18fce049820f5cce95d7f5eba3916; 22 GB after compression)
5. Precomputed algorithm results of prospective experiments (could be downloaded through `download_precomp.sh`):  
[prospective.tar.gz](http://static.lichtargelab.org/Diffusion2018/prospective.tar.gz) (md5: 5cbd2158c9b8895cb4aa836e1800ae48; 21.6 GB after compression)
--------
