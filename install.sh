#!/bin/bash

# Check Conda
if [[ `command -v conda` == *"conda"* ]]
then
CONDA_V=$( (conda -V) 2>&1)
echo "Found $CONDA_V" 
else
echo "No conda was found. Please install Anaconda/MiniConda."
exit 1
fi

# Check Julia
if [[ `command -v julia` == *"julia"* ]]
then
echo "Found" $(julia -v)
else
echo "No Julia was found. Please install Julia and add Julia into system variable PATH."
exit 1
fi

function check_download_file()
{
    FILENAME=$1
    MD5=$2
    CHECK_MD5SUM=$3
    if [ -f $FILENAME ]
    then
        if [[ `$CHECK_MD5SUM $FILENAME` = *"$MD5"* ]]
        then
            echo $(date) "$FILENAME was downloaded."
        else
            echo $(date) "Downloading $FILENAME"
            curl -O http://static.lichtargelab.org/Diffusion2018/$FILENAME
        fi
    else
        echo $(date) "Downloading $FILENAME"
        curl -O http://static.lichtargelab.org/Diffusion2018/$FILENAME
    fi

    if [[ `$CHECK_MD5SUM $FILENAME` = *"$MD5"* ]]
    then
        echo $(date) "Decompressing $FILENAME"
        tar zxvf $FILENAME -C ./
    else
        return
    fi
}

CHECK_MD5SUM=`command -v md5sum`
if [[ $CHECK_MD5SUM != *"md5sum"* ]]
then
    CHECK_MD5SUM=`command -v md5`
    if [[ $CHECK_MD5SUM != *"md5"* ]]
    then
        echo $(date) "md5 or md5sum is required to check file."
        exit 1
    fi
fi

check_download_file data.tar.gz 827e732c6480b2ffcc19ee16721869d9 $CHECK_MD5SUM

echo $(date) "Installing conda environment"
conda env create -f Diffusion2018.yml

echo $(date) "Activating conda environment"
source activate Diffusion2018

# Save path of the Python in environment
ENV_PYTHON_PATH=$(which python)

echo $(date) "Setting up .env"
PROJECT_DIR=$(pwd)
echo PROJECT_DIR=$PROJECT_DIR > .env

echo $(date) "Installing Julia packages"
chmod 755 ./src/install/install_julia_pkg.jl
./src/install/install_julia_pkg.jl $ENV_PYTHON_PATH

echo $(date) "Installing PyJulia"
unzip pyjulia-master_20180601.zip
cd pyjulia-master
python setup.py install --user
PYJULIA_DIR=$(pwd)
if [ -f "$HOME/.bash_profile" ]
then
    BASH_FILE="$HOME/.bash_profile"
elif [ -f "$HOME/.bash_rc" ]
then
    BASH_FILE="$HOME/.bash_rc"
else
    echo "Cannot find .bash_profile or .bash_rc in $HOME." 
    exit 1
fi
NEWLINE=$'\n'
echo "#Added by multimodal-network-diffusion/install.sh${NEWLINE}export PYTHONPATH=\"$PYJULIA_DIR:\$PYTHONPATH\"" >> $BASH_FILE 
source $BASH_FILE

echo $(date) "Testing PyJulia"
source activate Diffusion2018
PYJULIA_OUTPUT=`python ../src/install/test_pyjulia.py`
if [[ "$PYJULIA_OUTPUT" == *"succeed"* ]]
then
    echo $(date) "$PYJULIA_OUTPUT"
else
    echo $(date) "PyJulia was not installed correctly."
    exit 1
fi

echo $(date) "Installation was completed.${NEWLINE}Check ./notebooks/ for tutorials."