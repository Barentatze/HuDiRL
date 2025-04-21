#!/bin/bash

# Download and install Miniconda
MINICONDA_INSTALLER=Miniconda3-latest-Linux-x86_64.sh
wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Get the parent directory of the current working directory
#PARENT_DIR="$(cd .. && pwd)"

# Install Miniconda to $PARENT_DIR/miniconda
#bash ~/miniconda.sh -b -p "$PARENT_DIR/miniconda"

# Add to PATH
#export PATH="$PARENT_DIR/miniconda/bin:$PATH"

# Initialize Conda and create a new environment
conda init
source ~/.bashrc

conda create -n SinDiffusion python=3.11 -y
conda activate SinDiffusion

# Install MPI
sudo apt update
sudo apt install -y libopenmpi-dev openmpi-bin

conda install -y mpi4py

# Install related dependencies
pip install -r requirements.txt

echo "Installation completed!"
