#!/bin/bash
set -ex

# Create conda virtual environment.
CONDA_VENV_NAME=lightweather

# Adds the channel conda-forge to the top of the channel list, making it the highest priority.
conda config --add channels conda-forge

# Install all cdlearn dependencies.
conda create --yes --name ${CONDA_VENV_NAME} --file requirements.txt

# Initialize conda for shell interaction.
conda init bash