#!/bin/bash
# Export conda environment to a yml file.

# How do I activate my environment here?
eval "$(conda shell.bash hook)"

# Grab and activate environment.
AMB=$1
conda activate ${AMB}

# Grab today in lowercase.
NOW=$(date +%d%b%Y | tr [:upper:] [:lower:])

# Information.
echo ">>> Environment: ${AMB}"
echo ">>> NOW: ${NOW}"

# Export it.
FILENAME=$(echo "conda_environment_${AMB}_${NOW}.yml")
echo ">>> Exporting environment in ${FILENAME} ..."
conda env export > "${FILENAME}"
echo ">>> DONE!"
