#!/bin/bash

# This script is used for sharing a conda environment: 
# $ bash update.sh conda_env_name # -v for verbose!

# Error message. Display the usage and exit.
usage() {
    echo "Usage: ${0} [-v] [-e ENV]" >&2
    echo "Share conda environment named ENV." >&2
    echo " -e ENV Export this environment as .yml files named the same as the environment." >&2
    echo " -v     Verbose mode. Display complete information about conda environment." >&2
    exit 1
}

# Make sure the script is not being executed with superuser privileges.
if [[ "${UID}" -eq 0 ]]
then
    echo "Do not execute this script as root!" >&2
    usage
fi

# Parse options.
while getopts e:v OPTION
do
    case ${OPTION} in
        e) ENV="${OPTARG}" ;;
        v) VERBOSE="true" ;;
        ?) usage ;;
    esac
done

# Remove the options while leaving the remaining arguments.
shift "$(( OPTIND - 1 ))"

# If the user doesn't supply environment, give them help.
if [[ ( -z "${ENV}" ) || ( "${#}" -ne 0 ) ]]
then
    echo "Provide just one valid environment!" >&2
    usage
fi

# List all valid environments.
VALID_ENVS=$(conda env list | grep -v "^#" | awk '{print $1}')
mapfile -t < <(echo "${VALID_ENVS}")

# Display information about current conda install.
if [[ "${VERBOSE}" = "true" ]]
then
    echo ">>> Conda info:"
    conda info
    echo ">>> Valid conda envs: ${MAPFILE[@]}"
fi

# Check if the given environment is present in this conda installation.
for VALID_ENV in "${MAPFILE[@]}"
do
    if [ "${VALID_ENV}" = "${ENV}" ]
    then
        
        eval "$(conda shell.bash hook)"
        conda activate "${ENV}"

        # Export your active environment to a new file.
        conda env export > "${ENV}.yml"

	# Exporting an environment with packages I have asked for.
        conda env export --from-history > "${ENV}_basic.yml"
	
        # Exporting an environment explicitly.
        conda list --explicit > "${ENV}_explicit.txt"

	# Turn off.
        conda deactivate

        # Simple log file.
        LAST_RUN=$(date)  # When this script was run.
        OUTPUT="${ENV}_log.txt" # Log file.
        echo "Context" > "${OUTPUT}" 
        echo "==========" >> "${OUTPUT}"
        echo "USER: ${USER}" >> "${OUTPUT}"
        echo "HOST NAME: ${HOSTNAME}" >> "${OUTPUT}"
        echo "LAST RUN: ${LAST_RUN}" >> "${OUTPUT}"
        echo "CONDA ENV: ${ENV}" >> "${OUTPUT}"

        # Everything is OK!
        exit 0  
    fi    
done

# This environment does not exist!
echo ">>> ${ENV} is not present in this conda installation." >&2
echo ">>> The valid environments are the following: ${MAPFILE[@]}" >&2
usage
