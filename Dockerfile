# Bootstrapped installation of conda and python 3.x that is ready to use.
FROM continuumio/miniconda3:23.10.0-1

# Set working directory.
WORKDIR /cdlearn_app

# Disable apt from prompting.
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and clean installation.
RUN apt-get update --yes --no-install-recommends && \
    apt-get install --yes --no-install-recommends \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove --yes && \
    apt-get clean --yes    

# Default shell inside container.
ENV SHELL=/bin/bash

# Copy all files used in testing.
COPY cdlearn cdlearn
COPY colormaps colormaps
COPY test test 
COPY requirements.txt .

# Install packages using conda environment.
# See this reference: https://medium.com/@chadlagore/conda-environments-with-docker-82cdc9d25754
ARG CONDA_VENV_NAME=lightweather
RUN conda config --add channels conda-forge && \
    conda create --yes --name ${CONDA_VENV_NAME} --file requirements.txt 

# Add conda bin to path.
RUN echo "source activate ${CONDA_VENV_NAME}" > ~/.bashrc
ENV PATH /opt/conda/envs/${CONDA_VENV_NAME}/bin:$PATH
RUN /bin/bash -c "source activate ${CONDA_VENV_NAME}"

# Install packages for testing cdlearn.
RUN pip install pytest pylint