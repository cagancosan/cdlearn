# Bootstrapped installation of conda and python 3.x that is ready to use.
FROM continuumio/miniconda3:23.10.0-1

# Set working directory.
WORKDIR /cdlearn_app

# User specification.
ARG USER_NAME=cdlearn-user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

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

# Create a system user using user variables.
RUN groupadd --gid $USER_GID $USER_NAME && \
    useradd --uid $USER_UID --gid $USER_GID --create-home $USER_NAME  

# Add the non-root user to the sudo group and grant them sudo privileges.
# No password for sudo commands.
RUN adduser $USER_NAME sudo && \
    echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers 

# Use apt in interactive mode when we are actually using docker container.
ENV DEBIAN_FRONTEND=dialog

# Default shell inside container.
ENV SHELL=/bin/bash

# Set the newly created system user as default, instead of root.
USER $USER_NAME     