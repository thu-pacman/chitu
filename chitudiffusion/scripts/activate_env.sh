#!/bin/bash

# Check if the argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 'venv_name'"
    exit 1
fi

# Prepare the environment
source $1/bin/activate
# load spack
source /home/spack/spack/share/spack/setup-env.sh
# spack load cuda@12.1.1
# spack load /gjb # load cudnn@8.9.7.29-12
# load cuda@12.8 and cudnn@9
spack load cuda@12.8
spack load cudnn@9

spack load cmake
