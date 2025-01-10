#!/bin/bash

cd DiCoW-v1 || exit

# activate the virtual environment
conda activate dicow

# set the environment variables
export HF_TOKEN=''

# run the server
python app.py
wait