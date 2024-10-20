#!/usr/bin/env bash

# Determine the OS
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    # Linux or macOS
    ENV_DIR="./env/bin"
else
    # Windows
    ENV_DIR="./env/Scripts"
fi

# Install virtualenv
python -m pip install --user virtualenv

# Create virtual environment
python -m virtualenv env

# Activate virtual environment
source $ENV_DIR/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Deactivate virtual environment
deactivate