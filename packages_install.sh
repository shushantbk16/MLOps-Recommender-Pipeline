#!/bin/bash

echo "Starting custom installation script..."

# This forces pip to skip dependencies and only install the troublesome packages
# This is often needed in minimal cloud environments
pip install --no-deps lightfm
pip install --no-deps faiss-cpu
pip install --no-deps xgboost

echo "Custom package installation finished."