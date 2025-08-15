#!/bin/bash
set -e  # Exit on error

# Install system dependencies for Pillow
sudo apt-get update
sudo apt-get install -y libjpeg-dev zlib1g-dev

# Install Python packages
pip install --upgrade pip setuptools wheel
pip install pillow==9.5.0 --no-cache-dir  # Install first!
pip install -r requirements.txt --no-cache-dir
