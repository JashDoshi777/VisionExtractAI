#!/bin/bash
set -e  # Exit on error

# 1. System dependencies for Pillow
sudo apt-get update && sudo apt-get install -y \
    libjpeg-dev \
    zlib1g-dev

# 2. Install Python packages
pip install --upgrade pip setuptools wheel
pip install pillow==10.3.0 --no-cache-dir  # Install first!
pip install -r requirements.txt --no-cache-dir
