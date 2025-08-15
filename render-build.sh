#!/bin/bash
set -e  # Exit on error

# Fix dependency conflicts
pip install --upgrade pip setuptools wheel

# Install critical packages first
pip install pillow==9.5.0 --no-cache-dir
pip install faiss-cpu==1.7.4 --no-cache-dir

# Install remaining requirements
pip install -r requirements.txt --no-cache-dir