#!/bin/bash
set -e

# Force binary installs
export PIP_ONLY_BINARY=:all:

# Install in safe order
pip install --upgrade pip setuptools wheel
pip install pillow==10.2.0 --no-cache-dir --only-binary=:all:
pip install -r requirements.txt --no-cache-dir
