# Installation

This codebase is tested on Ubuntu 20.04.2 LTS with Python 3.10. Follow the steps below to create the environment and install dependencies.

## Step 1: Setup Conda Environment (Recommended)

```bash
# Create a conda environment
conda create -n rplmed python=3.10 -y

# Activate the environment
conda activate rplmed

# Install PyTorch 2.0.1 and torchvision
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

## Step 2: Clone RPL-Med and Install Requirements

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>/

# Install Python dependencies
pip install -r requirements.txt
```

## Step 3: Install Dassl.pytorch

```bash
# Instructions adapted from https://github.com/KaiyangZhou/Dassl.pytorch#installation
cd RPLMed/Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install the library in development mode
python setup.py develop
cd ..
```
