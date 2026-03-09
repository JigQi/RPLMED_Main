# Installation

This codebase is tested on Ubuntu 20.04.2 LTS with python 3.10. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -n rplmed python=3.10 -y

# Activate the environment
conda activate rplmed

# Install torch  and torchvision
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

* Clone RPLMed code repository and install requirements
```bash
# Clone RPLMed code base
git clone https://github.com/JigQi/RPLMED_Main

cd RPLMED_Main/
# Install requirements 

pip install -r requirements.txt

```

* Install dassl library
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation
cd RPLMED_Main/Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library
python setup.py develop
cd ..
```
