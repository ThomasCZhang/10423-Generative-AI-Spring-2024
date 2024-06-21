#!bin/bash

# sudo yum install python3-pip
# sudo yum install python3.10 python3.10-venv
# sudo yum install nvidia-cuda-toolkit

python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt