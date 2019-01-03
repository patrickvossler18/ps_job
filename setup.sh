#!/usr/bin/env bash

cd Deepknockoffs

python3 setup.py install --user

cd ..
pwd
python3 student_t_network.py