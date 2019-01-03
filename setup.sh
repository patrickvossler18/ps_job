#!/usr/bin/env bash
ls
cd DeepKnockoffs

python3 setup.py install --user

cd ..
pwd
python3 student_t_network.py