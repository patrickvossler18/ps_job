#!/usr/bin/env bash

cd DeepKnockoffs

sudo python setup.py install --user

cd ..
pwd
python3 student_t_network.py