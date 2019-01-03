#!/usr/bin/env bash

cd DeepKnockoffs

sudo python3 setup.py install --user

cd ..
pwd
python3 student_t_network.py