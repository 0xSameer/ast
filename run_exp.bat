#!/bin/bash
export PYTHONIOENCODING=utf-8
echo "Running mt experiment"
source activate ast
python nmt_run.py -m $1 -e $2
echo "Finished training mt model"

