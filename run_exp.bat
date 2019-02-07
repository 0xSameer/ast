#!/bin/bash
export PYTHONIOENCODING=utf-8
echo "Running mt experiment"
source activate $AST_ENV
python train.py -m $1 -e $2
echo "Finished training mt model"

