#!/bin/sh

python make_data.py

echo "*** train 1"
python train.py --seed 71 | tee seed71.log
echo "*** train 2"
python train.py --seed 72 | tee seed72.log
echo "*** train 3"
python train.py --seed 73 | tee seed73.log
echo "*** train 4"
python train.py --seed 74 | tee seed74.log

python predict.py

echo "*** done"
