#!/bin/sh

python make_data.py

python train.py --seed 71 > seed71.log &
python train.py --seed 72 > seed72.log &
python train.py --seed 73 > seed73.log &
exec python train.py --seed 74 | tee seed74.log # show log
wait

python predict.py

echo "*** done"
