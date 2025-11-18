#! /bin/bash
deepspeed --master_port=49662 --include="localhost:0,1,2,3" main.py --config-path configs/rm-train.yaml > ../logs/train.log 2>&1
sleep 30s
deepspeed --master_port=49662 --include="localhost:0,1,2,3" main.py --config-path configs/rm-precise.yaml > ../logs/train1.log 2>&1
sleep 30s
deepspeed --master_port=49662 --include="localhost:0,1,2,3" main.py --config-path configs/rm-compare.yaml > ../logs/train2.log 2>&1