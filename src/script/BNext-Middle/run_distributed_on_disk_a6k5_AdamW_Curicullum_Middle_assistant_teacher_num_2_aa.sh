#!/bin/bash
clear
mkdir log 
python3 train_assistant_group_amp.py --model bnext_middle --distillation False --teacher_num 3 --assistant_teacher_num 2 --weak_teacher EfficientNet_B0 --mixup 0.0 --cutmix 0.0 --aug-repeats 1  --dali_cpu  --multiprocessing-distributed --dist-url 'tcp://127.0.0.1:33489' --dist-backend 'nccl' --world-size 1 --rank 0 --data=/data/imagenet/  --batch_size 512 --learning_rate=1e-3  --epochs=512 --weight_decay=0 | tee -a log/training.txt


