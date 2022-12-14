
import shlex, subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=3)
parser.add_argument('--gpu', type=str, default="V100")

args = parser.parse_args()


# Currently fixing seed size and enabling deterministic torch

BATCH_SIZE = 256
EPOCHS = 80
PROTECTED_CLASS = "2 3"
LR = 0.05 
GPU = args.gpu

for i in range(args.runs):
    command = f"python -m resnet18.py --ckpt_folder ./logs_resnet18_cifar10_default_{GPU}_{i}/ --lr {LR} --protected_class {PROTECTED_CLASS} --batch_size {BATCH_SIZE} --epochs {EPOCHS} --deterministic"
    print(f"Running {command} \n Run: {i} ---------")
    train_args = shlex.split(command)
    p = subprocess.Popen(train_args)
    p.wait()
