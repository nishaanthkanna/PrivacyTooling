
import shlex, subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=3)

args = parser.parse_args()

# Currently fixing seed size and enabling deterministic torch

BATCH_SIZE = 128
EPOCHS = 200
PROTECTED_CLASS = "2 3"
LR = 4e-4

for i in range(args.runs):
    command = f"python3 -m resnet18.py --ckpt_folder ./logs_resnet18_cifar10_default_{i}/ --lr {LR} --protected_class {PROTECTED_CLASS} --batch_size {BATCH_SIZE} --epochs {EPOCHS}"
    print(f"Running {command} Count: {i}")
    train_args = shlex.split(command)
    p = subprocess.Popen(train_args)
    p.wait()
