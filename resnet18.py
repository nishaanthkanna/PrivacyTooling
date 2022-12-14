import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import OneCycleLR
import argparse
import numpy as np
from model.resnet import ResNet
from dataloader.data_loader import get_dataloaders
from harness.harness import TrainingHarness
from mle_logging import MLELogger
from utils.utils import set_seed
from torchinfo import summary

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--protected_class', nargs='+', type=int, required=True)
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--save_hessian', action='store_true')
parser.add_argument('--ckpt_folder', type=str, required=True)
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--tpu_zone', type=str, default=None)
parser.add_argument('--tpu_project', type=str, default=None)
parser.add_argument('--tpu_address', type=str, default=None)
parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
parser.add_argument('--deterministic_torch', action='store_true')
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--l2', type=float, default=0.0)
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='sgd')

args = parser.parse_args()

set_seed(0, 
        deterministic_torch=args.deterministic_torch, 
        deterministic=args.deterministic)

num_epochs = args.epochs


if args.dataset == 'cifar10':
    NUM_CLASSES = 10
elif args.dataset == 'cifar100':
    NUM_CLASSES = 100

layers = [2, 2, 2, 2]
model = ResNet(layers, num_classes=NUM_CLASSES)

if args.optimizer == "adam":
    optimizer = Adam(model.parameters(), lr=args.lr)
else:
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)

scheduler = OneCycleLR(optimizer, 0.1, epochs=args.epochs, total_steps=args.batch_size * args.epochs)


train_dl, test_dl = get_dataloaders(args.dataset, args.batch_size)

training_harness = TrainingHarness(train_dl=train_dl,
                                    test_dl=test_dl,
                                    model=model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    protected_class=args.protected_class)



log = MLELogger(time_to_track=['num_updates', 'num_epochs'],
                what_to_track=['train_loss', 'train_accuracy', 'test_accuracy', 'test_loss', 'lr'],
                experiment_dir=args.ckpt_folder,
                use_tboard=False,
                model_type='torch',
                print_every_k_updates=1,
                ckpt_time_to_track='num_epochs',
                save_every_k_ckpt=1,
                verbose=True,
                seed_id=0)


for epoch in range(num_epochs):
    train_loss, train_acc = training_harness.train_epoch()
    test_loss, test_acc, predictions = training_harness.valid_epoch()
    model_params = [p for p in training_harness.net.parameters() if p.grad is not None and p.requires_grad]
    
    if args.save_hessian:
        grad_norms, hessian_norms = training_harness.log_norm(), training_harness.log_hessian_norm()
        log.save_extra(grad_norms, f'grad_norms_epoch_{epoch}.pkl')
        log.save_extra(hessian_norms, f'hessian_norms_epoch_{epoch}.pkl')
    
    log.save_extra(predictions, f'predictions_epoch_{epoch}.pkl')
    log.save_extra(model_params, f'model_params_epoch_{epoch}.pkl') 
    log.update({'num_updates' : epoch+1, 'num_epochs' : epoch+1},
                {'train_loss' : train_loss, 'train_accuracy' : train_acc,
                'test_loss' : test_loss,  'test_accuracy' : test_acc,
                'lr' : training_harness.scheduler.get_last_lr()[0]}, training_harness.net, save=True)



