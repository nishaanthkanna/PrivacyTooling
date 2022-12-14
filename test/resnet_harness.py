## Standard Python
import os
import random

## Python additional
import numpy as np
import tqdm

from mle_logging import MLELogger
from pyhessian import hessian

## PyTorch Imports

# PT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam

# Torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torchvision

def set_seed(seed: int = 42, deterministic = False) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    
    if deterministic:
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
      #torch.use_deterministic_algorithms(True)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class TrainingHarness:
      def __init__(self, train_params):
        self.train_params = train_params

        self.train_dl = train_params['train_dl']
        self.test_dl = train_params['test_dl']
        self.protected_classes = self.train_params['protected_classes']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net = train_params['model']
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.05, momentum=0.99, weight_decay=5e-4)
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, 0.1, epochs=train_params['epochs'], total_steps=train_params['batch_size'] * train_params['epochs'])
        self.criterion = nn.CrossEntropyLoss()
        
        self.net.to(self.device)

      def train_one_epoch(self):
        self.net.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        for data in tqdm.tqdm(self.train_dl):
            ims, labels = data
            ims, labels = ims.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.net(ims)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1) 
            train_correct += pred.eq(labels).sum().item()
            total += labels.size(0)

        train_loss /= len(self.train_dl)
        train_acc = (train_correct / total) * 100
        
        print(self.scheduler._last_lr)
        return train_loss, train_acc

      def test_model(self):
        self.net.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
              predictions = []
              for data in self.test_dl:
                ims, labels = data
                ims, labels = ims.to(self.device), labels.to(self.device)
                outputs = self.net(ims)
                loss = self.criterion(outputs, labels)

                _, pred = torch.max(outputs, 1)
                predictions.extend(list(pred.cpu().numpy()))
                
                test_loss += loss.item()
                test_correct += pred.eq(labels).sum().item()
                test_total += labels.size(0)
        
        test_loss /= len(self.test_dl)
        test_accuracy = (test_correct / test_total) * 100
        return test_loss, test_accuracy, predictions

      def log_norm(self):
        grad_norm_dict = {0 : [], 1 : [], 2 : [], 3 : [], 4 : [], 5 : [], 6 : [], 7 : [], 8 : [], 9 : []}
        for data in self.test_dl:
            ims, labels = data
            ims, labels = ims.to(self.device), labels.to(self.device)
            outputs = self.net(ims)
            for i in self.protected_classes: # protected classes
                if len(labels[labels == i]) > 0:
                    self.net.zero_grad()
                    group_loss = self.criterion(outputs[labels == i], labels[labels == i])
                    group_loss.backward(retain_graph=True)
                    sub_norm = torch.norm(torch.stack([torch.norm(w.grad) for w in self.net.parameters() if w.grad is not None])).item()
                    grad_norm_dict[i].append(sub_norm)

        return grad_norm_dict

      def log_hessian_norm(self):
        hessian_norm_dict = {}
        for data in self.test_dl:
            ims, labels = data
            ims, labels = ims.to(self.device), labels.to(self.device)
            for i in self.protected_classes:
                if len(labels[labels == i]) > 0:
                    sub_hessian_comp = hessian(self.net, self.criterion, data=(ims[labels == i], labels[labels == i]), cuda=True)
                    sub_trace = np.mean(sub_hessian_comp.trace())
                    hessian_norm_dict[f'trace_hessian_{i}'] = sub_trace

        return hessian_norm_dict

def train_experiment(experiment_seed):
    
    set_seed(experiment_seed)

    training_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    
    train_params = {}
    train_params['epochs'] = 30
    train_params['batch_size'] = 256
    #train_params['bn'] = False
    train_params['is_deterministic'] = True
    train_params['protected_classes'] = [2, 3]
    train_dl = DataLoader(training_dataset, batch_size = train_params['batch_size'], shuffle=True, num_workers=2)
    test_dl  = DataLoader(test_dataset, batch_size = train_params['batch_size'], shuffle=False, num_workers=2)
    
    train_params['train_dl'] = train_dl
    train_params['test_dl'] = test_dl
    
    log = MLELogger(time_to_track=['num_updates', 'num_epochs'],
                  what_to_track=['train_loss', 'train_accuracy', 'test_accuracy', 'test_loss', 'lr'],
                  experiment_dir=f"test_all_stuff_{experiment_seed}/",
                  use_tboard=False,
                  model_type='torch',
                  print_every_k_updates=1,
                  ckpt_time_to_track='num_epochs',
                  save_every_k_ckpt=1,
                  verbose=True,
                  seed_id=experiment_seed)
    
    model = torchvision.models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    train_params['model'] = model

    training_run = TrainingHarness(train_params) 

    for epoch in range(train_params['epochs']):
        train_loss, train_acc = training_run.train_one_epoch()
        test_loss, test_acc, predictions = training_run.test_model()
        #grad_norms, hessian_norms = training_run.log_norm(), training_run.log_hessian_norm()
        
        #log.save_extra(predictions, f'predictions_epoch_{epoch}.pkl')
        #log.save_extra(model_params, f'model_params_epoch_{epoch}.pkl') 
        #log.save_extra(grad_norms, f'grad_norms_epoch_{epoch}.pkl')
        #log.save_extra(hessian_norms, f'hessian_norms_epoch_{epoch}.pkl')
        log.update({'num_updates' : epoch+1, 'num_epochs' : epoch+1},
                   {'train_loss' : train_loss, 'train_accuracy' : train_acc,
                    'test_loss' : test_loss,  'test_accuracy' : test_acc,
                   'lr' : training_run.scheduler.get_last_lr()[0]}, training_run.net, save=True)


for seed_option in [1234, 9899, 3483, 3407, 4538]:
    train_experiment(seed_option)