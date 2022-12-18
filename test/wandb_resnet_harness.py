
## Standard Python
import os
import random

## Python additional
import numpy as np
import tqdm

#from mle_logging import MLELogger
import wandb

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

device_name = ''

wandb.init(project='tooling-fairness', name=device_name)

def set_seed(seed: int = 42, deterministic = False) -> None:
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    
    if deterministic:
      print('This ran')
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

      def log_artifact(self, filename, model_path):
        artifact = wandb.Artifact(filename, type='model')
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)

def train_experiment(experiment_seed):
    
    set_seed(experiment_seed, is_deterministic=False)

    training_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    
    train_params = {}
    train_params['epochs'] = 80
    train_params['batch_size'] = 256
    #train_params['bn'] = False
    train_params['is_deterministic'] = False
    train_params['protected_classes'] = [2, 3]
    train_dl = DataLoader(training_dataset, batch_size = train_params['batch_size'], shuffle=True, num_workers=2)
    test_dl  = DataLoader(test_dataset, batch_size = train_params['batch_size'], shuffle=False, num_workers=2)
    
    train_params['train_dl'] = train_dl
    train_params['test_dl'] = test_dl
    
    model = torchvision.models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    train_params['model'] = model

    training_run = TrainingHarness(train_params) 

    for epoch in range(train_params['epochs']):
        train_loss, train_acc = training_run.train_one_epoch()
        test_loss, test_acc, predictions = training_run.test_model()
        wandb.log({"training_loss"     : train_loss,
                   "training_accuracy" : train_acc,
                   "test_loss"         : test_loss,
                   "test_accuracy"     : test_acc,
                   "lr"                : training_run.scheduler.get_last_lr()[0],
                   "epoch"             : epoch+1})
        torch.save(model.state_dict(), f'./model_epoch_{epoch}.pt')
        
        training_run.log_artifact(f'{device_name}_model-ckpt-epoch-{epoch}.pt', f'./model_epoch_{epoch}.pt')

        


for seed_option in [1234, 9899, 3407, 3483, 4538]:
    train_experiment(seed_option)

