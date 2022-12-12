import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from pyhessian import hessian
import numpy as np


class TrainingHarness:
    """ Harness to run one epoch and log norms and hessian trace """

    def __init__(self, train_dl, test_dl, model, optimizer, scheduler, protected_class):

        self.train_dl = train_dl
        self.test_dl = test_dl
        self.protected_classes = protected_class

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.net = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()

        self.net.to(self.device)

    def train_one_epoch(self):
        self.net.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        for data in self.train_dl:
            ims, labels = data
            ims, labels = ims.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.net(ims)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            train_correct += pred.eq(labels).sum().item()
            total += labels.size(0)

        train_loss /= len(self.train_dl)
        train_acc = (train_correct / total) * 100
        self.scheduler.step()
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
        # TODO: make it generic
        grad_norm_dict = {0: [], 1: [], 2: [], 3: [],
                        4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        for data in self.test_dl:
            ims, labels = data
            ims, labels = ims.to(self.device), labels.to(self.device)
            outputs = self.net(ims)
            for i in self.protected_classes:  # protected classes
                if len(labels[labels == i]) > 0:
                    self.net.zero_grad()
                    group_loss = self.criterion(
                        outputs[labels == i], labels[labels == i])
                    group_loss.backward(retain_graph=True)
                    sub_norm = torch.norm(torch.stack(
                        [torch.norm(w.grad) for w in self.net.parameters() if w.grad is not None])).item()
                    grad_norm_dict[i].append(sub_norm)

        return grad_norm_dict

    def log_hessian_norm(self):
        hessian_norm_dict = {}
        for data in self.test_dl:
            ims, labels = data
            ims, labels = ims.to(self.device), labels.to(self.device)
            for i in self.protected_classes:
                if len(labels[labels == i]) > 0:
                    sub_hessian_comp = hessian(self.net, self.criterion, data=(
                        ims[labels == i], labels[labels == i]), cuda=True)
                    sub_trace = np.mean(sub_hessian_comp.trace())
                    hessian_norm_dict[f'trace_hessian_{i}'] = sub_trace

        return hessian_norm_dict