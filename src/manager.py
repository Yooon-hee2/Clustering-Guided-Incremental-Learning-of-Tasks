import torch
import torch.nn as nn

import copy
import pruner
import discriminator
import cutout

import collections
import glob
import os

import numpy as np
from PIL import Image
import pandas as pd

import torch.optim as optim
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable

import random

random.seed(335)
np.random.seed(335)
torch.manual_seed(335)

device = 'cuda:6' if torch.cuda.is_available() else 'cpu'

class Manager(object):
    """Handles training and pruning."""

    def __init__(self, model, pruning_rate, previous_masks, current_dataset_idx, previous_samples=None, unrelated_tasks=None):

        self.model = model
        self.pruning_rate = pruning_rate
        self.pruner = pruner.Pruner(self.model, self.pruning_rate, previous_masks, current_dataset_idx)
        
        self.unrelated_tasks = unrelated_tasks
        self.discriminator = discriminator.Discriminator(previous_samples, unrelated_tasks)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
        self.decay = [10]
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.decay, gamma=0.2)

    def eval_task_specific(self, dataset_idx, test_loader):
        """Performs evaluation."""

        self.model = self.model.to(device)
        self.pruner.apply_hard_mask(dataset_idx, self.discriminator.unrelated_tasks[dataset_idx])

        self.model.eval()

        test_loss = 0
        total = 0
        correct = 0

        with torch.no_grad():
            for idx, (image, label) in enumerate(test_loader):

                image = image.to(device)
                label = label.to(device)
                image, label = Variable(image), Variable(label)
                
                pred = self.model(image) 
                loss = self.criterion(pred, label)

                test_loss += loss.item()

                _, predicted = torch.max(pred.data, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0) 

        acc = 100 * correct / total
        print('[test] loss: {:.3f} | acc: {:.3f}'.format(test_loss/(idx+1), acc))

        self.model.train_nobn()
        return acc

    def train(self, dataset_idx, epochs, train_loader, sample_loader, isRetrain=False):
        """Performs training."""

        self.model = self.model.to(device)

        #store original values of parameters
        original_model = copy.deepcopy(self.model)

        #find unrelated tasks
        if (dataset_idx > 1) and (isRetrain is False):
            self.discriminator.reduce_dimension(sample_loader) 
            excepted_task = self.discriminator.cluster()

            self.pruner.apply_hard_mask(dataset_idx, excepted_task)

        if isRetrain is True:
            self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
            self.pruner.apply_hard_mask(dataset_idx, self.discriminator.unrelated_tasks[dataset_idx])

        for epoch in range(epochs):
            epoch_idx = epoch + 1
            running_loss = 0
            acc = 0
            total = 0
            correct = 0
            best_acc = -1

            optimizer = self.optimizer
            
            #no batchnorm
            self.model.train_nobn()

            for idx, (image, label) in enumerate(train_loader):
                image = image.to(device)
                label = label.to(device)
                image, label = Variable(image), Variable(label)

                self.model.zero_grad()

                pred = self.model(image)
            
                loss = self.criterion(pred, label)
                loss.backward()
                    
                self.pruner.make_grads_zero() #set fixed param grads to 0.
                optimizer.step() #update parameters
                self.pruner.make_pruned_zero() #set pruned weights to zero

                running_loss += loss.item()

                _, predicted = torch.max(pred.data, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0)

            self.scheduler.step()

            acc = 100 * (correct / total)
            print('[train] epoch : {} | loss: {:.3f} | acc: {:.3f}'.format(epoch_idx, running_loss/(idx+1), acc))

        self.pruner.concat_original_model(dataset_idx, original_model)

    def save_model(self, dataset_idx):
        """Saves model to file."""
        model = self.model
        ckpt = {
            'previous_masks': self.pruner.current_masks,
            'previous_samples': self.discriminator.previous_samples,
            'unrelated_tasks' : self.discriminator.unrelated_tasks,
            'model': model,
        }

        torch.save(ckpt, self.model.datasets[dataset_idx - 1] + '.pt') 
        
    def prune(self, dataset_idx, train_loader):
        """Perform pruning."""
        retraining_epochs = 15 #number of retraining epochs after pruning

        self.pruner.prune()
        
        # retraining after pruning
        print('retraining after pruning...')
        self.train(dataset_idx, retraining_epochs, train_loader, None, True)
        
        self.save_model(dataset_idx) #save final version of model