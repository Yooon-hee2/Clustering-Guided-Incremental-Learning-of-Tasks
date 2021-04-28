import warnings
import argparse

import torch
import networks
from manager import *
from dataloader import *
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST

device = 'cuda' if torch.cuda.is_available() else 'cpu'

warnings.filterwarnings("ignore")

FLAGS = argparse.ArgumentParser()

FLAGS.add_argument('--num_outputs', type=int, default=-1,
                   help='Num outputs for dataset')
FLAGS.add_argument('--dataset', type=str, default='',
                   help='Name of dataset')   

FLAGS.add_argument('--train_path', type=str, default='',
                   help='Location of train data')
FLAGS.add_argument('--test_path', type=str, default='',
                   help='Location of test data')
FLAGS.add_argument('--loadname', type=str, default='',
                   help='Location to save model')

FLAGS.add_argument('--mode', choices=['train', 'eval'], help='Run mode')

def main():
    """Do stuff."""
    args = FLAGS.parse_args()

    batch_size = 32
    sample_size = 100
    training_epochs = 20

    if args.mode == 'train':

        ckpt = torch.load(args.loadname, map_location=device)
        model = ckpt['model']
        previous_masks = ckpt['previous_masks']

        model.add_dataset(args.dataset, args.dataset.num_outputs)
        model.set_dataset(args.dataset)

        current_dataset_idx = len(model.datasets)

        unrelated_tasks = None
        previous_samples = None

        if current_dataset_idx > 2:
            unrelated_tasks = ckpt['unrelated_tasks']
            previous_samples = ckpt['previous_samples']

            if current_dataset_idx == 3:
                previous_samples = ckpt['previous_samples'].cpu()

        manager = Manager(model, 0.75, previous_masks, current_dataset_idx, previous_samples, unrelated_tasks)
        
        if current_dataset_idx == 2:
            manager.pruner.current_masks = previous_masks
        else:
            manager.pruner.initialize_new_mask()

        if 'cropped' in args.train_path:
            train_loader = train_loader_cropped(args.train_path, batch_size)
            test_loader = test_loader_cropped(args.test_path, batch_size)
            sample_loader = train_loader_sample(args.train_path, sample_size)

        if args.dataset == 'mnist':

            mnist_trainset = MNIST(root='./data',
                                train=True, 
                                download=True, 
                                transform=transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                                            
                                ]))

            mnist_testset = MNIST(root='./data',
                                train=False, 
                                download=True, 
                                transform=transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
                                        transforms.Normalize((0.1307, ),(0.3081, ))
                                ]))

            mnist_sampleset = MNIST(root='./data',
                                train=True, 
                                download=True, 
                                transform=transform_sample())

            train_loader = data.DataLoader(mnist_trainset, batch_size, shuffle=True)
            test_loader = data.DataLoader(mnist_testset, batch_size)
            sample_loader = data.DataLoader(mnist_sampleset, sample_size, shuffle=True)

        if args.dataset == 'fashion_mnist':

            fmnist_trainset = FashionMNIST(root='./data',
                                train=True, 
                                download=True, 
                                transform=transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(), # image to Tensor
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
                                    transforms.Normalize((0.1307,), (0.3081,)) # image, label
                                                            
                                ]))

            fmnist_testset = FashionMNIST(root='./data',
                                train=False, 
                                download=True, 
                                transform=transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
                                        transforms.Normalize((0.1307, ),(0.3081, ))
                                ]))

            fmnist_sampleset = FashionMNIST(root='./data',
                                train=True, 
                                download=True, 
                                transform=transform_sample())

            train_loader = data.DataLoader(fmnist_trainset, batch_size, shuffle=True)
            test_loader = data.DataLoader(fmnist_testset, batch_size)
            sample_loader = data.DataLoader(fmnist_sampleset, sample_size, shuffle=True)

        else:
            train_loader = train_loader_noncropped(args.train_path, batch_size)
            test_loader = test_loader_noncropped(args.test_path, batch_size)
            sample_loader = train_loader_sample(args.train_path, sample_size)

        manager.train(dataset_idx=current_dataset_idx, epochs=training_epochs, train_loader=train_loader, sample_loader=sample_loader, isRetrain=False)
        manager.prune(current_dataset_idx, train_loader)
        manager.eval_task_specific(current_dataset_idx, test_loader)

    if args.mode == 'eval':

        ckpt = torch.load(args.loadname, map_location=device)
        model = ckpt['model']
        previous_masks = ckpt['previous_masks']
        
        unrelated_tasks = None
        previous_samples = None

        current_dataset_idx = model.datasets.index(args.dataset) + 1

        if len(model.datasets) > 2:
            unrelated_tasks = ckpt['unrelated_tasks']
            previous_samples = ckpt['previous_samples']

        model.set_dataset(args.dataset)

        if 'cropped' in args.test_path:
            test_loader = test_loader_cropped(args.test_path, batch_size)

        if args.dataset == 'mnist':

            mnist_testset = MNIST(root='./data',
                                train=False, 
                                download=True, 
                                transform=transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
                                        transforms.Normalize((0.1307, ),(0.3081, ))
                                ]))

            test_loader = data.DataLoader(mnist_testset, batch_size)

        if args.dataset == 'fashion_mnist':

            fmnist_testset = FashionMNIST(root='./data',
                                train=False, 
                                download=True, 
                                transform=transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
                                        transforms.Normalize((0.1307, ),(0.3081, ))
                                ]))

            test_loader = data.DataLoader(fmnist_testset, batch_size)

        else:
            test_loader = test_loader_noncropped(args.test_path, batch_size)

        manager = Manager(model, 0.75, previous_masks, current_dataset_idx, previous_samples, unrelated_tasks)
        manager.eval_task_specific(current_dataset_idx, test_loader)

if __name__ == '__main__':
    main()

    