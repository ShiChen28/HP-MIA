from torchvision import datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import numpy as np
import sys


class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    def __len__(self):
        return len(self.data)
    
    
def Texas100():
    data = np.load('./data/texas100.npz')
    features = data['features']
    labels_one = data['labels']
    shuffle_ix = np.random.permutation(np.arange(len(features)))
    features = features[shuffle_ix]
    labels_one = labels_one[shuffle_ix]
    labels = [np.argmax(one_hot)for one_hot in labels_one]
    train_data = features[0:50000]
    train_label = labels[0:50000]
    test_data = features[50000:60000]
    test_label = labels[50000:60000]#67330
    trainset = GetLoader(train_data, train_label)
    testset = GetLoader(test_data, test_label)
    return trainset,testset

def Purchase100():
    data = np.load('./data/purchase100.npz')
    features = data['features']
    labels_one = data['labels']
    shuffle_ix = np.random.permutation(np.arange(len(features)))
    features = features[shuffle_ix]
    labels_one = labels_one[shuffle_ix]
    labels = [np.argmax(one_hot)for one_hot in labels_one]
    train_data = features[0:100000]
    train_label = labels[0:100000]
    test_data = features[100000:120000]
    test_label = labels[100000:120000]
    trainset = GetLoader(train_data, train_label)
    testset = GetLoader(test_data, test_label)
    return trainset,testset




def subsetloader(ls_indices, start, end, trainset, batch_size):
    """
    Function that takes a list of indices and a certain split with start and end, creates a randomsampler and returns
    a subset dataloader with this sampler
    """
    ids = ls_indices[start:end]
    sampler = SubsetRandomSampler(ids)
    loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    return loader



def dataloader(dataset="cifar", batch_size_train = 128, batch_size_test=1000):
    """
    Dataloader function that returns dataloader of a subset for train and test data of CIFAR10 or MNIST.
    """
    try:
        if dataset == "cifar":
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),  
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
            
            trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=True)

        elif dataset == "cifar100":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),  
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
            trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
            testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
            testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=True)
        elif dataset == "mnist":

            transform = transforms.Compose([transforms.ToTensor()])
            trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            testloader = DataLoader(testset, batch_size=batch_size_test,shuffle=True)
        elif dataset == "fashionmnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                ])
            trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            testset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
            testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=True)
        elif dataset == "emnist":
            transform = transforms.Compose([transforms.ToTensor()])
            trainset = datasets.EMNIST(root='./data', train=True, download=True, transform=transform, split = 'letters' )
            testset = datasets.EMNIST(root="./data", train=False, download=True, transform=transform, split = 'letters' )
            testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=True)
        elif dataset == "texas":
            trainset,testset = Texas100()
            testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=True)
        elif dataset == "purchase":
            trainset,testset = Purchase100()
            testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=True)
        else:
            raise NotAcceptedDataset

    except NotAcceptedDataset:
        print('Dataset Error. Choose "mnist" ,"fashionmnist", "emnist","cifar" or "cifar100".')
        sys.exit()

    total_size = len(trainset)
    split1 = total_size // 5
    split2 = split1 * 2
    split3 = split1 * 3
    split4 = split1 * 4
    
    indices = [*range(total_size)]

    data0 = subsetloader(indices, 0, split1, trainset, batch_size_train)
    data1 = subsetloader(indices, split1, split2, trainset, batch_size_train) 
    data2 = subsetloader(indices, split2, split3, trainset, batch_size_train)
    data3 = subsetloader(indices, split3, split4, trainset, batch_size_train)
    data4 = subsetloader(indices,split4, total_size, trainset, batch_size_train)

    return data0,data1,data2,data3,data4,testloader

