import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

def parse_args(args):
    parser = argparse.ArgumentParser(description='EE148 Project')
    parser.add_argument('--polar', action='store_true',
                        help='use polar coordinates')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--validation-percentage', type=float, default=15., metavar='P',
                       help='percentage of training data used for validation')
    # parser.add_argument('--training-division', type=float, default=1., metavar='D',
    #                    help='divide the remaining training data by this factor')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--reg-lambda', type=float, default=0.001, metavar='L',
                        help='Regularization lambda (default:0.001)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')
    parser.add_argument('--save-model', type=str,
                        help='For Saving the current Model');
    return parser.parse_args(args)

class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.polar = args.polar
        
        self.act = nn.ELU()
        # (1, 60, 160)
        
        self.d1 = 32
        self.conv1 = nn.Conv2d(1, self.d1, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        # (32, 60, 160)
        self.batchNorm1 = nn.BatchNorm2d(self.d1)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout2d1 = nn.Dropout2d(0.9)
        # (32, 30, 80)
        
        self.d2 = 32
        self.conv2 = nn.Conv2d(self.d1, self.d2, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        # (32, 30, 80)
        self.batchNorm2 = nn.BatchNorm2d(self.d2)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2d2 = nn.Dropout2d(0.9)
        # (32, 15, 40)
        
        self.d3 = 64
        self.conv3 = nn.Conv2d(self.d2, self.d3, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        # (64, 15, 40)
        self.batchNorm3 = nn.BatchNorm2d(self.d3)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout2d3 = nn.Dropout2d(0.9)
        # (64, 7, 20)
        
        self.d4 = 64
        self.conv4 = nn.Conv2d(self.d3, self.d4, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        # (64, 7, 20)
        self.batchNorm4 = nn.BatchNorm2d(self.d4)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout2d4 = nn.Dropout2d(0.9)
        # (64, 3, 10)
        
        self.fc1 = nn.Linear(1920, 16)
        self.dropout1 = nn.Dropout(0.8)
        self.fc2 = nn.Linear(16, 16)
        self.dropout2 = nn.Dropout(0.8)
        self.fc3 = nn.Linear(16, 4)
        
        self.forward_pass = nn.Sequential(
            self.conv1, self.batchNorm1, self.pool1, #self.dropout2d1, 
            self.conv2, self.batchNorm2, self.pool2, #self.dropout2d2, 
            self.conv3, self.batchNorm3, self.pool3, #self.dropout2d3, 
            self.conv4, self.batchNorm4, self.pool4, #self.dropout2d4, 
            nn.Flatten(), 
            self.fc1, 
            self.fc2, 
            self.fc3
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=args.step, gamma=args.gamma)
    
    def criterion_cart(self, output, target, weight=torch.ones(1, 4), reg_lambda=0.):
        loss = torch.mean(weight[0, 0:3]*(output[:, 0:3]-target[:, 0:3])**2)*0.75
        #loss = F.mse_loss(weight[0, 0:3]*output[:, 0:3], weight[0, 0:3]*target[:, 0:3])
        if weight[0, 3]>0:
            loss += weight[0, 3]*F.binary_cross_entropy_with_logits(output[:, 3], target[:, 3], reduction='mean')*0.25
        if reg_lambda>0:
            for param in self.parameters():    # Compute regularization
                loss += reg_lambda*torch.mean(torch.abs(param))
        return loss
    
    def criterion_polar(self, output, target, weight=torch.ones(1, 4), reg_lambda=0.):
        loss = torch.mean(weight[0, 0:2]*(output[:, 0:2]-target[:, 0:2])**2)*0.5
        loss += torch.mean(weight[0, 2]*((output[:, 2]-target[:, 2])/target[:, 2])**2)*0.25
        if weight[0, 3]>0:
            loss += weight[0, 3]*F.binary_cross_entropy_with_logits(output[:, 3], target[:, 3], reduction='mean')*0.25
        if reg_lambda>0:
            for param in self.parameters():    # Compute regularization
                loss += reg_lambda*torch.mean(torch.abs(param))
        return loss
    
    def forward(self, x, target=None, weight=torch.ones(1, 4)):
        output = self.forward_pass(x)
        if target is None:
            return output
        else:
            if self.polar:
                criterion = self.criterion_polar
            else:
                criterion = self.criterion_cart
            if self.training:  
                loss = criterion(output, target, weight=weight, reg_lambda=self.args.reg_lambda)
                self.optimizer.zero_grad()               # Clear the gradient       
                loss.backward()                     # Gradient computation
                self.optimizer.step() 
            else:
                loss = criterion(output, target, weight=weight, reg_lambda=0)
            return output, loss

    def iterate(self, device, data_loader, epoch, weight, verbose=True):      
        total_loss = 0.
        for batch_idx, (data, target, ref) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)           
            output, loss = self.forward(data, target, weight=weight)                # Make predictions
            total_loss += loss*len(data)
            if batch_idx % self.args.log_interval == 0 and self.training and verbose:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.sampler)} ({100.*batch_idx/len(data_loader):.0f}%)]\tLoss: {loss.item():.4f}')
        total_loss /= len(data_loader.sampler)
        if not self.training:
            print(f'Validation set: Average loss: {total_loss.item():.4f}')
        return total_loss.item()
