from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
# from learning_curve import myplot
import pickle
from datetime import datetime
from EyeTrackingV2 import EyeTrackingDataset

'''
python main.py --batch-size 128 --epochs 1 --log-interval 20 --lr 1
python main.py --evaluate --load-model eye_tracking_model.pt
'''


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--step', type=int, default=1, metavar='N',
                    help='number of epochs between learning rate reductions (default: 1)')
parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--evaluate', action='store_true', default=False,
                    help='evaluate your model on the official test set')
parser.add_argument('--load-model', type=str,
                    help='model file path')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--data-partition', type=int, default=1, metavar='N',
                    help='Choose subset of  training set (default: 1)')
parser.add_argument('--reg-lambda', type=float, default=0.0008, metavar='L',
                    help='Regularization lambda (default:0.0008)')    

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
torch.manual_seed(args.seed)    
partition = args.data_partition

# Data settings
h_rs, w_rs = 60,160
path_pos = '../../data/pos0.bin'
dir_images = '../../data/images0'

rfd = '../../results_project/'
    
    
    

class Net(nn.Module):
 
    def __init__(self,h_rs,w_rs,args):
        super(Net, self).__init__()        
        self.args = args
        
        chns = [16, 32, 64, 64]
        self.kers = [3, 3, 3, 3]
        self.strides = [1, 1, 1, 1]
        self.pad = 1

        lin_in_w = self.calculate_size(w_rs)
        lin_in_h = self.calculate_size(h_rs)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=chns[0], kernel_size=self.kers[0], stride = self.strides[0], padding=self.pad, padding_mode='replicate')
        self.conv2 = nn.Conv2d(chns[0], chns[1], self.kers[1], self.strides[1], padding=self.pad, padding_mode='replicate')
        self.conv3 = nn.Conv2d(chns[1], chns[2], self.kers[2], self.strides[2], padding=self.pad, padding_mode='replicate')
        self.conv4 = nn.Conv2d(chns[2], chns[3], self.kers[3], self.strides[3], padding=self.pad, padding_mode='replicate')

        self.fc1 = nn.Linear(chns[-1]*lin_in_h*lin_in_w, 3)
        # self.fc2 = nn.Linear(16, 16)       
        # self.fc3 = nn.Linear(16, 3)
        
        self.activate = nn.ELU()
        kdrop = 0
        
        self.forward_pass = nn.Sequential(
            self.conv1, nn.BatchNorm2d(chns[0]),  nn.MaxPool2d(2), nn.Dropout2d(kdrop), 
            self.conv2, nn.BatchNorm2d(chns[1]),  nn.MaxPool2d(2), nn.Dropout2d(kdrop), 
            self.conv3, nn.BatchNorm2d(chns[2]),  nn.MaxPool2d(2), nn.Dropout2d(kdrop), 
            self.conv4, nn.BatchNorm2d(chns[3]),  nn.MaxPool2d(2), nn.Dropout2d(kdrop), 
            nn.Flatten(), self.fc1, # nn.Dropout2d(kdrop),
            # self.fc2, 
            # self.fc3
        )
        
        self.criterion_train = nn.MSELoss()
        self.criterion_test = nn.MSELoss(reduction='sum')
        # Try different optimzers here [Adadelta, Adam, SGD, RMSprop]
        self.optimizer = optim.Adadelta(self.parameters(), lr=args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        # self.scheduler = StepLR(self.optimizer, step_size=args.step, gamma=args.gamma)
    
    
    def calculate_size(self, size_in):
        # conv_out = (conv_in + 2×padding - kernel_size) / stride +1

        size_out = np.floor((size_in + 2*self.pad - self.kers[0] + 1) / 2)
        size_out = np.floor((size_out + 2*self.pad - self.kers[1] + 1) / 2)
        size_out = np.floor((size_out + 2*self.pad - self.kers[2] + 1) / 2)
        size_out = np.floor(np.floor((size_out + 2*self.pad - self.kers[3]) / self.strides[3] +1) / 2)

        return int(size_out)
    
    
    def forward(self, x, target):
       
        output = self.forward_pass(x)
        # for layer in self.forward_pass:
        #     x = layer(x)
        #     print(x.size())
        # output = x
        
            
        if self.training:  
            loss = self.criterion_train(output, target)     

            for param in self.parameters():    # Compute regularization
                loss += self.args.reg_lambda*torch.mean(torch.abs(param))
                
            self.optimizer.zero_grad()               # Clear the gradient              
            loss.backward()                     # Gradient computation
            self.optimizer.step() 
        else:
             loss = self.criterion_test(output, target)     

        return output,loss

        
    def train_iterate(self, device, epoch, train_loader):      
        start_time = time.time() 
        batch_losses = []
          
        for batch_idx, (data, target, target_fove) in enumerate(train_loader):
            data, target = data.to(device,dtype=torch.float), target[:,:3].to(device,dtype=torch.float)           
            output, batch_loss = self.forward(data, target)                # Make predictions
            
            batch_losses.append(batch_loss)
            

            if batch_idx % self.args.log_interval == 0:
                print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), batch_loss.item()))
                                
                print('Examples:')
                for i in range(2):
                    print('True:', [f'{target.tolist()[i][d]:.3f}' for d in range(3)], 
                          'Pred:', [f'{output.tolist()[i][d]:.3f}' for d in range(3)])  
                    # print('Train Max:', f'{data[i][0].max()}')

                total_time = time.time() - start_time
                print(f'Time per {self.args.log_interval} iters: {total_time:.2f}s')
                       
        #  Record loss for every batch for the 1st epoch:
        if epoch == 1:
             batch_loss = batch_losses
                                           
        return batch_loss
           
    def test_iterate(self, device, test_loader):     
        
        test_loss = 0   
        preds = []
        trues = [] 
        
        for batch_idx, (data, target, target_fove) in enumerate(test_loader):
            data, target = data.to(device,dtype=torch.float), target[:,:3].to(device,dtype=torch.float)           
            output, loss = self.forward(data, target)                # Make predictions
            test_loss += loss  # sum up batch loss                        
        
            if self.args.evaluate:
              print(f'Test batch {batch_idx}')              
              trues.append(target.detach().numpy())
              preds.append(output.detach().numpy())
            # if batch_idx == 0:
            #       break
                
        test_loss /= len(test_loader.dataset)
    
        print('\nTest set: Average loss: {:.4f}'.format(test_loss))
        print('Examples:')
        for i in range(3):
            print('True:', [f'{target.tolist()[i][d]:.3f}' for d in range(3)], 
                  'Pred:', [f'{output.tolist()[i][d]:.3f}' for d in range(3)])
                
        return test_loss, trues, preds
        
        
        
#%%
def main():

    # Load dataset    
    img_transform = transforms.Compose([
        transforms.Resize((h_rs,w_rs)),
        transforms.Grayscale(),
        # transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor()])
        
    full_dataset = EyeTrackingDataset(path_pos = path_pos, dir_images = dir_images, transform=img_transform, polar = False)
       
    
    train_length= int(0.7 * len(full_dataset))
    val_length = int(0.1 * len(full_dataset))
    test_length = len(full_dataset) - train_length - val_length

    np.random.seed(1)
    train_dataset, val_dataset, test_dataset=torch.utils.data.random_split(full_dataset,(train_length,val_length,test_length))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # Take a look at the data    
    img,target,target_fove = train_dataset[0]
    plt.imshow(img.numpy()[0].squeeze(),cmap = 'gray')
    plt.title(f'{target}')
    plt.show()

    
#%% Evaluate on test set
    if args.evaluate:
        assert os.path.exists(args.load_model)
        
        # Set the test model
        model = Net(h_rs,w_rs,args).to(device)
        model.load_state_dict(torch.load(args.load_model))
        
        model.eval()
        with torch.no_grad():
            test_loss, trues, preds = model.test_iterate(device,test_loader)
            np.save(rfd + 'loss_test.npy', test_loss.detach().numpy())
            np.save(rfd + 'model_prediction.npy',[trues.detach().numpy(),preds.detach().numpy()])

        return    
    
 
#%% Train model
    if os.path.exists(rfd + 'best_val_loss.npy'):
        best_loss = np.load(rfd + 'best_val_loss.npy')
    else:   
        best_loss = 10
    
    model = Net(h_rs,w_rs,args).to(device)
    
    train_batch_losses, val_losses = [],[]

    # Training loop   
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_batch_loss = model.train_iterate(device,epoch,train_loader)
        
        model.eval()                       
        with torch.no_grad():
            val_loss, _ , _ = model.test_iterate(device,val_loader)
                    
            
        # Plot training history for the first epoch
        if epoch == 1: 
            
            fig, ax = plt.subplots()  
            ax.plot(train_batch_loss,marker=".")
            ax.plot(len(train_batch_loss),val_loss,marker="+")
            ax.grid()
            ax.set_xlabel('Batch')
            ax.set_ylabel('Training loss')
            ax.text(0.75,1,model.optimizer, horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            now = datetime.now()
            dt_string = now.strftime("%m%d_%H%M")

            plt.savefig(rfd + dt_string + '_training_history' +  '.png')
            print(model, file=open(rfd + dt_string + "_model.txt", "w"))
            
            train_batch_loss = train_batch_loss[-1]
            
            
        # remember best loss and save   
        val_loss = val_loss.detach().numpy()             
        if args.save_model and val_loss < best_loss:            
            best_loss = val_loss           
            torch.save(model.state_dict(), 'eye_tracking_model.pt')
            np.save(rfd + 'best_val_loss.npy', best_loss)
            best_loss = val_loss
        
        # record train & val loss for every epoch 
        train_batch_losses.append(train_batch_loss.detach().numpy())
        val_losses.append(val_loss)
        
            
    np.save(rfd + 'loss_train.npy', [train_batch_losses, val_losses])
            


if __name__ == '__main__':
    main()

