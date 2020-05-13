from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import os
# from learning_curve import myplot
import pickle
from EyeTracking import EyeTrackingDataset

'''
python main.py --batch-size 64 --epochs 1 --log-interval 10 --lr 1
python main.py --evaluate --load-model eye_tracking_model.pt
'''


class Net(nn.Module):
 
    def __init__(self,h_rs,w_rs,args):
        super(Net, self).__init__()        
        self.args = args
        
        out1, out2, out3 = 8, 16, 16 
        self.ker1, self.ker2, self.ker3 = 2, 4, 4
        lin_in_w = self.calculate_size(w_rs)
        lin_in_h = self.calculate_size(h_rs)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out1, kernel_size=self.ker1, stride=1)
        self.conv2 = nn.Conv2d(out1, out2, self.ker2, 1)
        self.conv3 = nn.Conv2d(out2, out3, self.ker3, 1)

        self.fc1 = nn.Linear(out2*lin_in_h*lin_in_w, 64)
        self.fc2 = nn.Linear(64, 3)
        
        self.forward_pass = nn.Sequential(
            self.conv1, nn.ReLU(), nn.Dropout2d(0.5), nn.BatchNorm2d(out1),
            self.conv2, nn.ReLU(), nn.MaxPool2d(2, stride=2), nn.Dropout2d(0.5), nn.BatchNorm2d(out2),
            self.conv3, nn.ReLU(), nn.MaxPool2d(2, stride=2), nn.Dropout2d(0.5), nn.BatchNorm2d(out3),
            nn.Flatten(), self.fc1, nn.ReLU(),
            self.fc2
        )
        
        self.criterion = nn.MSELoss()
        # Try different optimzers here [Adadelta, Adam, SGD, RMSprop]
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        # self.scheduler = StepLR(self.optimizer, step_size=args.step, gamma=args.gamma)
    
    
    def calculate_size(self, size_in):
    
        size_out = size_in - self.ker1 + 1
        size_out = np.floor((size_out - self.ker2 + 1) / 2)
        size_out = np.floor((size_out - self.ker3 + 1) / 2)

        return int(size_out)
    
    
    def forward(self, x, target):
       
        output = self.forward_pass(x)
        loss = self.criterion(output, target)
        
        if self.training:  
            self.optimizer.zero_grad()               # Clear the gradient              
            loss.backward()                     # Gradient computation
            self.optimizer.step() 
            
        return output,loss

        
    def train_iterate(self, device, epoch, train_loader):      
        start_time = time.time()           
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device,dtype=torch.float), target.to(device,dtype=torch.float)           
            output, loss = self.forward(data, target)                # Make predictions

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                            
                print('Examples:')
                for i in range(2):
                    print('True:', [f'{target.tolist()[i][d]:.3f}' for d in range(3)], 
                          'Pred:', [f'{output.tolist()[i][d]:.3f}' for d in range(3)])
                
                total_time = time.time() - start_time
                print(f'Time per {self.args.log_interval} iter: {total_time}s\n')
 
           
    def test_iterate(self, device, test_loader):     
        
        test_loss = 0   
        
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device,dtype=torch.float), target.to(device,dtype=torch.float)           
            output, loss = self.forward(data, target)                # Make predictions
            test_loss += loss  # sum up batch loss        

        test_loss /= len(test_loader.dataset)
    
        print('\nTest set: Average loss: {:.4f} \n'.format(test_loss))
        print('Examples:')
        for i in range(3):
            print('True:', [f'{target.tolist()[i][d]:.3f}' for d in range(3)], 
                  'Pred:', [f'{output.tolist()[i][d]:.3f}' for d in range(3)])
                
        return 
        # return train_loss, test_loss, train_correct / train_num, test_correct / test_num

        
        
        
#%%
def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--data-partition', type=int, default=1, metavar='N',
                        help='Choose subset of  training set (default: 1)')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)    
    
    partition = args.data_partition
    
    h_rs, w_rs = 96,256
    path_pos = '../../data/Fixation Training Pos.bin'
    dir_images = '../../data/Fixation Training Images'
  
    
#%% Load dataset
    
    # version = '_augmented'
    img_transform = transforms.Compose([
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        transforms.Resize((h_rs,w_rs)),
        transforms.ToTensor()])
        # transforms.Normalize((0.1307,),(0.3081,))
        
    full_dataset = EyeTrackingDataset(path_pos = path_pos, dir_images = dir_images, transform=img_transform)
       
    
    train_length= int(0.7 * len(full_dataset))
    val_length = int(0.1 * len(full_dataset))
    test_length = len(full_dataset) - train_length - val_length

    train_dataset, val_dataset, test_dataset=torch.utils.data.random_split(full_dataset,(train_length,val_length,test_length))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    
#%% Evaluate on test set
    if args.evaluate:
        assert os.path.exists(args.load_model)
        
        # Set the test model
        model = Net(h_rs,w_rs,args).to(device)
        model.load_state_dict(torch.load(args.load_model))
        
        model.eval()
        # test_acc = 
        model.test_iterate(device,test_loader)
        # For ploting partition - test:
        # np.save('../results/loss_test' + str(model_sel) + version + '_part' + str(partition) + '.npy', test_acc)

        return    
    
 
#%% Train model

    model = Net(h_rs,w_rs,args).to(device)

    # train_losses, val_losses, train_accs, val_accs = [],[],[],[]

    # Training loop   
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.train_iterate(device,epoch,train_loader)
        
        model.eval()                       
        with torch.no_grad():
            # train_loss, val_loss, train_acc, val_acc = 
            model.test_iterate(device,val_loader)        
        # train_losses.append(train_loss)
        # val_losses.append(val_loss)
        # train_accs.append(train_acc)
        # val_accs.append(val_acc)

    if args.save_model:        
        torch.save(model.state_dict(), 'eye_tracking_model.pt')
        # np.save('../../results/loss_train.npy', [train_losses,val_losses,train_accs, val_accs])
        
        # For ploting partition - train:
        # torch.save(model.state_dict(), 'mnist_model' + str(model_sel) + version  + '_part.pt')
        # learning_curve_filename = '../results/loss_train' + str(model_sel) + version + '_part' + str(partition) + '.npy'
        # np.save(learning_curve_filename, [train_losses,val_losses,train_accs, val_accs])
        # myplot(learning_curve_filename)

if __name__ == '__main__':
    main()



#%% 2. Kernels visualization
# rfd = 'D:/git/results/'    
# import matplotlib.pyplot as plt


# model = Net()
# model.load_state_dict(torch.load('./mnist_model2.pt'))
# c1w = model.state_dict()['conv1.weight'].numpy()

# fig,axs = plt.subplots(3,3,figsize=(15,15))
# axs = axs.ravel()
# for i in range(8):  
#     axs[i].imshow(np.squeeze(c1w[i,0]),cmap = 'gray')
#     # axs[i].set_title(f'True: {trues[i]}, Pred: {preds[i][0]}')
    
# plt.savefig(rfd + 'kernels.png')









