from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
# from learning_curve import myplot
import pickle
from EyeTracking import EyeTrackingDataset

'''
python main.py --batch-size 128 --epochs 1 --log-interval 200 --model-number 0 --data-partition 4
python main.py --evaluate --load-model mnist_model.pt --model-number 2 --data-partition 1
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)

        # output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
 
    def __init__(self,h_rs,w_rs):
        super(Net, self).__init__()
        
        out1, out2 = 8, 16        
        lin_in_w = self.calculate_size(w_rs)
        lin_in_h = self.calculate_size(h_rs)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out1, kernel_size=(2,2), stride=1)
        self.conv2 = nn.Conv2d(out1, out2, 4, 1)
        # self.conv3 = nn.Conv2d(out2, out3, 4, 1)

        self.fc1 = nn.Linear(out2*lin_in_h*lin_in_w, 64)
        self.fc2 = nn.Linear(64, 3)
        
        self.batchnorm1 = nn.BatchNorm2d(out1)
        self.batchnorm2 = nn.BatchNorm2d(out2)
        # self.batchnorm3 = nn.BatchNorm2d(out3)
        
        self.forward_pass = nn.Sequential(
            self.conv1, nn.ReLU(), nn.MaxPool2d(2, stride=2), nn.Dropout2d(0.5), self.batchnorm1,
            self.conv2, nn.ReLU(),                            nn.Dropout2d(0.5), self.batchnorm1,
            nn.Flatten(),self.fc1, nn.ReLU(),
            self.fc2
        )
        
        self.criterion = nn.MSELoss()
        # Try different optimzers here [Adadelta, Adam, SGD, RMSprop]
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        self.scheduler = StepLR(self.optimizer, step_size=args.step, gamma=args.gamma)
    
    
    def calculate_size(self, size_in):
    
        size_out = size_in - 2
        size_out = np.floor((size_out - 2) / 2)
        return int(size_out)
    
    
    def forward(self, x):
       
        output = self.forward_pass(x)
        
        if self.training:  
            self.optimizer.zero_grad()               # Clear the gradient

            loss = self.criterion(output, target)   # Compute loss
            loss.backward()                     # Gradient computation
            self.optimizer.step() 
            
        return output

        
    def train_iterate(self,train_loader)：:        
                             
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device,dtype=torch.float), target.to(device)           
            output = self.forward(data)                # Make predictions

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), loss.item()))

            
    def test_iterate(self,test_loader)：:        
                            
       for batch_idx, (data, target) in enumerate(test_loader):
           data, target = data.to(device,dtype=torch.float), target.to(device)           
           output = self.forward(data)                # Make predictions
           test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
           pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability     
        
           test_correct += pred.eq(target.view_as(pred)).sum().item()
           test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, test_correct, test_num,
        100. * test_correct / test_num))
                    train_num += len(data)
        train_loss /= train_num
    
        print('Full Training Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            train_loss, train_correct, train_num,
            100. * train_correct / train_num))
        
        return train_loss, test_loss, train_correct / train_num, test_correct / test_num
    else:
        return test_correct / test_num
        
        
        
#%%
def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
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
    
    h_rs, w_rs = 24,64
    
    version = ''
    path_pos = '../../data/Fixation Training Pos.bin'
    dir_images = '../../data/Fixation Training Images'
    
    # Evaluate on test set
    if args.evaluate:
        assert os.path.exists(args.load_model)
        
        # Set the test model
        model = models[model_sel]().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../MNIST', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test_acc = test(model, device, test_loader)
        # For ploting partition - test:
        # np.save('../results/loss_test' + str(model_sel) + version + '_part' + str(partition) + '.npy', test_acc)

        return


    # Data augmentation
    
    # version = '_augmented'
    img_transform = transforms.Compose([
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        transforms.Resize((h_rs,w_rs)),
        transforms.ToTensor()])
        # transforms.Normalize((0.1307,),(0.3081,))
        
    full_dataset = EyeTrackingDataset(path_pos = path_pos, dir_images = dir_images, transform=img_transform)
       
    
    train_length= 1000  #int(0.7* len(ants_dataset))
    val_length = 400
    test_length = len(full_dataset)-train_length-val_length

    train_dataset,val_dataset, test_dataset=torch.utils.data.random_split(full_dataset,(train_length,val_length,test_length))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

 
    # Load your model 
    model = Net(h_rs,w_rs).cuda()


    train_losses, val_losses, train_accs, val_accs = [],[],[],[]

    # Training loop
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.train_iterate(args, model, device, train_loader, optimizer, epoch)
        
        model.eval()
                  
        test_loss = 0
        test_correct = 0
        test_num = 0   

        with torch.no_grad():
            train_loss, val_loss, train_acc, val_acc = model.test_iterate(model, device, val_loader, train_eval_loader)

        

        # train_losses.append(train_loss)
        # val_losses.append(val_loss)
        # train_accs.append(train_acc)
        # val_accs.append(val_acc)

    if args.save_model:        
        torch.save(model.state_dict(), 'mnist_model' + str(model_sel) + version  +'.pt')
        np.save('../../results/loss_train' + str(model_sel) + version + '.npy', [train_losses,val_losses,train_accs, val_accs])
        
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









