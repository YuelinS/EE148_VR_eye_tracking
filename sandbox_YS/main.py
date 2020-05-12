from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
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
    '''
    Build the best MNIST classifier.
    '''   
    def __init__(self,h_rs,w_rs):
        super(Net, self).__init__()
        
        out1, out2 = 8, 16        
        lin_in_w = self.calculate_size(w_rs)
        lin_in_h = self.calculate_size(h_rs)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out1, kernel_size=(2,2), stride=1)
        self.conv2 = nn.Conv2d(out1, out2, 4, 1)
        # self.conv3 = nn.Conv2d(out2, out3, 4, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        # self.dropout3 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(out2*lin_in_h*lin_in_w, 64)
        self.fc2 = nn.Linear(64, 3)
        self.batchnorm1 = nn.BatchNorm2d(out1)
        self.batchnorm2 = nn.BatchNorm2d(out2)
        # self.batchnorm3 = nn.BatchNorm2d(out3)
        
        self.criterion = nn.MSELoss()
        
    def calculate_size(self, size_in):
    
        size_out = size_in - 2
        size_out = np.floor((size_out - 2) / 2)
        return int(size_out)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout2(x)
        x = self.batchnorm2(x)

        # x = self.conv3(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout3(x)
        # x = self.batchnorm3(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # 4. Visualize feature
        feature = x
        output = self.fc2(x)

        # output = F.log_softmax(x, dim=1)
        return output #, feature

#%%

def train(args, model, device, train_loader, optimizer, epoch):

    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device,dtype=torch.float), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = model.criterion(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, *args):
    model.eval()    # Set the model to inference mode
        
    test_loss = 0
    test_correct = 0
    test_num = 0   
    # n_pick = 10
    with torch.no_grad():   # For the inference step, gradient is not computed
    
        # 3. For confusion matrix
        # preds = []
        # trues = []   
        # features = []
        
        # 4. For 8 neighbors
        # with open("../results/8neighbors_pos.txt", "rb") as fp:   # Unpickling
        #     ims_pos = pickle.load(fp)
        # ims_array = np.zeros((5,9,28,28))
        
        for ibatch, (data, target) in enumerate(test_loader):
            data, target = data.to(device,dtype=torch.float), target.to(device)
            
            # inds = [[n,ims_pos.index(m),m.index(n)] for m in ims_pos for n in m if n[0]==ibatch]
            # for k in inds:
            #     ims_array[k[1],k[2]] = data[k[0][1]]
            
            # np.save('../results/8neighbors_img',ims_array)


            output, feature = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # import pdb; pdb.set_trace()
            
            # 1.For accessing incorrect examples:
            # incrt_indices =  (pred.eq(target.view_as(pred))==0).nonzero()
            # imgs,trues,preds = [],[],[]
            # for eg in range(n_pick):
            #     ind = incrt_indices[eg,0]
            #     imgs.append(data[ind].cpu().numpy())
            #     trues.append(target[ind].cpu().numpy())
            #     preds.append(pred[ind].cpu().numpy())                
            # np.save('../results/incorrect_examples.npy',[imgs,trues,preds])
            # return
        
        #     trues.append(target.cpu().numpy())
        #     preds.append(pred.cpu().numpy())          
        # np.savez('../results/confusion_matrix_data.npz',trues,preds)
        
        #     features.append(feature.cpu().numpy())
        #     trues.append(target.cpu().numpy())    
            
        # np.savez('../results/feature_data',features,trues)
       
        # return
        
        
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, test_correct, test_num,
        100. * test_correct / test_num))
    
    if len(args)>0:       
        train_eval_loader = args[0]
        train_loss = 0
        train_correct = 0
        train_num = 0 
        with torch.no_grad():  
            for data, target in train_eval_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    train_correct += pred.eq(target.view_as(pred)).sum().item()
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
    parser.add_argument('--model-number', type=int, default=2, metavar='N',
                        help='select from fcNet, ConvNet, Net (default: 1)')
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

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    models = [fcNet, ConvNet, Net]
    model_sel = args.model_number
    
    partition = args.data_partition
    
    h_rs, w_rs = 24,64
    
    version = ''
    path_pos = '../../data/Fixation Training Pos.bin'
    dir_images = '../../data/Fixation Training Images'
    
    # Evaluate on the official test set
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

    # test_length=len(ants_dataset)-train_length

    train_dataset,val_dataset, test_dataset=torch.utils.data.random_split(full_dataset,(train_length,val_length,test_length))


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    
    #     np.random.seed(42)
    #     np.random.shuffle(target_indice)
        
    #     target_indice = target_indice[:int(np.floor(len(target_indice)/partition))]
        
    #     split = int(np.floor(val_split * len(target_indice)))
                       
    #     subset_indices_train = np.append(subset_indices_train,target_indice[split:])
    #     subset_indices_valid = np.append(subset_indices_valid,target_indice[:split])

    
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size,
    #     sampler=SubsetRandomSampler(subset_indices_train.astype(int).tolist())
    # )
    # train_eval_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.test_batch_size,
    #     sampler=SubsetRandomSampler(subset_indices_train.astype(int).tolist())
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.test_batch_size,
    #     sampler=SubsetRandomSampler(subset_indices_valid.astype(int).tolist())
    # )

    # Load your model 
    model = models[model_sel](h_rs,w_rs).to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    train_losses, val_losses, train_accs, val_accs = [],[],[],[]

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # train_loss, val_loss, train_acc, val_acc = test(model, device, val_loader, train_eval_loader)
            
        scheduler.step()    # learning rate scheduler

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









