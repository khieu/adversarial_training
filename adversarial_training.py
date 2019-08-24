import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
from torchvision import transforms, datasets, models
from random import shuffle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchsummary import  summary
from torch import nn
import pandas as pd
from timeit import default_timer as timer

torch.backends.cudnn.benchmark = False

caltech_dir = '/scratch/hle/101_ObjectCategories'
list_dirs = os.listdir(caltech_dir)
list_dirs.sort()

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

traindir = '/scratch/hle/caltech_101/train'
validdir = '/scratch/hle/caltech_101/valid'
testdir = '/scratch/hle/caltech_101/test'
batch_size = 32
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'val':
    datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
}

# Dataloader iterators
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
}


model = models.vgg16()

#for param in model.parameters():
#    param.requires_grad = False

n_inputs = model.classifier[6].in_features

model.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256,101),
    nn.LogSoftmax(dim=1)
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fgsm(model, X, y, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()


def epoch(loader, model, opt=None):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def epoch_adversarial(loader, model, attack, opt=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, **kwargs)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

#model, optimizer = load_checkpoint(path=checkpoint_path)
model = model.to(device)
model = nn.DataParallel(model)
opt = optim.Adam(model.parameters())
criterion = nn.NLLLoss()
#for t in range(30):
#    start = time.time()
#    train_err, train_loss = epoch_adversarial(dataloaders['train'], model, pgd_linf, opt)
#    train_time = (time.time() - start)/60
#    print("iter: %s train time: %s" %(t,train_time))
#    start = time.time()
#    test_err, test_loss = epoch(dataloaders['test'], model)
#    test_time = (time.time() - start)/60
#    print("iter: %s test time: %s" %(t, test_time))
#    start = time.time()
#    adv_err, adv_loss = epoch_adversarial(dataloaders['test'], model, pgd_linf)
#    adv_test_time = (time.time() - start)/60
#    #if t == 4:
#    #    for param_group in opt.param_groups:
#    #        param_group["lr"] = 1e-2
#    print(*("{:.6f}".format(i) for i in (train_err, test_err, adv_err)), sep="\t")
#    print("train time: %s, test time: %s, adv test time: %s" %(train_time, test_time, adv_test_time))
#    if t % 5 == 0:
#        torch.save(model.state_dict(), "model_cnn_robust_iter%s.pt" % t)

def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=2,
          attack=None):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print('Model has been trained for: %s epochs.\n' % model.epochs)
    except:
        model.epochs = 0
        print('Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0
        adv_loss = 0.0

        train_acc = 0
        valid_acc = 0
        adv_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            if attack:
                delta = attack(model, data, target)
                output = model(data + delta)
            else:
                output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                'Epoch: %s \t %s %% complete. %s seconds elapsed in epoch.' %\
                (epoch, 100 * (ii + 1) / len(train_loader), timer() - start),
                end='\r')

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for data, target in valid_loader:
                # Tensors to gpu
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()

                # Forward pass
                output = model(data)
                if attack:
                    delta = attack(model, data, target)
                    output_adv = model(data + delta)

                # Validation loss
                loss = criterion(output, target)
                # Multiply average loss times the number of examples in batch
                valid_loss += loss.item() * data.size(0)

                # Calculate validation accuracy
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                accuracy = torch.mean(
                    correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples
                valid_acc += accuracy.item() * data.size(0)

                if attack:
                    loss = criterion(output_adv, target)
                    adv_loss += loss.item() * data.size(0)
                    _, pred = torch.max(output_adv, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                    adv_acc += accuracy.item() * data.size(0)


            # Calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)
            if attack:
                adv_loss = adv_loss / len(valid_loader.dataset)

            # Calculate average accuracy
            train_acc = train_acc / len(train_loader.dataset)
            valid_acc = valid_acc / len(valid_loader.dataset)
            if attack:
                adv_acc = adv_acc / len(valid_loader.dataset)


            history.append([train_loss, valid_loss, adv_loss, train_acc, valid_acc, adv_acc])

            # Print training and validation results
            if (epoch + 1) % print_every == 0:
                print(
                    '\nEpoch: %s \tTraining Loss: %s \tValidation Loss: %s \tAdversarial Loss: %s' %\
                    (epoch, train_loss, valid_loss, adv_loss)
                )
                print(
                    '\t\tTraining Accuracy: %s %%\t Validation Accuracy: %s %%\t Adversarial Accuracy: %s %%' %\
                    (100 * train_acc, 100 * valid_acc, 100 * adv_acc)
                )

            # Save the model if validation loss decreases
            if valid_loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_file_name)
                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = valid_loss
                valid_best_acc = valid_acc
                best_epoch = epoch

            # Otherwise increment count of epochs with no improvement
            else:
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        '\nEarly Stopping! Total epochs: %s. Best epoch: %s with loss: %s and acc: %s%%' %\
                        (epoch, best_epoch, valid_loss_min, 100 * valid_acc)
                    )
                    total_time = timer() - overall_start
                    print(
                        '%s total seconds elapsed. %s seconds per epoch.' % \
                        (total_time, total_time/(epoch+1))
                    )

                    # Load the best state dict
                    model.load_state_dict(torch.load(save_file_name))
                    # Attach the optimizer
                    model.optimizer = optimizer

                    # Format history
                    history = pd.DataFrame(
                        history,
                        columns=[
                            'train_loss', 'valid_loss', 'adv_loss', 'train_acc',
                            'valid_acc', 'adv_acc'
                        ])
                    return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        '\nBest epoch: %s with loss: %s and acc: %s %%' %\
        (best_epoch, valid_loss_min, 100 * valid_acc)
    )
    print(
        '%s total seconds elapsed. %s seconds per epoch.' %\
        (total_time, total_time / (epoch))
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'adv_loss', 'train_acc', 'valid_acc', 'adv_acc'])
    return model, history


save_file_name = '/scratch/hle/data/caltech101_models/adversarial_training/vgg16-transfer-4.pt'
checkpoint_path = '/scratch/hle/data/caltech101_models/adversarial_training/vgg16-transfer-4.pth'
train_on_gpu = True
model, history = train(
    model,
    criterion,
    opt,
    dataloaders['train'],
    dataloaders['val'],
    save_file_name=save_file_name,
    max_epochs_stop=5,
    n_epochs=50,
    print_every=2,
    attack=fgsm)
