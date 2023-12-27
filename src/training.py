# import necessary packages
import torch
import torch.nn as nn
from tempfile import TemporaryDirectory
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

def eval(net, data_loader):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    net.eval()
    correct = 0.0
    num_images = 0.0
    for i_batch, (images, labels) in enumerate(data_loader):
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
        outs = net(images) 
#         _, preds = outs.max(1)
        preds = outs.argmax(dim=1)
        correct += preds.eq(labels).sum()
        num_images += len(labels)

    acc = correct / num_images
    return acc


# training function
def train(net, train_loader, valid_loader, learning_rate=0.01, momentum=0.9, weight_decay=0.0001, num_epochs=3):

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(params= net.parameters(), lr=LEARNING_RATE, momentum=momentum, weight_decay=0.0001)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999))
    scheduler = StepLR(optimizer, step_size=7, gamma=0.01)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()

    training_losses = []
    training_losses_epoch = []
    val_losses = []
    for epoch in range(num_epochs):
        net.train()
        correct = 0.0  # used to accumulate number of correctly recognized images
        num_images = 0.0  # used to accumulate number of images
        total_loss = 0.0

        for i_batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            output_train = net(images)
            loss = criterion(output_train, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicts = output_train.argmax(dim=1)
            correct += predicts.eq(labels).sum().item()
            num_images += len(labels)
            total_loss += loss.item()
            training_losses.append(loss.item())

            print('training -> epoch: %d, batch: %d, loss: %f' % (epoch, i_batch, loss.item()) + '\r', end='')

        print()
        acc = correct / num_images
        acc_eval, val_loss = eval(net, valid_loader)
        average_loss = total_loss / len(train_loader)
        val_losses.append(val_loss)
        training_losses_epoch.append(average_loss)
        print('\nepoch: %d, lr: %f, accuracy: %f, avg. loss: %f, valid accuracy: %f valid loss: %f\n' % (epoch, optimizer.param_groups[0]['lr'], acc, average_loss, acc_eval, val_loss))

        scheduler.step()

    return net, training_losses, training_losses_epoch, val_losses

# training function
def train3(net, train_loader, valid_loader, learning_rate=0.01, momentum=0.9, weight_decay=0.0001, num_epochs=3):
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params= net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
        
    for epoch in range(num_epochs):
        net.train() 
        correct = 0.0 # used to accumulate number of correctly recognized images
        num_images = 0.0 # used to accumulate number of images
        for i_batch, (images, labels) in enumerate(train_loader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            # TODO: rest of the training code
            # your code here, including the forward propagation (0.75 points), 
            # backward propagation (0.75 points) and calculating the accuracy (0.5 points)
            output_train = net(images)
            loss = loss_function(output_train, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predicts = output_train.argmax(dim=1)
            correct += predicts.eq(labels).sum().item()
            num_images += len(labels)
            
        acc = correct / num_images
        acc_eval = eval(net, valid_loader)
        print('epoch: %d, lr: %f, accuracy: %f, loss: %f, valid accuracy: %f' % (epoch, optimizer.param_groups[0]['lr'], acc, loss.item(), acc_eval))

    return net


# from torch.optim.lr_scheduler import PolynomialLR
# https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import StepLR
import time

# training function
def train_me(net, train_loader, valid_loader, learning_rate=0.01, momentum=0.9, weight_decay=0.0001, num_epochs=3, T_0=10):
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params= net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # scheduler = PolynomialLR(optimizer, total_iters=4, power=1.0, verbose = False)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    # iters = len(train_loader)
    # scheduler2 = CosineAnnealingWarmRestarts(optimizer, T_0, verbose = False, T_mult = 1, eta_min=1e-6)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    
    loss_log = [] # Log the training loss
    for epoch in range(num_epochs):
        net.train() 
        
        correct = 0.0 # used to accumulate number of correctly recognized images
        num_images = 0.0 # used to accumulate number of images
        for i_batch, (images, labels) in enumerate(train_loader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            output_train = net(images)
            loss = loss_function(output_train, labels)
            loss_log.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predicts = output_train.argmax(dim=1)
            correct += predicts.eq(labels).sum().item()
            num_images += len(labels)
            # scheduler2.step(epoch + i_batch / iters)
            
        acc = correct / num_images
        acc_eval = eval(net, valid_loader)
        scheduler.step()
        # scheduler1.step(loss.item())
        if acc_eval >= acc:
            print('\x1b[31mepoch: %2d, lr: %f, accuracy: %f, loss: %f, valid accuracy: %f\x1b[0m' % (epoch, optimizer.param_groups[0]['lr'], acc, loss.item(), acc_eval))
        else:
            print('epoch: %2d, lr: %f, accuracy: %f, loss: %f, valid accuracy: %f' % (epoch, optimizer.param_groups[0]['lr'], acc, loss.item(), acc_eval))
    
    plt.title('Training loss:')
    plt.plot(loss_log)
    plt.show()
    return net


# references: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model