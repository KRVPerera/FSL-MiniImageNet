# import necessary packages
import torch
import torch.nn as nn
from tempfile import TemporaryDirectory
import os
import time
import matplotlib.pyplot as plt

def eval_method(net, data_loader, criterion=nn.CrossEntropyLoss()):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    net.eval()
    correct = 0.0
    num_images = 0.0
    loss = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i_batch, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        outs = net(images)
        loss += criterion(outs, labels).item()
        _, predicted = torch.max(outs.data, 1)
        correct += (predicted == labels).sum().item()
        num_images += len(labels)
        print('testing -> batch: %d correct: %d num of images: %d' % (i_batch, correct, num_images) + '\r', end='')
    acc = correct / num_images
    loss /= len(data_loader)
    return acc, loss

def eval_func(net, data_loader):
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
        preds = outs.argmax(dim=1)
        correct += preds.eq(labels).sum()
        num_images += len(labels)

    acc = correct / num_images
    return acc


def train_fine_tuning(net, dataloaders, criterion, optimizer, scheduler, num_epochs, learning_rate):
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if use_cuda:
        net = net.cuda()

    training_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        net.train()
        correct = 0.0
        num_images = 0.0
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

        acc = correct / num_images
        acc_test, test_loss = eval_method(net, test_loader, criterion)
        average_loss = total_loss / len(train_loader)
        training_losses.append(average_loss)
        val_losses.append(test_loss)
        print('epoch: %d, accuracy: %f, avg. loss: %f, test accuracy: %f' % (epoch, acc, average_loss, acc_test))

        scheduler.step()

    return net, training_losses, val_losses

# references: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, dataset_sizes):
    since = time.time()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    else:
        print("No Cude")
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        train_losses = [] # Log the training loss
        train_avg_losses = [] # Log the training loss
        val_losses = []
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
                correct = 0.0 # used to accumulate number of correctly recognized images
                num_images = 0.0 # used to accumulate number of images
                epoch_loss = 0.0
                
                # Iterate over data.
                for i_batch, (images, labels) in enumerate(dataloaders[phase]):
                    if use_cuda:
                        images = images.cuda()
                        labels = labels.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()

                    # statistics
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels)
                if phase == 'train':
                    # scheduler.step()
                    train_losses.append(loss.item())
                    train_avg_losses.append(epoch_loss / dataset_sizes[phase])
                else:
                    val_losses.append(loss.item())

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
        
        plt.title('Train and Validation losses')
        plt.plot(train_losses, label='training losses')
        plt.plot(val_losses, label='validation losses')
        plt.legend()
        plt.show()

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model