import os
import copy
import json
import torch
import logging

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from argparse import ArgumentParser
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

logger = logging.getLogger(__file__)

parser = ArgumentParser(description='Process solution arguments.')
parser.add_argument('--device', type=str, default='cpu', help='Device used for training (cuda or cpu)')
parser.add_argument('--name', type=str, choices=['alexnet', 'vgg11', 'vgg16', 'vgg19', 'resnet18', 'resnet50',
                                                 'resnet152'], help='One of pre-trained model names', default='vgg11')
parser.add_argument('--lr', type=int, help='Learning rate', default=1.0e-3)
parser.add_argument('--layers', type=int, help='Number of hidden layers excluding input and output', default=1)
parser.add_argument('--units', type=int, help='Number of hidden units per hidden layer', default=128)
parser.add_argument('--epochs', type=int, help='Number of epochs used for training', default=5)


def load_and_process_data():
    data_dir = 'flower_data'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            data_transforms[x]) for x in ['train', 'valid']}

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=4,
            shuffle=True,
            num_workers=4
        ) for x in ['train', 'valid']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

    return dataloaders, dataset_sizes


def get_cat_to_name():
    cat_to_name = None
    try:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
    except FileNotFoundError:
        return cat_to_name

    return cat_to_name


class DeepFeedForwardNet(nn.Module):
    def __init__(self, input_shape, layers=2, units=128, dropout=0.5):
        super(DeepFeedForwardNet, self).__init__()
        self.input_shape = input_shape
        self.input = nn.Linear(input_shape, units)
        self.out = nn.Linear(units, 102)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout

        self.layers = list()
        for i in range(layers):
            self.layers.append(nn.Linear(units, units))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        if self.dropout is not None:
            y = F.relu(self.dropout(self.input(x)))

            for layer in self.layers:
                y = F.relu(self.dropout(layer(y)))
        else:
            y = F.relu(self.input(x))

            for layer in self.layers:
                y = F.relu(layer(y))

        out = self.out(y)

        return out


def instantiate_model(name_, n_layers=1, n_units=128, lr_=0.001, dropout=None, device_='cpu'):
    logger.info("Instantiating model with params {}".format([name, n_layers, n_units, lr_, dropout]))
    model_rn = models.__dict__[name_](pretrained=True)

    if 'vgg' in name:
        input_features = 25088  # VGG input
    elif 'resnet' in name:
        input_features = 512  # Resnet input
    else:
        input_features = 9216  # Alexnet input

    dff_net = DeepFeedForwardNet(input_features, n_layers, n_units, dropout)
    dff_net = dff_net.to(device_)

    for param in model_rn.parameters():
        param.requires_grad = False

    # This happens because classifier's last layer doesn't have default names.
    if 'resnet' in name:
        model_rn.fc = dff_net
    else:
        model_rn.classifier = dff_net

    model_rn = model_rn.to(device_)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(dff_net.parameters(), lr=lr_, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    logger.info("Model, criterion, optimizer and lr-scheduler created.")

    return model_rn, criterion, optimizer, exp_lr_scheduler


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device_='cpu', num_epochs=20):
    logger.info("Training model with epochs:{}".format(num_epochs))
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device_)
                labels = labels.to(device_)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    logger.info('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


def save_model(model, optimizer, image_datasets, lr_scheduler_, criterion, layers_, hidden_units_, name_, epochs_, path_):
    directory = os.path.join(path_, '{}-dnn{}'.format(name_, layers_))
    if not os.path.exists(directory):
        os.makedirs(directory)

    model.class_to_index = image_datasets['train'].dataset.class_to_idx

    torch.save({
        'epochs': epochs,
        'model': model.state_dict(),
        'model_opt': optimizer.state_dict(),
        'classes': image_datasets['train'].dataset.class_to_idx,
        'lr_scheduler': lr_scheduler_.state_dict(),
        'criterion': criterion.state_dict()
    }, os.path.join(directory, '{}-dnn{}-{}_{}_{}.tar'.format(name_, layers_, hidden_units_, epochs_, 'checkpoint')))


if __name__ == '__main__':

    args = parser.parse_args()

    name = args.name
    lr = args.lr
    layers = args.layers
    hidden_units = args.units
    epochs = args.epochs

    dls, ds_sizes = load_and_process_data()

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            device = 'cpu'
            logger.warning('Cuda is not available on this machine, setting device to cpu')
        else:
            device = args.device
    else:
        device = args.device

    logger.info('Device mode set to {}'.format(device))

    nn, loss, opt, lr_scheduler = instantiate_model(name, layers, hidden_units, lr, 0.2, device)
    m = train_model(nn, loss, opt, lr_scheduler, dls, ds_sizes, device_=device, num_epochs=epochs)

    path = os.path.join(os.path.dirname(__file__), 'checkpoints')
    save_model(m, opt, dls, lr_scheduler, loss, layers, hidden_units, name, epochs, path)

    logger.info("Model trained and saved")
