import os
import copy
import json
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from argparse import ArgumentParser
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


parser = ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))


device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    def __init__(self, dropout=0.2):
        super(DeepFeedForwardNet, self).__init__()
        self.hid1 = nn.Linear(512, 1024)
        self.hid2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 102)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.dropout(self.hid1(x)))
        x = F.relu(self.hid2(x))
        out = self.out(x)

        return out


def instantiate_model():
    dff_net = DeepFeedForwardNet()
    model_rn = models.resnet18(pretrained=True)

    for param in model_rn.parameters():
        param.requires_grad = False

    model_rn.fc = dff_net
    model_rn = model_rn.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_rn.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model_rn, criterion, optimizer, exp_lr_scheduler


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=20):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


def save_model(model, optimizer, image_datasets, lr_scheduler, criterion, path, name, epochs):
    directory = os.path.join(path, name, '{}-{}'.format('linear', '2'))
    if not os.path.exists(directory):
        os.makedirs(directory)

    model.class_to_index = image_datasets['train'].class_to_idx

    torch.save({
        'epochs': epochs,
        'model': model.state_dict(),
        'model_opt': optimizer.state_dict(),
        'classes': image_datasets['train'].class_to_idx,
        'lr_scheduler': lr_scheduler.state_dict(),
        'criterion': criterion.state_dict()
    }, os.path.join(directory, '{}_{}.tar'.format(epochs, 'checkpoint')))