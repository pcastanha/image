
# coding: utf-8

import os
import copy
import json
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


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


# ### Best model: 
# 
#         {nn.Linear(512, 102)}* - Best val Acc: 0.924205
# 
#         {self.hid1 = nn.Linear(512, 1024)
#         self.hid2 = nn.Linear(1024, 256)
#         self.out = nn.Linear(256, 102)
#         self.dropout = nn.Dropout(dropout)
#     
#         x = F.relu(self.dropout(self.hid1(x)))
#         x = F.relu(self.hid2(x))
#         out = self.out(x)} - Best val Acc: 0.911980
#         
#         {self.hid1 = nn.Linear(512, 512)
#         self.hid2 = nn.Linear(512, 256)
#         self.out = nn.Linear(256, 102)
#         self.dropout = nn.Dropout(dropout)
# 
#         x = F.relu(self.dropout(self.hid1(x)))
#         x = F.relu(self.hid2(x))
#         out = self.out(x)} - Best val Acc: 0.899756


# Save the checkpoint
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


def load_model_checkpoint_only(model_dir, device='cpu'):
    checkpoint = torch.load(model_dir)
    model_checkpoint = checkpoint['model']
    
    net = models.resnet18(pretrained=True)
    dff_net = DeepFeedForwardNet()

    for p in net.parameters():
        p.requires_grad = False

    net.fc = dff_net
    net = net.to(device)

    net.load_state_dict(model_checkpoint)
    net.class_to_index = checkpoint['classes']
    
    return net


# Loads a checkpoint and rebuild the model
def load_and_rebuild_checkpoint(model_dir, device='cpu'):
    checkpoint = torch.load(model_dir)
    
    epochs = checkpoint['epochs']
    model = checkpoint['model']
    optimizer = checkpoint['model_opt']
    scheduler = checkpoint['lr_scheduler']
    criterion = checkpoint['criterion']
    class_mappings = checkpoint['classes']
    
    t_net = models.resnet18(pretrained=True)
    t_dff_net = DeepFeedForwardNet()

    for p in t_net.parameters():
        p.requires_grad = False

    t_net.fc = t_dff_net
    t_net = t_net.to(device)
    t_crit = nn.CrossEntropyLoss()
    t_opt = optim.SGD(t_net.fc.parameters(), lr=0.001, momentum=0.9)
    t_sched = lr_scheduler.StepLR(t_opt, step_size=7, gamma=0.1)

    t_net.load_state_dict(model)
    t_crit.load_state_dict(criterion)
    t_opt.load_state_dict(optimizer)
    t_sched.load_state_dict(scheduler)
    
    return t_net, t_opt, t_crit, t_sched, epochs, class_mappings


def resize_and_keep_ar(pil_image, smaller_side):
    w, h = pil_image.size

    if w < h:
        new_w = smaller_side
        w_ratio = (new_w / float(pil_image.size[0]))
        new_h = int((float(pil_image.size[1]) * float(w_ratio)))
    else:
        new_h = smaller_side
        h_ratio = (new_h / float(pil_image.size[1]))
        new_w = int((float(pil_image.size[0]) * float(h_ratio)))

    return pil_image.resize((new_w, new_h))


def center_square_crop(pil_image, size):
    width, height = pil_image.size
    
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    
    return pil_image.crop((left, top, right, bottom))


def normalize_image(pil_image):    
    image_as_array = np.array(pil_image) 
    img = torch.from_numpy(image_as_array.transpose((2, 0, 1)))
    
    if isinstance(img, torch.ByteTensor):
        img = img.float().div(255)
    
    for t, m, s in zip(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
        t.sub_(m).div_(s)
        
    return img.numpy()


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    _CROP_DIM = 224
    _RESIZE_SIZE = 256
    
    im = Image.fromarray(np.array(image))
    
    resized = resize_and_keep_ar(im, _RESIZE_SIZE)
    cropped = center_square_crop(resized, _CROP_DIM)
    normalized = normalize_image(cropped)
    
    return normalized


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Flower Species")

    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image_as_tensor = torch.Tensor(process_image(Image.open(image_path))).reshape([1, 3, 224, 224])
    reverse = {k: v for v, k in model.class_to_index.items()}

    # Switch the model to eval mode
    model.eval()

    # Predict input
    predicted = F.softmax(model(image_as_tensor), dim=1)
    preds = predicted.topk(topk)

    probs = [float(prob) for prob in preds[0][0]]
    classes = [reverse[int(cls)] for cls in preds[1][0]]
    
    return probs, classes


def plot_charts(tensor_image, true_label, pred_classes, pred_probas):
    ''' Auxiliar function used to create and plot charts given an image and some required information.
    '''
    
    fig, (ax1, ax2) = plt.subplots(2, figsize=(5, 10))

    cat_to_name = get_cat_to_name()
    
    y_pos = np.arange(len(pred_classes))
    performance = np.asarray(pred_probas)
    classes = (cat_to_name[class_] for class_ in pred_classes)

    ax2.barh(y_pos, performance, align='center', color='blue', ecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')
    ax2.set_title('Predicted classes')

    fig.subplots_adjust(hspace=0.3)

    imshow(tensor_image, ax=ax1, title=cat_to_name[true_label])

    plt.show()


def sanity_check(image_path, model, topk=5):
    prb_, cls_ = predict(image_path, model, topk)
    
    tensor_image = torch.Tensor(process_image(Image.open(image_path)))
    true_label = image_path.split('/')[2]
    
    plot_charts(tensor_image, true_label, cls_, prb_)
