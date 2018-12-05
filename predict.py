import json
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from torch import nn
from torchvision import models
from argparse import ArgumentParser

parser = ArgumentParser(description='Process solution arguments.')
parser.add_argument('--device', type=str, default='cpu', help='Device used for training (cuda or cpu)')
parser.add_argument('--image-path', type=str, help='Path to image')
parser.add_argument('--model-dir', type=str, help='Path to model checkpoint')
parser.add_argument('--k', type=int, help='Number of elements to be used by top-k', default=1)


def get_cat_to_name(path='cat_to_name.json'):
    cat_to_name = None
    try:
        path_list = path.split('.')
        if len(path_list) == 0:
            raise ValueError('Invalid path')
        elif len(path_list) > 0 and path_list[-1] != 'json':
            raise ValueError('Invalid file format. Json type should be selected')
        else:
            with open(path, 'r') as f:
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


def load_model_checkpoint_only(model_dir_, device_='cpu'):
    checkpoint = torch.load(model_dir_)
    model_checkpoint = checkpoint['model']

    net = models.resnet18(pretrained=True)
    dff_net = DeepFeedForwardNet()

    for p in net.parameters():
        p.requires_grad = False

    net.fc = dff_net
    net = net.to(device_)

    net.load_state_dict(model_checkpoint)
    net.class_to_index = checkpoint['classes']

    return net


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


def sanity_check(image_path_, model_, topk=5):
    prb_, cls_ = predict(image_path_, model_, topk)

    tensor_image = torch.Tensor(process_image(Image.open(image_path)))
    true_label = image_path.split('/')[2]

    plot_charts(tensor_image, true_label, cls_, prb_)


if __name__ == '__main__':

    args = parser.parse_args()

    image_path = args.image_path
    model_dir = args.model_dir
    k = args.k

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            device = 'cpu'
            print('Cuda is not available in this machine, setting device to cpu')
        else:
            device = args.device
    else:
        device = args.device

    model = load_model_checkpoint_only(model_dir, device)
    predicted = predict(image_path, model, topk=1)
    print(predicted)

    top_k = predict(image_path, model, k)
    print(top_k)
