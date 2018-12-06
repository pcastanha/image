import json
import torch
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from torch import nn
from torchvision import models
from argparse import ArgumentParser

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

logger = logging.getLogger(__file__)

parser = ArgumentParser(description='Process solution arguments.')
parser.add_argument('--device', type=str, default='cpu', help='Device used for training (cuda or cpu)')
parser.add_argument('--image-path', type=str, help='Path to image')
parser.add_argument('--class-file-path', type=str, help='Path to file mapping classes to names',
                    default='cat_to_name.json')
parser.add_argument('--model-path', type=str, help='Path to model checkpoint')
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


def load_model_checkpoint_only(model_path_, device_='cpu'):
    logger.info('Loading checkpoint located at: {}'.format(model_path_))

    parameters = model_path_.split('/')

    if len(parameters) == 0:
        raise ValueError('Wrong model dir format')

    parameters = parameters[-1].split('-')

    name = parameters[0]  # Pretrained model name.
    layers = int(parameters[1].replace('dnn', ''))  # Number of layers in the checkpoint name.
    hidden_units = int(parameters[2].split('_')[0])

    # checkpoint = torch.load(model_dir_)
    checkpoint = torch.load(model_path_, map_location=lambda storage, loc: storage)
    model_checkpoint = checkpoint['model']
    net = models.__dict__[name](pretrained=True)

    if 'vgg' in name:
        input_features = 25088  # VGG input
    elif 'resnet' in name:
        input_features = 512  # Resnet input
    else:
        input_features = 9216  # Alexnet input

    dff_net = DeepFeedForwardNet(input_features, layers, hidden_units)
    # dff_net = dff_net.to(device_)

    for p in net.parameters():
        p.requires_grad = False

    if 'resnet' in name:
        net.fc = dff_net
    else:
        net.classifier = dff_net

    net = net.to(device_)
    net.load_state_dict(model_checkpoint)
    net.class_to_index = checkpoint['classes']

    logger.info('Model loaded')

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


def predict(image_path_, model_, device_='cpu', topk=5, class_mapping_file=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    logger.info("Starting prediction mode with top-{}".format(topk))

    image_as_tensor = torch.Tensor(process_image(Image.open(image_path_))).reshape([1, 3, 224, 224])
    reverse = {k: v for v, k in model_.class_to_index.items()}

    # Switch the model to eval mode
    model_.eval()
    model_ = model_.to(device_)
    image_as_tensor = image_as_tensor.to(device_)  # Switching input to same device as model.

    class_labels = get_cat_to_name(class_mapping_file if class_mapping_file is not None else "cat_to_name.json")

    # Predict input
    predicted_ = F.softmax(model_(image_as_tensor), dim=1)
    preds = predicted_.topk(topk)

    probs = [float(prob) for prob in preds[0][0]]
    classes = [class_labels[str(reverse[int(cls)])] for cls in preds[1][0]]

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
    model_path = args.model_path
    class_file = args.class_file_path
    k = args.k

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            device = 'cpu'
            logger.warning('Cuda is not available on this machine, setting device to cpu')
        else:
            device = args.device
    else:
        device = args.device

    logger.info('Device mode set to {}'.format(device))

    model = load_model_checkpoint_only(model_path, device)
    probs, classes = predict(image_path, model, device_=device, topk=1, class_mapping_file=class_file)
    logger.info(list(zip(probs, classes)))

    top_k_probs, top_k_classes = predict(image_path, model, device_=device, topk=k, class_mapping_file=class_file)
    logger.info(list(zip(top_k_probs, top_k_classes)))
