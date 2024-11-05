"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import medmnist
import torch
from torch.nn import ReLU
from torchvision.models.resnet import ResNet
import numpy as np
from PIL import Image
from torch.autograd import Variable
from medmnist import INFO, Evaluator
import torchvision.transforms as transforms

from models import model_A


class GuidedBackprop:
    """
    Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, model: ResNet):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = next(self.model.children())
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
        Updates relu activation functions so that
            1- stores output in forward pass
            2- imputes zero for gradient values that are less than zero
        """

        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(
                grad_in[0], min=0.0
            )
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for module in self.model.modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = file_name + ".png"
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    save_image(gradient, path_to_file)


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1)
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = np.maximum(0, gradient) / gradient.max()
    neg_saliency = np.maximum(0, -gradient) / -gradient.min()
    return pos_saliency, neg_saliency

if __name__ == "__main__":
    data_flag = 'pathmnist'
    data_path = '../'
    model = "../best_model.pt"
    download = False
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    dataset = DataClass(root=data_path, split='test', transform=data_transform, size=64, download=download)
    img, class_vec = dataset[0]
    prep_img = torch.tensor(img).unsqueeze(0)
    prep_img = Variable(prep_img, requires_grad=True)
    class_vec = torch.tensor(class_vec)  # Convert to tensor if it's a numpy array

    # Now you can safely apply torch.argmax()
    target_class = torch.argmax(class_vec).item()
    file_name_to_export = "figs/gradient"

    # Ensure the model and input image are on the same device (GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained_model = torch.load(model).to(device)
    prep_img = prep_img.to(device)

    save_gradient_images(
        prep_img.detach().cpu().numpy()[0], file_name_to_export + "_original"
    )
    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export + "_Guided_BP_color")
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(
        grayscale_guided_grads, file_name_to_export + "_Guided_BP_gray"
    )
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + "_pos_sal")
    save_gradient_images(neg_sal, file_name_to_export + "_neg_sal")
    print("Guided backprop completed")
