import torch
import numpy as np
from PIL import Image
from .image_tool import apply_colormap_on_image, get_raw_img


class GradCam(object):
    """
        Produces class activation map
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam(self, image, target, weights, type):
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        if type == 'grad_cam':
            cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (
                np.max(cam) - np.min(cam) + 1e-5)  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((image.shape[2], image.shape[1]), Image.ANTIALIAS))

        raw_img = get_raw_img(image.detach())

        # heatmap, heatmap_on_image = apply_colormap_on_image(raw_img, cam, 'hsv')
        heatmap, heatmap_on_image = apply_colormap_on_image(raw_img, cam, 'jet')

        return cam, heatmap, heatmap_on_image

    def cam_heatmap(self, input_image, labels, fc_weights):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        model_output, conv_outputs = self.model(input_image.cuda(), True, True)

        # Target for backprop

        one_hot_output = torch.FloatTensor(model_output.size()).zero_()
        for i, l in enumerate(labels):
            one_hot_output[i][l] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output.cuda(), retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.model.gradients.cpu().data.numpy()
        grad_weight = np.mean(guided_gradients, axis=(2, 3))  # Take averages for each gradient
        conv_outputs = conv_outputs.cpu().data.numpy()
        cam_hms = []
        cam_hois = []
        grad_cams = []
        grad_cam_hms = []
        grad_cam_hois = []
        for i in range(input_image.shape[0]):
            fc_weight = fc_weights[labels[i]]
            _, cam_h, cam_hoi = self.generate_cam(input_image[i], conv_outputs[i], fc_weight, 'cam')
            grad_cam, grad_cam_h, grad_cam_hoi = self.generate_cam(input_image[i], conv_outputs[i], grad_weight[i], 'grad_cam')

            cam_hms.append(cam_h)
            cam_hois.append(cam_hoi)

            grad_cams.append(grad_cam)
            grad_cam_hms.append(grad_cam_h)
            grad_cam_hois.append(grad_cam_hoi)

        return cam_hms, cam_hois, grad_cams, grad_cam_hms, grad_cam_hois
