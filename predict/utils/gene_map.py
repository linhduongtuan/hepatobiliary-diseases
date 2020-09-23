import numpy as np
from .guided_bp import GuidedBackprop
from .grad_cam import GradCam
from torch.autograd import Variable


def gene_guided_bp(model, testloader):
    gbp = GuidedBackprop(model)
    gbs = []
    for imgs, labels in testloader:
        imgs = Variable(imgs, requires_grad=True)
        gbp.generate_gradients(imgs.cuda(), labels)
        g = imgs.grad.cpu().numpy().copy()
        gbs.append(g)
    gbs = np.concatenate(gbs)
    return gbs


def gene_grad_cam(model, testloader, weights):
    gc = GradCam(model)
    cam_hms = []
    cam_hois = []
    grad_cams = []
    grad_cam_hms = []
    grad_cam_hois = []
    for imgs, labels in testloader:
        # imgs = Variable(imgs, requires_grad=True)
        cam_hm, cam_hoi, grad_cam, grad_cam_hm, grad_cam_hoi = gc.cam_heatmap(imgs, labels, weights)
        cam_hms += cam_hm
        cam_hois += cam_hoi
        grad_cams += grad_cam
        grad_cam_hms += grad_cam_hm
        grad_cam_hois += grad_cam_hoi
    # cam_hms = np.concatenate(cam_hms)
    # cam_hois = np.concatenate(cam_hois)
    # grad_cams = np.concatenate(grad_cams)
    # grad_cam_hms = np.concatenate(grad_cam_hms)
    # grad_cam_hois = np.concatenate(grad_cam_hois)
    return cam_hms, cam_hois, grad_cams, grad_cam_hms, grad_cam_hois
