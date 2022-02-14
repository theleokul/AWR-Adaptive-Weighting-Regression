import torch
from torchvision.transforms import Normalize, Compose, ToPILImage, ToTensor
import numpy as np
from prettytable import PrettyTable



def xyz2uvd(pts, paras, flip=1):
    # paras: [fx, fy, fu, fv]
    pts_uvd = pts.copy()
    pts_uvd = pts_uvd.reshape(-1, 3)
    pts_uvd[:, 1] *= flip
    pts_uvd[:, :2] = pts_uvd[:, :2] * paras[:2] / pts_uvd[:, 2:] + paras[2:]

    return pts_uvd.reshape(pts.shape).astype(np.float32)


def uvd2xyz(pts, paras, flip=1):
    # paras: (fx, fy, fu, fv)
    pts_xyz = pts.copy()
    pts_xyz = pts_xyz.reshape(-1, 3)
    pts_xyz[:, :2] = (pts_xyz[:, :2] - paras[2:]) * pts_xyz[:, 2:] / paras[:2]
    pts_xyz[:, 1] *= flip

    return pts_xyz.reshape(pts.shape).astype(np.float32)


def tensor_stack(x, mode='hstack'):
    bs = x.shape[0]
    
    if mode == 'hstack':
        return torch.cat(x.chunk(bs), dim=3).squeeze(0)
    elif mode == 'vstack':
        return torch.cat(x.chunk(bs), dim=2).squeeze(0)


def topil(x, normalize=True):
    if len(x.shape) == 4:
        if x.shape[0] == 1:
            x = x.squeeze(0)
        else:
            print('Input is not a single image')
            x = tensor_stack(x)
            
    if normalize:
        x = x.clamp(-1, 1).mul(0.5).add(0.5)
    else:
        x = x.clamp(0, 1)
        
    return ToPILImage()(x.cpu())


def totensor(img, normalize=True):
    img = ToTensor()(img).unsqueeze(0)
    if normalize:
        img = img.add(-0.5).mul(2)
    return img


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
