import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import LambdaLR, StepLR
import numpy as np
import torch.nn.functional as F


device = torch.device("cuda")
real = torch.float32

L2_Loss = nn.MSELoss().cuda()

class SineResidualBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # add shortcut
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
            )

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        out = torch.sin(self.omega_0 * self.linear(input))
        out += self.shortcut(input)
        out = nn.functional.relu(out)
        return out

class Dynamics_Net(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 1
        out_dim = 1
        width = 40
        self.layers = nn.Sequential(SineResidualBlock(in_dim, width, omega_0=1., is_first=True),
                                SineResidualBlock(width, width, omega_0=1.),
                                SineResidualBlock(width, width, omega_0=1.),
                                SineResidualBlock(width, width, omega_0=1.),
                                nn.Linear(width, out_dim),
                                )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

class Position_Net(nn.Module):
    def __init__(self, num_vorts):
        super().__init__()
        in_dim = 1
        out_dim = num_vorts * 2
        self.layers = nn.Sequential(SineResidualBlock(in_dim, 64, omega_0=1., is_first=True),
                                SineResidualBlock(64, 128, omega_0=1.),
                                SineResidualBlock(128, 256, omega_0=1.),
                                nn.Linear(256, out_dim)
                                )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


def create_bundle(logdir, num_vorts, decay_step, decay_gamma, pretrain_dir = None):
    model_len = Dynamics_Net().to(device)
    model_pos = Position_Net(num_vorts).to(device)
    grad_vars = list(model_len.parameters())
    grad_vars2 = list(model_pos.parameters())
    ##########################
    # Load checkpoints
    ckpts = [os.path.join(logdir, f) for f in sorted(os.listdir(logdir)) if 'tar' in f]
    pretrain_ckpts = []
    if pretrain_dir:
        pretrain_ckpts = [os.path.join(pretrain_dir, f) for f in sorted(os.listdir(pretrain_dir)) if 'tar' in f]

    if len(ckpts) <= 0: # no checkpoints to load
        w_pred = torch.zeros(num_vorts, 1, device = device, dtype = real)
        w_pred.requires_grad = True
        size_pred = torch.zeros(num_vorts, 1, device = device, dtype = real)
        size_pred.requires_grad = True
        start = 0
        optimizer = torch.optim.Adam([{'params': grad_vars}, \
                                    {'params': grad_vars2, 'lr':3.e-4},\
                                    {'params': w_pred, 'lr':5.e-3},\
                                    {'params': size_pred, 'lr':5.e-3}], lr=1.e-3, betas=(0.9, 0.999))
        # Load pretrained if there is one and no checkpoint exists
        if len(pretrain_ckpts) > 0:
            pre_ckpt_path = pretrain_ckpts[-1]
            print ("[Initialize] Has pretrained available, reloading from: ", pre_ckpt_path)
            pre_ckpt = torch.load(pre_ckpt_path)
            model_pos.load_state_dict(pre_ckpt['model_pos_state_dict'])

    else: # has checkpoints to load:
        ckpt_path = ckpts[-1]
        print ("[Initialize] Has checkpoint available, reloading from: ", ckpt_path)
        ckpt = torch.load(ckpt_path)
        start = ckpt['global_step']
        w_pred = ckpt['w_pred']
        w_pred.requires_grad = True
        size_pred = ckpt['size_pred']
        size_pred.requires_grad = True
        model_len.load_state_dict(ckpt["model_len_state_dict"])
        model_pos.load_state_dict(ckpt["model_pos_state_dict"])
        optimizer = torch.optim.Adam([{'params': grad_vars}, \
                                    {'params': grad_vars2, 'lr':3.e-4},\
                                    {'params': w_pred, 'lr':5.e-3},\
                                    {'params': size_pred, 'lr':5.e-3}], lr=1.e-3, betas=(0.9, 0.999))
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    
    lr_scheduler = StepLR(optimizer, step_size = decay_step, gamma = decay_gamma)


    ##########################
    net_dict = {
        'model_len' : model_len,
        'model_pos' : model_pos,
        'w_pred' : w_pred,
        'size_pred' : size_pred,
    }

    return net_dict, start, grad_vars, optimizer, lr_scheduler


# vels: [batch, width, height, 2]
def calc_div(vels):
    batch_size, width, height, D = vels.shape
    dx = 1./height
    du_dx = 1./(2*dx) * (vels[:, 2:, 1:-1, 0] - vels[:, :-2, 1:-1, 0])
    dv_dy = 1./(2*dx) * (vels[:, 1:-1, 2:, 1] - vels[:, 1:-1, :-2, 1])
    return du_dx + dv_dy

# field: [batch, width, height, 1]
def calc_grad(field):
    batch_size, width, height, _ = field.shape
    dx = 1./height
    df_dx = 1./(2*dx) * (field[:, 2:, 1:-1] - field[:, :-2, 1:-1])
    df_dy = 1./(2*dx) * (field[:, 1:-1, 2:] - field[:, 1:-1, :-2])
    return torch.cat((df_dx, df_dy), dim = -1)

def calc_vort(vel_img, boundary = None): # compute the curl of velocity
    W, H, _ = vel_img.shape
    dx = 1./H
    vort_img = torch.zeros(W, H, 1, device = device, dtype = real)
    u = vel_img[...,[0]]
    v = vel_img[...,[1]]
    dvdx = 1/(2*dx) * (v[2:, 1:-1] - v[:-2, 1:-1])
    dudy = 1/(2*dx) * (u[1:-1, 2:] - u[1:-1, :-2])
    vort_img[1:-1, 1:-1] = dvdx - dudy
    if boundary is not None:
        # set out-of-bound pixels to 0 because velocity undefined there
        OUT = (boundary[0] >= -boundary[2] - 4)
        vort_img[OUT] *= 0
    return vort_img

# sdf: [W, H]
# sdf normal: [W, H, 2]
def calc_sdf_normal(sdf):
    W, H = sdf.shape
    sdf_normal = torch.zeros((W, H, 2)).cuda() #[W, H, 2]
    sdf_normal[1:-1, 1:-1] = calc_grad(sdf[None,...,None])[0] # outward pointing [W, H, 2]
    sdf_normal = F.normalize(sdf_normal, dim = -1, p = 2)
    return sdf_normal

# vorts_pos: [batch, num_vorts, 2] 
# query_pos: [num_query, 2] or [batch, num_query, 2]
# return: [batch, num_queries, num_vorts, 2]
def calc_diff_batched(_vorts_pos, _query_pos):
    vorts_pos = _vorts_pos[:, None, :, :] # [batch, 1, num_vorts, 2]
    if len(_query_pos.shape) > 2:
        query_pos = _query_pos[:, :, None, :] # [batch, num_query, 1, 2]
    else: 
        query_pos = _query_pos[None, :, None, :] # [1, num_query, 1, 2]
    diff = query_pos - vorts_pos # [batch, num_queries, num_vorts, 2]
    return diff


# vorts_pos shape: [batch, num_vorts, 2]
# vorts_w shape: [num_vorts, 1] or [batch, num_vorts, 1]
# vorts_size shape: [num_vorts, 1] or [batch, num_vorts, 1]
def vort_to_vel(network_length, vorts_size, vorts_w, vorts_pos, query_pos, length_scale):
    diff = calc_diff_batched(vorts_pos, query_pos) # [batch_size, num_query, num_query, 2]
    # some broadcasting
    if len(vorts_size.shape) > 2:
        blob_size = vorts_size[:, None, ...] # [batch, 1, num_vorts, 1]
    else:
        blob_size = vorts_size[None, None, ...] # [1, 1, num_vorts, 1] 
    if len(vorts_w.shape) > 2:
        vorts_w = vorts_w[:, None, ...] # [batch, num_query, num_vort, 1]
    else:
        vorts_w = vorts_w[None, None, ...] # [1, 1, num_vort, 1]

    diff = calc_diff_batched(vorts_pos, query_pos)
    dist = torch.norm(diff, dim = -1, p = 2, keepdim = True) 
    dist_not_zero = dist > 0.0 

    # cross product in 2D
    R = diff.flip([-1]) # (x, y) becomes (y, x)
    R[..., 0] *= -1 # (y, x) becomes (-y, x)
    R = F.normalize(R, dim = -1)

    dist = dist / (blob_size/length_scale)
    dist[dist_not_zero] = torch.pow(dist[dist_not_zero], 0.3) 
    magnitude = network_length(dist)
    magnitude = magnitude / (blob_size/length_scale)

    result = magnitude * R * vorts_w
    result = torch.sum(result, dim = -2) # [batch_size, num_queries, 2]

    return result