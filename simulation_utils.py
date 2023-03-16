import torch
torch.manual_seed(123)
import numpy as np
import os
from learning_utils import calc_grad
import math
import time
from functorch import vmap
import torch.nn.functional as F

device = torch.device("cuda")
real = torch.float32

def RK1(pos, u, dt):
    return pos + dt * u

def RK2(pos, u, dt):
    p_mid = pos + 0.5 * dt * u
    return pos + dt * sample_grid_batched(u, p_mid)

def RK3(pos, u, dt):
    u1 = u
    p1 = pos + 0.5 * dt * u1
    u2 = sample_grid_batched(u, p1)
    p2 = pos + 0.75 * dt * u2
    u3 = sample_grid_batched(u, p2)
    return pos + dt * (2/9 * u1 + 1/3 * u2 + 4/9 * u3)

def advect_quantity_batched(quantity, u, x, dt, boundary):
    return advect_quantity_batched_BFECC(quantity, u, x, dt, boundary)

# pos: [num_queries, 2]
# if a backtraced position is out-of-bound, project it to the interior
def project_to_inside(pos, boundary):
    if boundary is None: # if no boundary then do nothing
        return pos
    sdf, sdf_normal, _ = boundary
    W, H = sdf.shape
    dx = 1./H
    pos_grid = (pos / dx).floor().long()
    pos_grid_x = pos_grid[...,0]
    pos_grid_y = pos_grid[...,1]
    pos_grid_x = torch.clamp(pos_grid_x, 0, W-1)
    pos_grid_y = torch.clamp(pos_grid_y, 0, H-1)
    sd_at_pos = sdf[pos_grid_x, pos_grid_y][...,None] # [num_queries, 1]
    sd_normal_at_pos = sdf_normal[pos_grid_x, pos_grid_y] # [num_queries, 2]
    OUT = (sd_at_pos >= -boundary[2]).squeeze(-1) # [num_queries]
    OUT_pos = pos[OUT] #[num_out_queries, 2]
    OUT_pos_fixed = OUT_pos - (sd_at_pos[OUT]+boundary[2]) * dx * sd_normal_at_pos[OUT] # remember to multiply by dx
    pos[OUT] = OUT_pos_fixed
    return pos


def index_take_2D(source, index_x, index_y):
    W, H, Channel = source.shape
    W_, H_ = index_x.shape
    index_flattened_x = index_x.flatten()
    index_flattened_y = index_y.flatten()
    sampled = source[index_flattened_x, index_flattened_y].view((W_, H_, Channel))
    return sampled

index_take_batched = vmap(index_take_2D)

# clipping used for MacCormack and BFECC
def MacCormack_clip(advected_quantity, quantity, u, x, dt, boundary):
    batch, W, H, _ = u.shape
    prev_pos = RK3(x, u, -1. * dt) # [batch, W, H, 2]
    prev_pos = project_to_inside(prev_pos.view((-1, 2)), boundary).view(prev_pos.shape)
    dx = 1./H
    pos_grid = (prev_pos / dx - 0.5).floor().long()
    pos_grid_x = torch.clamp(pos_grid[..., 0], 0, W-2)
    pos_grid_y = torch.clamp(pos_grid[..., 1], 0, H-2)
    pos_grid_x_plus = pos_grid_x + 1
    pos_grid_y_plus = pos_grid_y + 1
    BL = index_take_batched(quantity, pos_grid_x, pos_grid_y)
    BR = index_take_batched(quantity, pos_grid_x_plus, pos_grid_y)
    TR = index_take_batched(quantity, pos_grid_x_plus, pos_grid_y_plus)
    TL = index_take_batched(quantity, pos_grid_x, pos_grid_y_plus)
    stacked = torch.stack((BL, BR, TR, TL), dim = 0)
    maxed = torch.max(stacked, dim = 0).values # [batch, W, H, 3]
    mined = torch.min(stacked, dim = 0).values # [batch, W, H, 3]
    _advected_quantity = torch.clamp(advected_quantity, mined, maxed)
    return _advected_quantity

# SL
def advect_quantity_batched_SL(quantity, u, x, dt, boundary):
    prev_pos = RK3(x, u, -1. * dt) # [batch, W, H, 2]
    prev_pos = project_to_inside(prev_pos.view((-1, 2)), boundary).view(prev_pos.shape)
    new_quantity = sample_grid_batched(quantity, prev_pos)
    return new_quantity

# BFECC
def advect_quantity_batched_BFECC(quantity, u, x, dt, boundary):
    quantity1 = advect_quantity_batched_SL(quantity, u, x, dt, boundary)
    quantity2 = advect_quantity_batched_SL(quantity1, u, x, -1.*dt, boundary)
    new_quantity = advect_quantity_batched_SL(quantity + 0.5 * (quantity-quantity2), u, x, dt, boundary)
    new_quantity = MacCormack_clip(new_quantity, quantity, u, x, dt, boundary)
    return new_quantity

# MacCormack
def advect_quantity_batched_MacCormack(quantity, u, x, dt, boundary):
    quantity1 = advect_quantity_batched_SL(quantity, u, x, dt, boundary)
    quantity2 = advect_quantity_batched_SL(quantity1, u, x, -1.*dt, boundary)
    new_quantity = quantity1 + 0.5 * (quantity - quantity2)
    new_quantity = MacCormack_clip(new_quantity, quantity, u, x, dt, boundary)
    return new_quantity

# data = [batch, X, Y, n_channel]
# pos = [batch, X, Y, 2]
def sample_grid_batched(data, pos):
    data_ = data.permute([0, 3, 2, 1])
    pos_ = pos.clone().permute([0, 2, 1, 3])
    pos_ = (pos_ - 0.5) * 2
    F_sample_grid = F.grid_sample(data_, pos_, padding_mode = 'border', align_corners = False, mode = "bilinear")
    F_sample_grid = F_sample_grid.permute([0, 3, 2, 1])
    return F_sample_grid

# pos: [num_query, 2] or [batch, num_query, 2]
# vel: [batch, num_query, 2]
# mode: 0 for image, 1 for vort
def boundary_treatment(pos, vel, boundary, mode = 0):
    vel_after = vel.clone()
    batch, num_query, _ = vel.shape
    sdf = boundary[0] # [W, H]
    sdf_normal = boundary[1]
    if mode == 0:
        score = torch.clamp((sdf / -15.), min = 0.).flatten()
        inside_band = (score < 1.).squeeze(-1).flatten()
        score = score[None, ..., None]
        vel_after[:, inside_band, :] = score[:, inside_band, :] * vel[:, inside_band, :]
    else:
        W, H = sdf.shape
        dx = 1./H
        pos_grid = (pos / dx).floor().long()
        pos_grid_x = pos_grid[...,0]
        pos_grid_y = pos_grid[...,1]
        pos_grid_x = torch.clamp(pos_grid_x, 0, W-1)
        pos_grid_y = torch.clamp(pos_grid_y, 0, H-1)
        sd = sdf[pos_grid_x, pos_grid_y][...,None]
        sd_normal = sdf_normal[pos_grid_x, pos_grid_y]
        score = torch.clamp((sd / -75.), min = 0.)
        inside_band = (score < 1.).squeeze(-1)
        vel_normal = torch.einsum('bij,bij->bi', vel, sd_normal)[...,None] * sd_normal
        vel_tang = vel - vel_normal
        tang_at_boundary = 0.33
        vel_after[inside_band] = ((1.-tang_at_boundary) * score[inside_band] + tang_at_boundary) * vel_tang[inside_band] + score[inside_band] * vel_normal[inside_band]

    return vel_after

# simulate a single step
def simulate_step(img, img_x, vorts_pos, vorts_w, vorts_size, vel_func, dt, boundary):
    batch_size = vorts_pos.shape[0]
    img_x_flattened = img_x.view(-1, 2)
    if boundary is None:
        img_vel_flattened = vel_func(vorts_size, vorts_w, vorts_pos, img_x_flattened)
        img_vel = img_vel_flattened.view((batch_size, img_x.shape[0], img_x.shape[1], -1))
        new_img = torch.clip(advect_quantity_batched(img, img_vel, img_x, dt, boundary), 0., 1.)
        vorts_vel = vel_func(vorts_size, vorts_w, vorts_pos, vorts_pos)
        new_vorts_pos = RK1(vorts_pos, vorts_vel, dt)
    else:
        OUT = (boundary[0]>=-boundary[2])
        IN = ~OUT
        img_x_flattened = img_x.view(-1, 2)
        IN_flattened = IN.expand(img_x.shape[:-1]).flatten()
        img_vel_flattened = torch.zeros(batch_size, *img_x_flattened.shape).to(device)
        # only the velocity of the IN part will be computed, the rest will be left as 0
        img_vel_flattened[:, IN_flattened] = vel_func(vorts_size, vorts_w, vorts_pos, img_x_flattened[IN_flattened])
        img_vel_flattened = boundary_treatment(img_x_flattened, img_vel_flattened, boundary, mode = 0)
        img_vel = img_vel_flattened.view((batch_size, img_x.shape[0], img_x.shape[1], -1))
        new_img = torch.clip(advect_quantity_batched(img, img_vel, img_x, dt, boundary), 0., 1.)
        new_img[:, OUT] = img[:, OUT] # the image of the OUT part will be left unchanged
        vorts_vel = vel_func(vorts_size, vorts_w, vorts_pos, vorts_pos)
        vorts_vel = boundary_treatment(vorts_pos, vorts_vel, boundary, mode = 1)
        new_vorts_pos = RK1(vorts_pos, vorts_vel, dt)

    return new_img, new_vorts_pos, img_vel, vorts_vel

# simulate in batches
# img: the initial image
# img_x: the grid coordinates (meshgrid)
# vorts_pos: init vortex positions
# vorts_w: vorticity
# vorts_size: size
# num_steps: how many steps to simulate
# vel_func: how to compute velocity from vorticity
def simulate(img, img_x, vorts_pos, vorts_w, vorts_size, num_steps, vel_func, boundary = None, dt = 0.01):
    imgs = []
    vorts_poss = []
    img_vels = []
    vorts_vels = []
    for i in range(num_steps):
        img, vorts_pos, img_vel, vorts_vel = simulate_step(img, img_x, vorts_pos, vorts_w, vorts_size, vel_func, dt, boundary = boundary)
        imgs.append(img.clone())
        vorts_poss.append(vorts_pos.clone())
        img_vels.append(img_vel)
        vorts_vels.append(vorts_vel)

    return imgs, vorts_poss, img_vels, vorts_vels
