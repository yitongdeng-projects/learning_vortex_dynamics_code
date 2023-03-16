import os
import shutil
import numpy as np
import imageio.v2 as imageio
import torch
import matplotlib.pyplot as plt
import copy
import configargparse

def to_numpy(x):
    return x.detach().cpu().numpy()
    
def to8b(x):
    return (255*np.clip(x,0,1)).astype(np.uint8)

# create gif from images using FFMPEG
def merge_imgs(framerate, save_dir):
    os.system('ffmpeg -hide_banner -loglevel error -y -i {0}/%03d.jpg -vf palettegen {0}/palette.png'.format(save_dir))
    os.system('ffmpeg -hide_banner -loglevel error -y -framerate {0} -i {1}/%03d.jpg -i {1}/palette.png -lavfi paletteuse {1}/output.gif'.format(framerate, save_dir))

# remove everything in dir
def remove_everything_in(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# read image
def imwrite(f, img):
    img = to8b(img)
    imageio.imwrite(f, img) # save frame as jpeg file

# generate grid coordinates 
def gen_grid(width, height, device):
    img_n_grid_x = width
    img_n_grid_y = height
    img_dx = 1./img_n_grid_y
    c_x, c_y = torch.meshgrid(torch.arange(img_n_grid_x), torch.arange(img_n_grid_y), indexing = "ij")
    img_x = img_dx * (torch.cat((c_x[..., None], c_y[..., None]), axis = 2) + 0.5).to(device) # grid center locations
    return img_x

# write image
def write_image(img_xy, outdir, i):
    img_xy = copy.deepcopy(img_xy)
    c_pred = np.flip(img_xy.transpose([1,0,2]), 0)
    img8b = to8b(c_pred)
    save_filepath = os.path.join(outdir, '{:03d}.jpg'.format(i))
    imageio.imwrite(save_filepath, img8b)

# write vortex (particles) with their velocities
def write_vorts(vorts_pos, vorts_uv, outdir, i):
    vorts_pos = copy.deepcopy(vorts_pos)
    pos = vorts_pos
    fig = plt.figure(num=1, figsize=(7, 7), clear=True)
    ax = fig.add_subplot()
    fig.subplots_adjust(0.1,0.1,0.9,0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    s = ax.scatter(pos[..., 0], pos[..., 1], s = 100)
    ax.quiver(pos[..., 0], pos[..., 1], vorts_uv[..., 0], vorts_uv[..., 1], color = "red", scale = 10.)
    fig.savefig(os.path.join(outdir, '{:03d}.jpg'.format(i)), dpi = 512//8)    

# write vorticity field
def write_vorticity(vort_img, outdir, i):
    vort_img = copy.deepcopy(vort_img)
    array = vort_img
    scale = array.shape[1]
    array = np.transpose(array, (1, 0, 2)) # from X, Y to Y, X
    fig = plt.figure(num=1, figsize=(8, 7), clear=True)
    ax = fig.add_subplot()
    fig.subplots_adjust(0.05,0.,0.9,1)
    ax.set_xlim([0, array.shape[0]])
    ax.set_ylim([0, array.shape[1]])
    p = ax.imshow(array, alpha = 0.75, vmin = -10, vmax = 10)
    fig.colorbar(p, fraction=0.04)
    fig.savefig(os.path.join(outdir, '{:03d}.jpg'.format(i)), dpi = 512//8) 

# write vortices over image
def write_visualization(img, vorts_pos, vorts_w, outdir, i, boundary = None):
    img = copy.deepcopy(img)
    # mask the out-of-bound area as green
    if boundary is not None:
        OUT = (boundary[0] >= -boundary[2]).cpu().numpy()
        img[OUT] = (0.5 * img[OUT])
        img[OUT, 1] += 0.5
    vorts_pos = copy.deepcopy(vorts_pos)
    vorts_w = copy.deepcopy(vorts_w)
    array = img
    scale = array.shape[1]
    pos = vorts_pos * scale 
    array = np.transpose(array, (1, 0, 2)) # from X, Y to Y, X
    fig = plt.figure(num=1, figsize=(8, 7), clear=True)
    ax = fig.add_subplot()
    fig.subplots_adjust(0.05,0.,0.9,1)
    ax.set_xlim([0, array.shape[0]])
    ax.set_ylim([0, array.shape[1]])
    s = ax.scatter(pos[..., 0], pos[..., 1], s = 100, c = vorts_w.flatten(), vmin = None, vmax = None)
    p = ax.imshow(array, alpha = 0.75)
    fig.colorbar(s, fraction=0.04)
    fig.savefig(os.path.join(outdir, '{:03d}.jpg'.format(i)), dpi = 512//8)

# convert rgb image to yuv parametrization
def rgb_to_yuv(_rgb_image):
    rgb_image = _rgb_image[..., None]
    matrix = np.array([[0.299, 0.587, 0.114],
                    [-0.14713, -0.28886, 0.436],
                    [0.615, -0.51499, -0.10001]]).astype(np.float32)
    matrix = torch.from_numpy(matrix)[None, None, None, ...].to(rgb_image.get_device())
    yuv_image = torch.einsum("abcde, abcef -> abcdf", matrix, rgb_image)
    return yuv_image.squeeze()

# command line tools
def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument('--seen_ratio', type=float, default=0.3333, 
                        help='fraction of input video available during training')
    parser.add_argument("--data_name", type=str, default='synthetic_1',
                        help='name of video data')
    parser.add_argument("--run_pretrain", type=bool, default = False, 
                        help='whether to run pretrain only')
    parser.add_argument("--test_only", type=bool, default = False, 
                        help='whether to run test only')
    parser.add_argument("--start_over", type=bool, default = False, 
                        help='whether to clear previous record on this experiment')
    parser.add_argument("--exp_name", type=str, default = "exp_0", 
                        help='the name of current experiment')
    parser.add_argument('--vort_scale', type=float, default=0.33, 
                        help='characteristic scale for vortices')
    parser.add_argument('--num_train', type=int, default=40000, 
                        help='number of training iterations')
    parser.add_argument("--init_vort_dist", type=float, default=1.0,
                        help='how spread out are init vortices')
    return parser
