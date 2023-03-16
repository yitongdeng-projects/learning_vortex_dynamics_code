from io_utils import *
from simulation_utils import *
from learning_utils import *
torch.manual_seed(123)
import sys
import os
from functorch import jacrev, vmap

device = torch.device("cuda")
real = torch.float32

# command line parse
parser = config_parser()
args = parser.parse_args()

# some switches
run_pretrain = args.run_pretrain # if this is set to true, then only pretrain() will be run
test_only = args.test_only # if this is set to true, then only test() will be run
start_over = args.start_over # if this is set to true, then the logs/[exp_name] dir. will be emptied

# some hyperparameters
print("[Train] Number of training iters: ", args.num_train)
num_iters = args.num_train # total number of training iterations
decimate_point = 20000 # LR decimates at this point
decay_gamma = 0.99
decay_step = max(1, int(decimate_point/math.log(0.1, decay_gamma))) # decay once every (# >= 1) learning steps
save_ckpt_every = 1000
test_every = 1000
print_every = 20
num_sims = 2 # the "m" param in paper
batch_size = 4

# load data
datadir = os.path.join('data', args.data_name)
print("[Data] Load from path: ", datadir)
imgs = torch.from_numpy(np.load(os.path.join(datadir, 'imgs.npy'))).to(device).type(real)
try:
    sdf = torch.from_numpy(np.load(os.path.join(datadir, 'sdf.npy'))).to(device).type(real)
except:
    print("[Boundary] SDF file doesn't exist, no boundary")
    boundary = None
else:
    print("[Boundary] SDF file exists, has boundary")
    sdf = torch.flip(sdf, [0])
    sdf = torch.permute(sdf, (1, 0))
    sdf_normal = calc_sdf_normal(sdf)
    # 1. signed distance field
    # 2. unit normal of sdf
    # 3. thickness (in pixels) 
    boundary = (sdf, sdf_normal, 2)

num_total_frames = imgs.shape[0] # seen + unseen frames
print("[Data] Number of frames we have: ", num_total_frames)
imgs = imgs[:math.ceil(num_total_frames * args.seen_ratio)] # select a number of frames to be revealed for training
num_frames, width, height, num_channels = imgs.shape
print("[Data] Number of frames revealed: ", num_frames)
num_unseen_frames = num_total_frames - num_frames
print("[Data] Number of frames concealed: ", num_unseen_frames)
timestamps = torch.arange(num_frames).type(real)[..., None].to(device) * 0.01
num_available_frames = num_frames - num_sims
probs = torch.ones((num_available_frames), device = device, dtype = real)

# setup initial vort (as a vorts_num_x X vorts_num_y grid)
vorts_num_x = 4
vorts_num_y = 4
num_vorts = vorts_num_x * vorts_num_y

# create some directories
# logs dir
exp_name = args.exp_name
logsdir = os.path.join('logs', exp_name)
print("[Output] Results saving to: ", logsdir)
os.makedirs(logsdir, exist_ok=True)
if start_over:
    remove_everything_in(logsdir)
# folder for tests
testdir = 'tests'
testdir = os.path.join(logsdir, testdir)
os.makedirs(testdir, exist_ok=True)
# folder for ckpts
ckptdir = 'ckpts'
ckptdir = os.path.join(logsdir, ckptdir)
os.makedirs(ckptdir, exist_ok=True)
# folder for pre_trained
pretraindir = 'pretrained'
pretraindir = os.path.join(pretraindir, exp_name)
os.makedirs(pretraindir, exist_ok=True)
if run_pretrain: # if calling pretrain, then remove previous pretrain records
    remove_everything_in(pretraindir)
pre_ckptdir = os.path.join(pretraindir, 'ckpts') # ckpt for pretrain
os.makedirs(pre_ckptdir, exist_ok=True)
pre_testdir = os.path.join(pretraindir, 'tests') # test for pretrain
os.makedirs(pre_testdir, exist_ok=True)

# init or load networks
net_dict, start, grad_vars, optimizer, lr_scheduler = create_bundle(ckptdir, num_vorts, decay_step, decay_gamma, pretrain_dir = pre_ckptdir)
img_x = gen_grid(width, height, device) # grid coordinates

def eval_vel(vorts_size, vorts_w, vorts_pos, query_pos):
    return vort_to_vel(net_dict['model_len'], vorts_size, vorts_w, vorts_pos, query_pos, length_scale = args.vort_scale)

def dist_2_len_(dist):
    return net_dict['model_len'](dist)

def size_pred():
    pred = net_dict['size_pred']
    size =  0.03 + torch.sigmoid(pred)
    return size

def w_pred():    
    pred = net_dict['w_pred']
    w = torch.sin(pred)
    return w

def comp_velocity(timestamps):
    jac = vmap(jacrev((net_dict['model_pos'])))(timestamps)
    post = jac[:, :, 0:1].view((timestamps.shape[0],-1,2,1))
    xt = post[:, :, 0, :]
    yt = post[:, :, 1, :]
    uv = torch.cat((xt, yt), dim = 2)
    return uv

# pretrain (of the trajectory module)
# the scale parameter influences to the initial positions of the vortices
def pretrain(scale = 1.):
    if start > 0:
        print("[Pretrain] Pretraining needs to be the start of the training pipeline. Please re-run with --start_over set to True.")
        sys.exit()

    with torch.no_grad():
        init_poss = gen_grid(vorts_num_x, vorts_num_y, device).view([-1, 2])
        init_poss = scale * init_poss + 0.5 * (1.-scale) # scale the initial grid
        pos_GT = init_poss[None, ...].expand(num_frames, -1, -1)
        vel_GT = torch.zeros_like(pos_GT)
    
    for it in range(10000):
        pos_pred = net_dict['model_pos'](timestamps).view(-1, num_vorts, 2)
        pos_loss = L2_Loss(pos_pred, pos_GT) 

        vel_pred = comp_velocity(timestamps)
        vel_loss = 0.001 * L2_Loss(vel_pred, vel_GT)

        loss = pos_loss + vel_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 200 == 0:
            print("[Pretrain] Iter: ", it, ", loss: ", loss.detach().cpu().numpy(), "/ pos loss: ", pos_loss.detach().cpu().numpy(), "/ vel loss: ", vel_loss.detach().cpu().numpy())
    
    # save pretrained results (trajectory module only)
    path = os.path.join(pre_ckptdir, 'pretrained.tar')
    torch.save({
        'model_pos_state_dict': net_dict['model_pos'].state_dict(),
    }, path)
    print('[Pretrain] Saved checkpoint to: ', path)
    with torch.no_grad():
        # output all vort positions with velocity
        values = net_dict["model_pos"](timestamps)
        values = values.view([values.shape[0], -1, 2])
        uvs = comp_velocity(timestamps)
        for i in range(values.shape[0]):
            print("[Pretrain] Writing test frame: ", i)
            vorts_pos_numpy = values[i].detach().cpu().numpy()
            vel_numpy = uvs[i].detach().cpu().numpy()
            write_vorts(vorts_pos_numpy, vel_numpy, pre_testdir, i)

    print('[Pretrain] Complete.')


# test learned simulation
def test(curr_it):
    print ("[Test] Testing at iter: " + str(curr_it))
    currdir = os.path.join(testdir, str(curr_it))
    os.makedirs(currdir, exist_ok=True)

    with torch.no_grad():
        total_imgs = [imgs[[0]]]
        total_vels = [None]
        total_vorts = [None]
        for i in range(num_available_frames):
            num_to_sim = 1
            if i == num_available_frames-1:
                num_to_sim += num_sims + max(num_unseen_frames, int(1.5 * num_frames)) -1 # if at the last reveal image, simulate to the end of the video
            pos_pred = net_dict['model_pos'](timestamps[[i]]).view((1,num_vorts,2))
            sim_imgs, sim_vorts_poss, sim_vels, sim_vorts_vels = simulate(total_imgs[-1].clone(), img_x, pos_pred.clone(), \
                                                w_pred().clone(), size_pred().clone(),\
                                                num_to_sim, vel_func = eval_vel, \
                                                boundary = boundary)
            total_imgs = total_imgs + sim_imgs
            total_vels = total_vels + sim_vels
            total_vorts = total_vorts + sim_vorts_poss
       
        visdir = os.path.join(currdir, 'particles')
        os.makedirs(visdir, exist_ok=True)        
        imgdir = os.path.join(currdir, 'imgs')
        os.makedirs(imgdir, exist_ok=True)
        vortdir = os.path.join(currdir, 'vorts')
        os.makedirs(vortdir, exist_ok=True)
        write_image(total_imgs[0][0].cpu().numpy(), imgdir, 0) # write init image
        for i in range(1, len(total_imgs)):
            print("[Test] Writing test frame: ", i)
            img = total_imgs[i].squeeze()
            vorts_pos = total_vorts[i]
            vorts_w = w_pred()[None,...]
            vorts_size = size_pred()[None,...]
            img_vel = total_vels[i].squeeze()
            vort_img = calc_vort(img_vel, boundary)
            vort_img_numpy = vort_img.detach().cpu().numpy()
            img_numpy = img.detach().cpu().numpy()
            vorts_pos_numpy = vorts_pos.detach().cpu().numpy()
            vorts_w_numpy = vorts_w.detach().cpu().numpy()
            write_visualization(img_numpy, vorts_pos_numpy, vorts_w_numpy, visdir, i, boundary = boundary)
            write_image(img_numpy, imgdir, i)
            write_vorticity(vort_img_numpy, vortdir, i)

# # # # #

# if pretrain is True then run pretrain() and quit
if run_pretrain:
    pretrain(args.init_vort_dist)
    sys.exit()

# if test_only is True then run test() and quit
if test_only:
    test(start)
    sys.exit()

# below is training code
prev_time = time.time()
for it in range(start, num_iters):
    # each iter select some different starting frames
    init_frames = probs.multinomial(num_samples = batch_size, replacement = False)

    # compute velocity prescribed by dynamics module
    with torch.no_grad():
        pos_pred_gradless = net_dict['model_pos'](timestamps[init_frames]).view((-1,num_vorts,2))
        D_vel = eval_vel(size_pred(), w_pred(), pos_pred_gradless, pos_pred_gradless)
        if boundary is not None:
            D_vel = boundary_treatment(pos_pred_gradless, D_vel, boundary, mode = 1)

    # velocity loss
    T_vel = comp_velocity(timestamps[init_frames]) # velocity prescribed by trajectory module
    vel_loss = 0.001 * L2_Loss(T_vel, D_vel)
    
    pos_pred = net_dict['model_pos'](timestamps[init_frames]).view((batch_size,num_vorts,2))
    sim_imgs, sim_vorts_poss, sim_img_vels, sim_vorts_vels = simulate(imgs[init_frames].clone(), img_x, pos_pred, w_pred(), \
                            size_pred(), num_sims, vel_func = eval_vel, boundary = boundary)
    sim_imgs = torch.stack(sim_imgs)

    # comp img loss
    img_losses = []
    if boundary is None: # if no boundary then compute loss on entire images
        for i in range(batch_size):
            pred = rgb_to_yuv(sim_imgs[:, i])
            GT = rgb_to_yuv(imgs[init_frames[i]+1: init_frames[i]+1+num_sims])
            img_losses.append(L2_Loss(pred, GT))
    else: # if has boundary then compute loss only on the valid regions
        OUT = (boundary[0] >= -boundary[2])
        IN = ~OUT
        for i in range(batch_size):
            pred = rgb_to_yuv(sim_imgs[:, i])[:, IN]
            GT = rgb_to_yuv(imgs[init_frames[i]+1: init_frames[i]+1+num_sims])[:, IN]
            img_losses.append(L2_Loss(pred, GT))
    img_loss = torch.stack(img_losses).sum()

    # loss is the sum of the two losses
    loss = img_loss + vel_loss

    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    if it % print_every == 0:
        print("[Train] Iter: ", it, ", loss: ", loss.detach().cpu().numpy(), "/ img loss: ", img_loss.detach().cpu().numpy(), "/ vel loss: ", vel_loss.detach().cpu().numpy())
        curr_time = time.time()
        print("[Train] Time Cost: ", curr_time-prev_time)
        prev_time = curr_time

    next_it = it + 1
    # save ckpt
    if (next_it % save_ckpt_every == 0 and next_it > 0) or (next_it == num_iters):
        path = os.path.join(ckptdir, '{:06d}.tar'.format(next_it))
        torch.save({
            'global_step': next_it,
            'w_pred': net_dict['w_pred'],
            'size_pred': net_dict['size_pred'],
            'model_pos_state_dict': net_dict['model_pos'].state_dict(),
            'model_len_state_dict': net_dict['model_len'].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print('[Train] Saved checkpoints at', path)

    if (next_it % test_every == 0 and next_it > 0) or (next_it == num_iters):
        test(next_it)

print('[Train] Complete.')
