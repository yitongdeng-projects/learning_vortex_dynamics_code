# [ICLR 2023] Learning Vortex Dynamics for Fluid Inference and Prediction
by [Yitong Deng](https://yitongdeng.github.io/), [Hong-Xing Yu](https://kovenyu.com/), [Jiajun Wu](https://jiajunwu.com/), and [Bo Zhu](https://cs.dartmouth.edu/~bozhu/).

Our paper can be found at: https://arxiv.org/abs/2301.11494.

Video results can be found at: https://yitongdeng.github.io/vortex_learning_webpage.

## Environment
The environment can be installed by conda via:
```bash
conda env create -f environment.yml
conda activate vortex_env
```

Our code is tested on `Windows 10` and `Ubuntu 20.04`.

## Data
The 5 videos (2 synthetic and 3 real-world) used in our paper can be downloaded from [Google Drive](https://drive.google.com/file/d/1avJrZPOI9JEURTa2wWQQ9QWqxw5UTo5s/view?usp=sharing). Once downloaded, place the unzipped `data` folder to the project root directory.

## Synthetic 1

#### Pretrain

First, execute the command below to pretrain the trajectory network so that the initial vortices are regularly spaced to cover the simulation domain and remains stationary.

```bash
python train.py --config configs/synthetic_1.txt --run_pretrain True
```

Once completed, navigate to `pretrained/exp_synthetic_1/tests/` and check that the plotted dots are regularly spaced and remain roughly stationary. A file `pretrained.tar` shall also appear at `pretrained/exp_synthetic_1/ckpts/`.

#### Train

Then, run the command below to train.

```bash
python train.py --config configs/synthetic_1.txt
```

Checkpoints and testing results are written to `logs/exp_synthetic_1/tests/` once every 1000 training iterations.

#### Results

When run on our Windows machine with AMD Ryzen Threadripper 3990X and NVIDIA RTX A6000, this is the final testing result we get:

![synthetic_1](gifs/synthetic_1.gif)

Note that since our PyTorch code includes nondeterministic components (e.g., the CUDA grid sampler), it is expected that each training session will not generate the exact same outcome.

## Synthetic 2

#### Pretrain

```bash
python train.py --config configs/synthetic_2.txt --run_pretrain True
```

#### Train

```bash
python train.py --config configs/synthetic_2.txt
```

#### Results

![synthetic_2](gifs/synthetic_2.gif)

## Real 1

#### Pretrain

```bash
python train.py --config configs/real_1.txt --run_pretrain True
```

#### Train

```bash
python train.py --config configs/real_1.txt
```

#### Results

![real_1](gifs/real_1.gif)

## Real 2

#### Pretrain

```bash
python train.py --config configs/real_2.txt --run_pretrain True
```

#### Train

```bash
python train.py --config configs/real_2.txt
```

#### Results

![real_2](gifs/real_2.gif)

## Real 3

#### Pretrain

```bash
python train.py --config configs/real_3.txt --run_pretrain True
```

#### Train

```bash
python train.py --config configs/real_3.txt
```

#### Results

![real_3](gifs/real_3.gif)

## Trying your own video
We assume the input is a Numpy array of shape `[num_frames], 256, 256, 3`, with the last dimension representing RGB pixel values between 0.0 and 1.0, located in `data/[your_name_here]/imgs.npy`. For fluid videos with boundaries (like in our real-world examples), it is required that a Numpy array of shape `256, 256` representing the signed distance field to the boundary be supplied in `data/[your_name_here]/sdf.npy`. We assume the signed distance has a unit of pixels.

For videos of higher dynamical complexity, we also encourage playing around with the number of vortex particles used. Currently, this is determined by the `vorts_num_x` and `vorts_num_y` parameters in `train.py` hard coded to 4, which might need to be increased as needed.

## Bibliography
If you find our paper or code helpful, please consider citing:
```
@inproceedings{deng2023vortex,
 title={Learning Vortex Dynamics for Fluid Inference and Prediction},
 author={Yitong Deng and Hong-Xing Yu and Jiajun Wu and Bo Zhu},
 booktitle={Proceedings of the International Conference on Learning Representations},
 year={2023},
}
```
