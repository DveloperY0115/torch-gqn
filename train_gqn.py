"""
Training routine for GQN on 'rooms ring camera' dataset.
"""

from random import sample
from models.gqn import GQNCls

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.scheduler import AnnealingStepLR
from utils.data_loader import RoomsRingCameraDataset, sample_from_batch
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Parse arguments
parser = argparse.ArgumentParser()

# data loader parameters
parser.add_argument('--batch_size', type=int, default=36, help='input batch size')

parser.add_argument('--n_workers', type=int, default=0, help='number of data loading workers')

# training parameters
parser.add_argument('--max_step', type=int, default=1e6, help='maximum number of training steps')


# model parameters
parser.add_argument('--level', type=int, default=8, help='number of generation/inference core levels')
parser.add_argument('--shared_core', type=bool, default=False, help='Use shared generation/inference core')

# optimizer & scheduler parameters
parser.add_argument('--beta1', type=float, default=0.9, help='beta 1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta 2')
parser.add_argument('--eps', type=float, default=1e-8, help='epsilon')
parser.add_argument('--mu_i', type=float, default=5e-4, help='initial learning rate')
parser.add_argument('--mu_f', type=float, default=5e-5, help='final learning rate')

# pixel-wise variance
parser.add_argument('--sigma_i', type=float, default=2.0, help='Pixel standard deviation initial value')
parser.add_argument('--sigma_f', type=float, default=0.4, help='Pixel standard deviation final value')
parser.add_argument('--sigma_n', type=int, default=1e6, help='Pixel standard deviation step size')

# I/O parameters
parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint file')
parser.add_argument('--out_dir', type=str, default='outputs',
                    help='output directory')
parser.add_argument('--gen_interval', type=int, default=100, help='Period for generation core testing')
parser.add_argument('--save_interval', type=int, default=10000, help='Period for making checkpoint')

args = parser.parse_args()

def main():
    # print parsed arguments
    print(args)

    # load datasets
    train_dataset = RoomsRingCameraDataset('./data/rooms_ring_camera_torch/train')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=int(args.n_workers),
                              pin_memory=True)

    train_iter = iter(train_loader)

    test_dataset = RoomsRingCameraDataset('./data/rooms_ring_camera_torch/test')
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True)
    test_iter = iter(test_loader)
    
    # construct model
    model = GQNCls(repr_architecture='Tower', L=args.level, shared_core=args.shared_core)
    if torch.cuda.is_available():
        model.cuda()

    # configure optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.mu_i,
        betas=(args.beta1, args.beta2),
        eps=args.eps)

    # configure scheduler
    scheduler = AnnealingStepLR(optimizer, args.mu_i, args.mu_f)

    # training step
    s_begin = 0

    # ...or load an existing checkpoint
    if args.checkpoint != '':
        print('Loading checkpoint at: {}'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        s_begin = checkpoint['step']

        model.train()

    # create the output directory
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # Tensorboard
    writer = SummaryWriter(args.out_dir)

    sigma_i = args.sigma_i
    sigma_f = args.sigma_f

    sigma_t = sigma_i

    # training routine
    for s in tqdm(range(s_begin, int(args.max_step))):
        try:
            f_batch, c_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            f_batch, c_batch = next(train_iter)

        f_batch = f_batch.to(device)
        c_batch = c_batch.to(device)
        x, v, x_q, v_q = sample_from_batch(f_batch, c_batch)

        # initialize gradient
        optimizer.zero_grad()

        # forward
        elbo, kl_div, likelihood = model(x, v, x_q, v_q, sigma_t)

        # back propagation
        (-elbo.mean()).backward()

        # update optimizer, scheduler
        optimizer.step()
        scheduler.step()

        # Pixel-variance annealing
        sigma_t = max(args.sigma_f + (args.sigma_i - args.sigma_f)*(1 - s/(args.sigma_n)), args.sigma_f)

        # write summary
        if writer:
            writer.add_scalars('Train statistics', {
                'ELBO (avg)': -elbo.mean(),
                'KL Divergence (avg)': kl_div.mean(),
                'Likelihood (avg)': likelihood.mean(),
                'sigma (avg)': sigma_t
            }, s)

        with torch.no_grad():
            try:
                test_f_batch, test_c_batch = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                test_f_batch, test_c_batch = next(test_iter)
                
            test_f_batch = test_f_batch.to(device)
            test_c_batch = test_c_batch.to(device)
            x_test, v_test, x_q_test, v_q_test = sample_from_batch(test_f_batch, test_c_batch)
          
            # generate images and record
            if (s+1) % args.gen_interval == 0:
                pred = model.generate(x_test, v_test, v_q_test, sigma_t)

                if writer:
                    writer.add_images('GT', x_q_test, s)
                    writer.add_images('Prediction', pred, s)

            # add checkpoint
            if (s+1) % args.save_interval == 0:
                filename = os.path.join(args.out_dir, '{}.tar'.format(s+1))
                torch.save({
                    'step': s,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, filename)

                print('Saved {}'.format(filename))
    
    writer.close()


if __name__ == '__main__':
    main()
