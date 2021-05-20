"""
Training routine for GQN on 'rooms ring camera' dataset.
"""

from random import sample
from models.gqn import GQNCls

import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models.scheduler import AnnealingStepLR
from utils.data_loader import RoomsRingCameraDataset, sample_from_batch
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Parse arguments
parser = argparse.ArgumentParser()

# data loader parameters
parser.add_argument('--batch_size', type=int, default=36, help='input batch size')
parser.add_argument('--n_iter', type=int, default=1000000, help='number of iterations')
parser.add_argument('--n_workers', type=int, default=20, help='number of data loading workers')

# model parameters
parser.add_argument('--level', type=int, default=12, help='Number of generation/inference core levels')
parser.add_argument('--shared_core', type=bool, default=False, help='Use shared generation/inference core')

# optimizer & scheduler parameters
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta 1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta 2')
parser.add_argument('--step_size', type=int, default=20, help='step size')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

# pixel-wise variance
parser.add_argument('--sigma_i', type=float, default=2.0, help='Pixel standard deviation initial value')
parser.add_argument('--sigma_f', type=float, default=0.7, help='Pixel standard deviation final value')
parser.add_argument('--sigma_n', type=int, default=200000, help='Pixel standard deviation step size')

# I/O parameters
parser.add_argument('--model', type=str, default='', help='Path to the pretrained parameters')
parser.add_argument('--gen_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--out_dir', type=str, default='outputs',
                    help='output directory')

args = parser.parse_args()


def train_one_epoch(train_dataset, train_dataloader,
                    test_dataset, test_dataloader,
                    model, optimizer, scheduler, epoch=None, writer=None):
    """
    Train the model in one epoch
    Generate images in every 1000 iteration

    Args:
    - dataset: Pytorch Dataset object.
    - dataloader: Pytorch DataLoader object.
    - model: Pytorch model object.
    - optimizer: Pytorch optimizer object.
    - scheduler: Pytorch scheduler object.
    - epoch: Int. Index of current epoch
    - writer: SummaryWriter object.
    """

    # initialize pixel-variance
    sigma_t = args.sigma_i

    total_elbo = 0

    n_data = len(train_dataset)

    # Create a progress bar
    pbar = tqdm(total=n_data, leave=False)

    for i, (f_batch, c_batch) in enumerate(train_dataloader):

        f_batch = f_batch.to(device)
        c_batch = c_batch.to(device)

        # sample from batch
        x, v, x_q, v_q = sample_from_batch(f_batch, c_batch, dataset='Room', num_observations=5)

        # initialize gradients
        optimizer.zero_grad()

        # compute ELBO & increment total elbo
        elbo = model(x, v, x_q, v_q, sigma_t)

        # back propagation
        elbo.backward()
        optimizer.step()

        # update scheduler
        scheduler.step()

        # update pixel-variance annealing
        sigma_t = max(args.sigma_f + (args.sigma_i - args.sigma_f) * (1 - i / args.sigma_n), args.sigma_f)

        batch_size = x.shape[0]
        total_elbo += elbo

        pbar.update(batch_size)

        # generate images
        if (i + 1) % 1000 == 0:
            with torch.no_grad():
                x_q, pred = generate_images(test_dataloader, model, sigma_t)

                if writer:
                    writer.add_images('GT', x_q, int(i / 1000))
                    writer.add_images('Prediction', pred, int(i / 1000))

        # save the model
        if (i + 1) % 10000 == 0:
            model_file = os.path.join(
                args.out_dir, 'model_{:d}-{:d}.pth'.format(i + 1, epoch))
            torch.save(model.state_dict(), model_file)
            print("Saved '{}'.".format(model_file))

        # write summary
        if writer:
            writer.add_scalar('ELBO', elbo, i)

    pbar.close()
    n_batch = n_data / args.batch_size
    mean_elbo = total_elbo / n_batch
    return mean_elbo


def generate_images(test_dataloader, model):
    f_batch, c_batch = next(iter(test_dataloader))

    f_batch = f_batch.to(device)
    c_batch = c_batch.to(device)

    # sample query images/viewpoints from batch
    x, v, x_q, v_q = sample_from_batch(f_batch, c_batch, dataset='Room', num_observations=5)
    pred = model.generate(x, v, v_q)    # (B, 3, 64, 64)

    return x_q, pred


def main():
    # print parsed arguments
    print(args)

    # load datasets
    train_dataset = RoomsRingCameraDataset('./data/rooms_ring_camera_torch/train')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=int(args.n_workers))
    train_iter = iter(train_loader)

    test_dataset = RoomsRingCameraDataset('./data/rooms_ring_camera_torch/test')
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=0)
    test_iter = iter(test_loader)

    # prepare data for testing
    f_test_batch, c_test_batch = next(test_iter)
    f_test_batch = f_test_batch.to(device)
    c_test_batch = c_test_batch.to(device)
    x_test, v_test, x_q_test, v_q_test = sample_from_batch(f_test_batch, c_test_batch)
    
    # construct model
    model = GQNCls(repr_architecture='Tower', L=args.level, shared_core=args.shared_core)
    if torch.cuda.is_available():
        model.cuda()
    
    # ...or load an existing model
    if args.model != '':
        model.load_state_dict(torch.load(args.model))

    # configure optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=1e-08)

    # configure scheduler
    scheduler = AnnealingStepLR(optimizer)

    # initialize pixel variance
    sigma = args.sigma_i

    # create the output directory
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    writer = SummaryWriter(args.out_dir)

    for s in tqdm(range(args.n_iter)):
        try:
            f_batch, c_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            f_batch, c_batch = next(train_iter)

        f_batch = f_batch.to(device)
        c_batch = c_batch.to(device)
        
        x, v, x_q, v_q = sample_from_batch(f_batch, c_batch)

        optimizer.zero_grad()

        elbo = model(x, v, x_q, v_q, sigma)

        # back propagation
        elbo.backward()
        optimizer.step()

        # update scheduler
        scheduler.step()

        # Pixel-variance annealing
        sigma = max(args.sigma_f + (args.sigma_i - args.sigma_f)*(1 - s/(2e5)), args.sigma_f)

        if writer:
            writer.add_scalar('Train loss', elbo, s)

        with torch.no_grad():

            # generate images and record
            if (s+1) % args.gen_interval == 0:
                pred = model.generate(x_test, v_test, v_q_test)

                if writer:
                    writer.add_images('GT', x_q_test)
                    writer.add_iamges('Prediction', pred)

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



    """
    for epoch in range(args.n_epochs):
        train_one_epoch(train_dataset, train_loader, test_dataset, test_loader, model, optimizer, scheduler, epoch, writer)

        # Save the model.
        model_file = os.path.join(
                args.out_dir, 'model_{:d}.pth'.format(epoch + 1))
        torch.save(model.state_dict(), model_file)
        print("Saved '{}'.".format(model_file))

    writer.close()
    """

if __name__ == '__main__':
    main()
