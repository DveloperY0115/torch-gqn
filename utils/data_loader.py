import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class RoomsRingCameraDataset(Dataset):
    """
    Pytorch dataset object GQN dataset
    """

    def __init__(self, root, transform=None):
        """
        Constructor.

        Args:
        - root: String. Root directory of training/test datasets
        - transform: Object. Usually function to be applied on data
        """

        self.root = root
        self.transform = transform


    def __len__(self):
        """
        Get the number of data in the dataset
        """

        return len(os.listdir(self.root))
        # return 1000000


    def __getitem__(self, idx):
        """
        Get specific data from the dataset

        Args:
        - idx: Int. The index of the data to be retrieved
        """

        scene_file = os.path.join(self.root, f"{idx}.p")
        
        with open(scene_file, 'rb') as file:
            context = pickle.load(file)

        frames = context[0]    # 0th element holds frame data
        cameras = context[1]    # 1st element holds camera extrinsics

        frames = torch.from_numpy(frames)
        cameras = torch.from_numpy(cameras)

        pos, orientation = torch.split(cameras, 3, dim=-1)
        yaw, pitch = torch.split(orientation, 1, dim=-1)

        cameras = [pos, torch.cos(yaw), torch.sin(yaw), torch.cos(pitch), torch.sin(pitch)]
        cameras = torch.cat(cameras, dim=-1)

        return frames, cameras


def sample_from_batch(frame_batch, camera_batch, dataset='Room', num_observations=None, seed=None):
    """
    Sample random number of views from each scenes in a batch

    Args:
    - frame_batch: A Tensor of shape (B, S, W, H, C)
    - camera_batch: A Tensor of shape (B, S, 7)
    - dataset: String. Indicates one of GQN datasets
    - num_observations: Int. Number of views to be sampled
    - seed: Int. Seed for random sampling

    Returns:
    - x: A Tensor of shape (B, S', C, H, W). Batch of sequences of images
    - v: A Tensor of shape (B, S', 7). Batch of sequences of viewpoint information associated with each of x
    - x_q: A Tensor of shape (B, C, H, W). Batch of target images
    - v_q: A Tensor of shape (B, 7). Batch of query viewpoints
    """

    random.seed(seed)

    if dataset == 'Room':
        num_frames = 5
    
    if not num_observations:
        num_observations = random.randint(1, num_frames)

    len_sequence = frame_batch.size(1)

    context_idx = random.sample(range(len_sequence), num_observations)
    query_idx = random.randint(0, len_sequence-1)

    x, v = frame_batch[:, context_idx, :, :, :], camera_batch[:, context_idx, :]
    x_q, v_q = frame_batch[:, query_idx, :, :, :], camera_batch[:, query_idx, :]

    # unsqueeze viewpoint tensors to make it of shape (B, 7, 1, 1)
    v = v.unsqueeze(3)
    v = v.unsqueeze(4)
    v_q = v_q.unsqueeze(2)
    v_q = v_q.unsqueeze(3)

    # (B, M, W, H, C) -> (B, M, C, H, W)
    x = x.transpose(2, 4)
    x_q = x_q.transpose(1, 3)

    return x, v, x_q, v_q