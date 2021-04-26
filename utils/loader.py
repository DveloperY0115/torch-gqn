import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class RoomsRingCameraDataset(Dataset):
    """
    Pytorch dataset object for 'rooms_ring_camera' dataset
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


def sample_from_batch(frame_batch, camera_batch, dataset='Room', num_observations=5, seed=None):
    random.seed(seed)

    if dataset == 'Room':
        num_frames = 5
    
    if not num_observations:
        num_observations = random.randint(1, num_frames)

    context_idx = random.sample(range(frame_batch.size(1)), num_observations)
    query_idx = random.randint(0, frame_batch.size(1)-1)

    x, v = frame_batch[:, context_idx], camera_batch[:, context_idx]
    x_q, v_q = frame_batch[:, query_idx], camera_batch[:, query_idx]

    return x, v, x_q, v_q