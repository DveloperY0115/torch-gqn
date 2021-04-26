import torch

from torch.utils.data import DataLoader
from utils.loader import RoomsRingCameraDataset
from utils.visualizer import Visualizer

def main():
    # configure dataloader
    train_set = RoomsRingCameraDataset('./data/rooms_ring_camera_torch/train')
    train_loader = DataLoader(dataset=train_set,
                            batch_size=32,
                            shuffle=True,
                            num_workers=2)
    
    dataiter = iter(train_loader)
    f_batch, c_batch = dataiter.next()

    print(f_batch.shape)
    print(c_batch.shape)

    vis = Visualizer()

    vis.show_img_grid(f_batch[0], 2, 5)

if __name__ == '__main__':
    main()
