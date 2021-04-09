"""
GQN
"""

from representation import PyramidCls, TowerCls, PoolCls
from generation import GenerationCls

class GQNCls(nn.Module):

    def __init__(self, repr_architecture='Pool'):
        """
        Entire GQN architecture formed by combining
        representation network and generation network

        Args:
        - repr_architecture: String. Can be 'Pyramid', 'Tower', or 'Pool'. Determine the architecture of representation to be used
        """

        super(GQNCls, self).__init__()

        # initialize representation network
        architectures = {'Pyramid': PyramidCls, 'Tower': TowerCls, 'Pool', PoolCls}

        if repr_architecture not in architectures.keys():
            raise ValueError('[!] Representation network can be \'Pyramid\', \'Tower\', or \'Pool\'.')

        self.repr_net = architectures[repr_architecture]()
        
        # initialize generation network
        self.gen_net = GenerationCls()


    def forward(self, x, y, q):
        """
        Forward propagation

        Args:
        - x: A Tensor of shape (B, W, H, C). Batch of images
        - y: A Tensor of shape (B, 1, 1, 7). Batch of camera extrinsics
        - q: A Tensor of shape (?, 1, 1, 7). (Batch of) query viewpoints

        Returns: A Tensor of shape (?, W, H, C). (Batch of) predicted images at query viewpoints
        """
        return self.gen_net(self.repr_net(x, y), q)