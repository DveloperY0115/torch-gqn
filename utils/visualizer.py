"""
Set of classes & functions for data visualization
"""

import numpy as np
import matplotlib.pyplot as plt
# lt.rcParams['figure.figsize'] = (10, 8)

class Visualizer:

    def __init__(self):
        pass

    def show_img_grid(self, imgs, row, col):
        """
        Visualize multiple images in grid

        Args:
        - imgs: A set of images (types can vary)
        - row: Int. Number of images in one row
        """
        fig, axs = plt.subplots(nrows=row, ncols=col)

        for idx, data in enumerate(imgs):
            axs.ravel()[idx].imshow(data)
            axs.ravel()[idx].set_title('Image # {}'.format(idx))
            axs.ravel()[idx].set_axis_off()
        plt.tight_layout()
        plt.show()

    def show_img(self, img):
        """
        Visualize the given image

        Args:
        - img: An image object (types can vary)
        """

        # if input is Numpy array
        if isinstance(img, np.ndarray):
            self._show_numpy(img)

    def _show_numpy(self, array):
        imgplot = plt.imshow(array)
        plt.show()
