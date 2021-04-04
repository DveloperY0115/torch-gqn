"""
Set of classes & functions for data visualization
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

class Visualizer:

    def __init__(self):
        pass

    def show_img_grid(self, imgs, row):
        """
        Visualize multiple images in grid

        Args:
        - imgs: A set of images (types can vary)
        - row: Int. Number of images in one row 
        """
        col = len(imgs) // row
        fig, axs = plt.subplots(row, col)

        for i in range(row):
            for j in range(col):
                ax[row, col].imshow(imgs[i * row + col])

    def show_img(self, img):
        """
        Visualize the given image

        Args:
        - img: An image object (types can vary)
        """

        # if input is Numpy array
        if isinstance(img, np.ndarray):
            self._show_numpy(img)
        pass

    def _show_numpy(self, array):
        imgplot = plt.imshow(array)
        plt.show()
        pass