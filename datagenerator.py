import numpy as np
from skimage.transform import rotate
from skimage.filters import gaussian

class DataGenerator(object):

    def __init__(self,):
        self.train_X = None
        self.train_y = None
        self.val_X = None
        self.val_y = None

    def generate_data(self, X, y, val_split=0.2, rotation_angle_step=30, gaussian_sigma_step=0.2):
        self.train_X = X[0:int(X.shape[0]*(1-val_split)), :, :, :]
        self.train_y = y[0:int(y.shape[0]*(1-val_split)), :, :, :]

        self.val_X = X[int(X.shape[0]*(1-val_split)):, :, :, :]
        self.val_y = y[int(y.shape[0]*(1-val_split)):, :, :, :]

        self.train_X, self.train_y = self.create_rotations(self.train_X, self.train_y, rotation_angle_step)
        self.val_X, self.val_y = self.create_rotations(self.val_X, self.val_y, rotation_angle_step)

        self.train_X, self.train_y = self.create_blur(self.train_X, self.train_y, gaussian_sigma_step)
        self.val_X, self.val_y = self.create_blur(self.val_X, self.val_y, gaussian_sigma_step)

    def create_rotations(self, X, y, angle_step):
        rotated_X = []
        rotated_y = []

        for i in range(X.shape[0]):
            for angle in range(0, 360, angle_step):
                rotated_X.append(rotate(X[i, :, :, 0], angle))
                rotated_y.append(rotate(y[i, :, :, 0], angle))

        rotated_X = np.array(rotated_X).reshape(-1, 256, 256, 1)
        rotated_y = np.array(rotated_y).reshape(-1, 256, 256, 1)

        return rotated_X, rotated_y

    def create_blur(self, X, y, sigma_step):
        blurred_X = []
        blurred_y = []

        for i in range(X.shape[0]):
            for sigma in np.arange(0, 2, sigma_step):
                blurred_X.append(rotate(X[i, :, :, 0], sigma))
                blurred_y.append(rotate(y[i, :, :, 0], sigma))

        blurred_X = np.array(blurred_X).reshape(-1, 256, 256, 1)
        blurred_y = np.array(blurred_y).reshape(-1, 256, 256, 1)

        return blurred_X, blurred_y

