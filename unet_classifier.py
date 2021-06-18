import keras
import numpy as np
from skimage.io import imread
from skimage.util import img_as_float
from skimage.exposure import rescale_intensity
from keras.models import load_model
from tkinter import filedialog as fd

class UnetSegmentationClassifier(object):

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = fd.askopenfilename()
        self.model = load_model(model_path)

    def create_mask(self, img):
        img = img_as_float(rescale_intensity(np.array([img.reshape(256, 256, 1)])))
        print(img.shape)

        mask = self.model.predict(img)[0]
        mask = keras.preprocessing.image.array_to_img(mask)
        print(mask.shape)

        return mask[0]

