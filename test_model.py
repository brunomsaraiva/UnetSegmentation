import keras
import numpy as np
from skimage.io import imread
from unet_classifier import UnetSegmentationClassifier
from matplotlib import pyplot as plt

classifier = UnetSegmentationClassifier(model_path="model_sim_nr_1.h5")
mask = classifier.create_mask(imread("test_data\\nr_1_0.tif"))

plt.imshow(mask, cmap="gray")
plt.show()