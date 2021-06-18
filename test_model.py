import keras
import numpy as np
from skimage.io import imread
from unet_classifier import UnetSegmentationClassifier

classifier = UnetSegmentationClassifier(model_path="model_sim_nr_1.h5")
mask = classifier.create_mask(imread("data\\sim_nr\\nr_1_1.tif"))
print(mask.shape)

