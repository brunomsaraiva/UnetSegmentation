import keras
import numpy as np
from skimage.io import imread
from unet_classifier import UnetSegmentationClassifier
from skimage.util import img_as_float
from matplotlib import pyplot as plt

classifier = UnetSegmentationClassifier(model_path="model_wf_nr_1.h5")
mask = classifier.create_mask(imread("test_data\\OMX_nr.tif"))
mask = img_as_float(mask)

plt.imshow(mask, cmap="gray")
plt.show()