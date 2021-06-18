import os
from skimage.io import imread, imsave
from skimage.util import img_as_float

data_path = "data\\wf_masks"

for nmask in os.listdir(data_path):
    mask = imread(data_path + os.sep + nmask)
    mask = img_as_float(mask > 0.0)
    imsave(data_path + os.sep + nmask, mask)

