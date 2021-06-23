import os
import pickle
import numpy as np
from skimage.io import imread
from skimage.color import rgb2grey
from skimage.exposure import rescale_intensity
from skimage.util import img_as_float

path = "data"

X = []
y = []

for imgname in os.listdir(path + os.sep + "wf_bf"):
    X_image = img_as_float(rescale_intensity(imread(path + os.sep + "wf_bf" + os.sep + imgname)))
    X.append(X_image)
    #maskname = "mask_" + imgname.split("_")[1] + "_" + imgname.split("_")[2]
    #y_image = img_as_float(rgb2grey(imread(path + os.sep + "sim_masks" + os.sep + maskname)))
    #y_image = img_as_float(rgb2grey(imread(path + os.sep + "wf_masks" + os.sep + imgname)))
    #y.append(y_image)

X = np.array(X)
#y = np.array(y)

pickle.dump(X, open("X_wf_bf.p", "wb"))
#pickle.dump(y, open("y_wf_masks.p", "wb"))

