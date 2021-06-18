import os
import numpy as np
from skimage.io import imread, imsave
from skimage.exposure import rescale_intensity

fluor = rescale_intensity(imread("elyra_membrane.tif"))
mask = rescale_intensity(imread("elyra_mask.tif"))

for i in range(int(fluor.shape[0]/256)):
    for ii in range(int(fluor.shape[1]/256)):
        imsave("data\\sim_nr" + os.sep + "nr_" + str(i) + "_" + str(ii) + ".tif", fluor[i*256:(i+1)*256, ii*256:(ii+1)*256])
        imsave("data\\sim_masks" + os.sep + "mask_" + str(i) + "_" + str(ii) + ".tif", mask[i*256:(i+1)*256, ii*256:(ii+1)*256])

botright_fluor = np.zeros((256, 256))
botright_mask = np.zeros((256, 256))

botright_fluor[256-fluor.shape[0]+int(fluor.shape[0]/256)*256:, 256-fluor.shape[1]+int(fluor.shape[1]/256)*256:] = fluor[int(fluor.shape[0]/256)*256:, int(fluor.shape[1]/256)*256:]
botright_mask[256-mask.shape[0]+int(mask.shape[0]/256)*256:, 256-mask.shape[1]+int(mask.shape[1]/256)*256:] = mask[int(fluor.shape[0]/256)*256:, int(mask.shape[1]/256)*256:]
imsave("data\\sim_nr" + os.sep + "nl_br.tif", botright_fluor)
imsave("data\\sim_masks" + os.sep + "mask_br.tif", botright_mask)

for i in range(int(fluor.shape[0]/256)):
    tmp_fluor = np.zeros((256, 256))
    tmp_mask = np.zeros((256, 256))
    tmp_fluor[:, 256-fluor.shape[0]+int(fluor.shape[0]/256)*256:] = fluor[i*256:(i+1)*256, int(fluor.shape[0]/256)*256:]
    tmp_mask[:, 256-mask.shape[0]+int(mask.shape[0]/256)*256:] = mask[i*256:(i+1)*256, int(mask.shape[0]/256)*256:]
    imsave("data\\sim_nr" + os.sep + "nr_r_" + str(i) + ".tif", tmp_fluor)
    imsave("data\\sim_masks" + os.sep + "mask_r_" + str(i) + ".tif", tmp_mask)

for i in range(int(fluor.shape[1]/256)):
    tmp_fluor = np.zeros((256, 256))
    tmp_mask = np.zeros((256, 256))
    tmp_fluor[256-fluor.shape[1]+int(fluor.shape[1]/256)*256:, :] = fluor[int(fluor.shape[0]/256)*256:, i*256:(i+1)*256]
    tmp_mask[256-mask.shape[1]+int(mask.shape[1]/256)*256:, :] = mask[int(mask.shape[0]/256)*256:, i*256:(i+1)*256]
    imsave("data\\sim_nr" + os.sep + "nr_b_" + str(i) + ".tif", tmp_fluor)
    imsave("data\\sim_masks" + os.sep + "mask_b_" + str(i) + ".tif", tmp_mask)

