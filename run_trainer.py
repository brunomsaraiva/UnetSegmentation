from unet_trainer import UnetSegmentationModel

model = UnetSegmentationModel()
model.run_network("X_wf_bf.p", "y_wf_masks.p", "model_wf_bf_1.h5")

