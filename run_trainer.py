from unet_trainer import UnetSegmentationModel

model = UnetSegmentationModel()
model.run_network("X_sim_nr.p", "y_sim_masks.p", "model_sim_nr_1.h5")

