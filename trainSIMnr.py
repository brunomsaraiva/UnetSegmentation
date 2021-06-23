from unet_trainer import UnetSegmentationModel

model = UnetSegmentationModel()
model.run_network("X_sim_nr.p", "y_sim_masks.p", "model_sim_nr_2.h5",
                  val_split=0.2, rotation_angle_step=30, gaussian_sigma_step=0.2,
                  n_epochs=500, n_batch_size=10)

