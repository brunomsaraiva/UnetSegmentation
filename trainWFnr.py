from unet_trainer import UnetSegmentationModel

model = UnetSegmentationModel()
model.run_network("X_wf_nr.p", "y_wf_masks.p", "model_wf_nr_1.h5",
                  val_split=0.2, rotation_angle_step=30, gaussian_sigma_step=0.2,
                  n_epochs=500, n_batch_size=10)

