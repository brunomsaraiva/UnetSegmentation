import keras
import pickle
import numpy as np
from keras import layers
from keras.callbacks import TensorBoard, EarlyStopping
from datagenerator import DataGenerator
from tkinter import filedialog as fd


class UnetSegmentationModel(object):

    def __init__(self):
        self.X = None
        self.y = None
        self.model = None
        self.max_width = 256
        self.max_height = 256
        self.datagenerator = DataGenerator()

    def load_data(self, X_path, y_path):
        if X_path is None:
            X_path = fd.askopenfilename()
        self.X = np.array(pickle.load(open(X_path, "rb"))).reshape(-1, 256, 256, 1)

        if y_path is None:
            y_path = fd.askopenfilename()
        self.y = np.array(pickle.load(open(y_path, "rb"))).reshape(-1, 256, 256, 1)

    def create_model(self):
        inputs = layers.Input((256, 256, 1))

        x = layers.Conv2D(32, (3, 3), strides=2, padding='same', input_shape=(256, 256, 1))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x

        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, (3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, (3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(2, 3, activation="softmax", padding="same")(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        model.summary()
        self.model = model

    def compile_model(self):
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam')

    def train_model(self, model_path,
                    val_split, rotation_angle_step, gaussian_sigma_step,
                    n_epochs, n_batch_size):

        tbCallBack = TensorBoard(log_dir="./Graph", histogram_freq=0,
                                 write_graph=True, write_images=True)
        checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True)
        earlystopper = EarlyStopping(patience=20, monitor="val_loss", verbose=1)

        self.datagenerator.generate_data(self.X, self.y,
                                         val_split=val_split,
                                         rotation_angle_step=rotation_angle_step,
                                         gaussian_sigma_step=gaussian_sigma_step)

        self.model.fit(self.datagenerator.train_X, self.datagenerator.train_y,
                       validation_data=(self.datagenerator.val_X, self.datagenerator.val_y),
                       epochs=n_epochs, batch_size=n_batch_size,
                       verbose=1,
                       callbacks=[tbCallBack, checkpoint, earlystopper])

    def run_network(self, x_path=None, y_path=None, model_path=None, val_split=0.2, rotation_angle_step=30, gaussian_sigma_step=0.2,
                    n_epochs=500, n_batch_size=10):
        self.load_data(x_path, y_path)
        self.create_model()
        self.compile_model()
        self.train_model(model_path,
                         val_split, rotation_angle_step, gaussian_sigma_step,
                         n_epochs, n_batch_size)

