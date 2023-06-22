import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import sys
from src.logger import logging
from src.exception import CustomException


class DataProcessing:
    # Method for creating train, test, and validation batches
    def train_test_val_batches(self, dataset, training_size, validation_size, shuffle_size):
        try:
            # Shuffling the dataset
            dataset = dataset.shuffle(shuffle_size, seed=12)

            # Creating batches for train, test, and validation
            train_dataset = dataset.take(int(len(dataset) * training_size))
            test_dataset = dataset.skip(int(len(dataset) * training_size))
            validation_dataset = test_dataset.take(int(len(test_dataset) * validation_size))

            # Cache, Shuffle, and Prefetch the Datasets
            train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            test_dataset = test_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            validation_dataset = validation_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

            return train_dataset, test_dataset, validation_dataset

        except Exception as e:
            raise CustomException(e, sys)

    def initialize_process_train(self, directory):
        try:
            # Loading the data into tensorflow dataset
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                directory, batch_size=32, image_size=(256, 256), shuffle=True
            ) # "../artifacts/data"

            logging.info("Loaded data into tensorflow datasets")

            # Getting train, test, and validation batches
            train_ds, test_ds, val_ds = self.train_test_val_batches(dataset, 0.8, 0.1, 10000)
            logging.info("Created train, test, and validation batches")

            # Rescaling and resizing layer
            resize_and_rescale = tf.keras.Sequential(
                [
                    layers.experimental.preprocessing.Resizing(256, 256),
                    layers.experimental.preprocessing.Rescaling(1.0 / 255),
                ]
            )

            # Data augmentation layer
            data_augmentation = tf.keras.Sequential(
                [
                    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                    layers.experimental.preprocessing.RandomRotation(0.2),
                ]
            )

            input_shape = (32, 256, 256, 3)
            n_classes = 2

            model = models.Sequential(
                [
                    # First we will resize and rescale our images
                    resize_and_rescale,
                    # We will create new data for making our model robust
                    data_augmentation,
                    # Let's now add convolutional and pooling layers
                    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation="relu"),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation="relu"),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation="relu"),
                    layers.MaxPooling2D((2, 2)),
                    # Flattening the feature map for feeding in vanilla artificial neural network
                    layers.Flatten(),
                    layers.Dense(64, activation="relu"),
                    layers.Dense(n_classes, activation="softmax"),
                ]
            )

            model.build(input_shape=input_shape)

            # Compiling the model
            model.compile(
                optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=["accuracy"],
            )
            logging.info("Built and compiled the model")

            model.fit(
                train_ds,
                batch_size=32,
                validation_data=val_ds,
                verbose=1,
                epochs=3,
            )

            logging.info("Trained the model")

            # Saving the model
            current_model_version = 1
            model.save(f"artifacts/Model_trained/{current_model_version}")
            logging.info("Saved the model")

            return 90

        except Exception as e:
            raise CustomException(e, sys)
