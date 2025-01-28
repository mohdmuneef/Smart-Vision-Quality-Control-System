import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Optional: Disable oneDNN optimizations (if desired)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class FreshnessDetectionModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        # Use VGG16 as the base model, exclude the top layers
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        # Freeze layers except the last 4
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        # Add custom layers on top of the base model
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        # Define the model
        model = Model(inputs=base_model.input, outputs=output)
        # Compile the model with Adam optimizer and binary crossentropy loss
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_data_dir, validation_data_dir, epochs=10, batch_size=32):
        # Check and create validation directory if not exists
        if not os.path.exists(validation_data_dir):
            os.makedirs(validation_data_dir)
            print(f"Created directory: {validation_data_dir}")

        # Image data generators for real-time data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # Rescale pixel values
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        validation_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale validation images

        # Load training images from the directory
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary'
        )

        # Load validation images from the directory
        validation_generator = validation_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary'
        )

        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=[early_stopping]  # Add the early stopping callback
        )

        return history

    def predict(self, image):
        # Ensure the image is in the correct format (224x224x3)
        if image.shape != (224, 224, 3):
            image = tf.image.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize the image

        # Predict freshness score (0 to 1)
        prediction = self.model.predict(image)
        return prediction[0][0]  # Return the freshness score

    def save_model(self, filepath):
        if not filepath.endswith('.keras'):
            filepath += '.keras'  # Ensure the file ends with .keras
        self.model.save(filepath)

    def load_model(self, filepath):
        if not filepath.endswith('.keras'):
            filepath += '.keras'  # Ensure the file ends with .keras
        self.model = tf.keras.models.load_model(filepath)

if __name__ == "__main__":
    # Example usage
    model = FreshnessDetectionModel()

    # Train the model (you would need to have your dataset ready)
    train_data_dir = 'datasets/train'
    validation_data_dir = 'datasets/test'
    history = model.train(train_data_dir, validation_data_dir)

    # Save the model
    model.save_model('freshness_detection_model')

    # Load the model
    # model.load_model('freshness_detection_model')

    # Make a prediction (you would need to load an image here)
    # image = load_and_preprocess_image('path/to/image.jpg')
    # freshness_score = model.predict(image)
    # print(f"Freshness score: {freshness_score}")
