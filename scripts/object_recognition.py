from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import os
import joblib
from tqdm import tqdm
import multiprocessing

class ObjectRecognitionModel:
    def __init__(self, num_classes=10):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.num_classes = num_classes
        self.class_mapping = {}

    def preprocess_image(self, image_path):
        """Preprocess the image by resizing and normalizing."""
        try:
            with Image.open(image_path) as img:
                img = img.resize((64, 64))  # Resize to save memory
                img_array = np.array(img.convert('RGB')) / 255.0  # Normalize pixel values
                return img_array.flatten()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def load_all_data(self, data_dir, batch_size=100):
        """Load all image data and corresponding labels in batches from the specified directory."""
        X, y = [], []
        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        for class_id, class_name in enumerate(class_dirs):
            self.class_mapping[class_id] = class_name
            class_dir = os.path.join(data_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Process images in batches
            for i in tqdm(range(0, len(image_files), batch_size), desc=f"Loading {class_name} images"):
                batch = image_files[i:i+batch_size]
                with multiprocessing.Pool() as pool:
                    processed_batch = pool.map(self.preprocess_image, [os.path.join(class_dir, f) for f in batch])
                
                X += [img for img in processed_batch if img is not None]
                y += [class_id] * len(processed_batch)

        return np.array(X), np.array(y)

    def train(self, train_data_dir, validation_split=0.2):
        """Train the model using the images in the specified training directory."""
        print("Loading all data in batches...")
        X, y = self.load_all_data(train_data_dir)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)

        print("Training model...")
        self.model.fit(X_train, y_train)

        # Evaluate the model
        train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        val_accuracy = accuracy_score(y_val, self.model.predict(X_val))

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        return {"train_accuracy": train_accuracy, "val_accuracy": val_accuracy}

    def predict(self, image_path):
        """Predict the class of a single image."""
        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            raise ValueError(f"Image at {image_path} could not be processed.")
        
        prediction = self.model.predict_proba([processed_image])[0]
        predicted_class_id = np.argmax(prediction)
        predicted_class_name = self.class_mapping.get(predicted_class_id, "Unknown")
        return prediction, predicted_class_name

    def predict_all_in_folder(self, test_data_dir):
        """Predict classes for all images in the specified test directory."""
        print(f"Looking for images in folder: {os.path.abspath(test_data_dir)}")
        if not os.path.isdir(test_data_dir):
            print(f"Test directory {test_data_dir} not found.")
            return
        
        image_files = [f for f in os.listdir(test_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print("No valid image files found in the folder.")
            return
        
        print(f"Found {len(image_files)} images in the folder.")
        
        for image_name in tqdm(image_files, desc="Processing test images"):
            image_path = os.path.join(test_data_dir, image_name)
            try:
                prediction, predicted_class_name = self.predict(image_path)
                print(f"\nTest image: {image_name}")
                print(f"Prediction probabilities: {prediction}")
                print(f"Predicted class: {predicted_class_name}")
            except Exception as e:
                print(f"Error predicting {image_path}: {e}")

    def save_model(self, filepath):
        """Save the trained model and class mapping to a file."""
        joblib.dump((self.model, self.class_mapping), filepath)

    def load_model(self, filepath):
        """Load the model and class mapping from a file."""
        self.model, self.class_mapping = joblib.load(filepath)

if __name__ == "__main__":
    # Example usage
    model = ObjectRecognitionModel()
    
    # Train the model on all data at once
    train_data_dir = 'datasets/train'
    history = model.train(train_data_dir)
    
    # Save the model
    model.save_model('models/object_recognition_model.joblib')
    
    # Predict on all images in a folder after training
    test_data_dir = 'datasets/test'  # Specify the test folder here
    if os.path.isdir(test_data_dir):
        model.predict_all_in_folder(test_data_dir)
    else:
        print(f"Test directory {test_data_dir} not found. Please provide a valid test folder path.")





# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from PIL import Image
# import numpy as np
# import os
# import joblib
# from tqdm import tqdm

# class ObjectRecognitionModel:
#     def __init__(self, num_classes=10):
#         self.model = RandomForestClassifier(n_estimators=100, random_state=42)
#         self.num_classes = num_classes
#         self.class_mapping = {}

#     def preprocess_image(self, image_path):
#         with Image.open(image_path) as img:
#             img = img.resize((64, 64))  # Reduced size to save memory
#             img_array = np.array(img.convert('RGB')) / 255.0
#             return img_array.flatten()

#     def load_data_batch(self, data_dir, batch_size=100):
#         X, y = [], []
#         class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
#         for class_id, class_name in enumerate(class_dirs):
#             self.class_mapping[class_id] = class_name
#             class_dir = os.path.join(data_dir, class_name)
#             image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
#             for i in range(0, len(image_files), batch_size):
#                 batch_files = image_files[i:i+batch_size]
#                 for image_name in batch_files:
#                     image_path = os.path.join(class_dir, image_name)
#                     try:
#                         X.append(self.preprocess_image(image_path))
#                         y.append(class_id)
#                     except Exception as e:
#                         print(f"Error processing {image_path}: {e}")
                
#                 if len(X) >= batch_size:
#                     yield np.array(X), np.array(y)
#                     X, y = [], []
        
#         if X:
#             yield np.array(X), np.array(y)

#     def train(self, train_data_dir, validation_split=0.2, batch_size=100):
#         X_train, X_val, y_train, y_val = [], [], [], []
        
#         for X_batch, y_batch in tqdm(self.load_data_batch(train_data_dir, batch_size), desc="Loading data"):
#             X_train_batch, X_val_batch, y_train_batch, y_val_batch = train_test_split(
#                 X_batch, y_batch, test_size=validation_split, random_state=42
#             )
#             X_train.extend(X_train_batch)
#             X_val.extend(X_val_batch)
#             y_train.extend(y_train_batch)
#             y_val.extend(y_val_batch)

#         X_train, y_train = np.array(X_train), np.array(y_train)
#         X_val, y_val = np.array(X_val), np.array(y_val)

#         print("Training model...")
#         self.model.fit(X_train, y_train)

#         train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
#         val_accuracy = accuracy_score(y_val, self.model.predict(X_val))

#         print(f"Training Accuracy: {train_accuracy:.4f}")
#         print(f"Validation Accuracy: {val_accuracy:.4f}")

#         return {"train_accuracy": train_accuracy, "val_accuracy": val_accuracy}

#     def predict(self, image_path):
#         image_array = self.preprocess_image(image_path)
#         prediction = self.model.predict_proba([image_array])[0]
#         predicted_class_id = np.argmax(prediction)
#         predicted_class_name = self.class_mapping.get(predicted_class_id, "Unknown")
#         return prediction, predicted_class_name

#     def save_model(self, filepath):
#         joblib.dump((self.model, self.class_mapping), filepath)

#     def load_model(self, filepath):
#         self.model, self.class_mapping = joblib.load(filepath)

# if __name__ == "__main__":
#     # Example usage
#     model = ObjectRecognitionModel()
    
#     # Train the model
#     train_data_dir = 'datasets/train'
#     history = model.train(train_data_dir)
    
#     # Save the model
#     model.save_model('models/object_recognition_model.joblib')
    
#     # Make a prediction
#     test_image_path = 'datasets/train/rottenbanana/b_r005.png'
#     if os.path.isfile(test_image_path):
#         prediction, predicted_class = model.predict(test_image_path)
#         print(f"Prediction probabilities: {prediction}")
#         print(f"Predicted class: {predicted_class}")
#     else:
#         print(f"Test image not found at {test_image_path}. Please provide a valid image path for prediction.")