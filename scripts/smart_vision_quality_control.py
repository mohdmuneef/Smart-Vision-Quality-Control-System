import cv2
import numpy as np
import pytesseract
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(filename='quality_control_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Tesseract path (update this based on your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class SmartVisionQualityControl:
    def __init__(self):
        self.object_recognition_model = self.load_object_recognition_model()
        self.freshness_detection_model = self.load_freshness_detection_model()

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Error: Could not load image from path {image_path}")
            return None
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        equalized_img = cv2.equalizeHist(blurred_img)
        thresh_img = cv2.adaptiveThreshold(equalized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        
        return thresh_img, img  # Return both preprocessed and original image

    def extract_text_from_image(self, image):
        custom_config = r'--oem 3 --psm 6'
        extracted_text = pytesseract.image_to_string(image, config=custom_config)
        return extracted_text.strip()

    def load_object_recognition_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        output = Dense(10, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Note: In a real scenario, you would load pre-trained weights here
        return model

    def load_freshness_detection_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # Note: In a real scenario, you would load pre-trained weights here
        return model

    def recognize_object(self, image):
        resized_image = cv2.resize(image, (224, 224))
        expanded_image = np.expand_dims(resized_image, axis=0)
        normalized_image = expanded_image / 255.0
        predictions = self.object_recognition_model.predict(normalized_image)
        # This is a placeholder. In a real scenario, you would map the prediction to actual class labels
        recognized_object = f"Object {np.argmax(predictions[0])}"
        return recognized_object

    def detect_freshness(self, image):
        resized_image = cv2.resize(image, (224, 224))
        expanded_image = np.expand_dims(resized_image, axis=0)
        normalized_image = expanded_image / 255.0
        freshness_score = self.freshness_detection_model.predict(normalized_image)[0][0]
        return freshness_score

    def extract_expiry_date(self, text):
        # This is a simple placeholder. In a real scenario, you would use more sophisticated date extraction
        import re
        date_pattern = r'\d{2}/\d{2}/\d{4}'
        matches = re.findall(date_pattern, text)
        if matches:
            return matches[0]
        return None

    def provide_feedback(self, product_info, freshness_score, expiry_date):
        current_date = datetime.now().date()
        expiry_date = datetime.strptime(expiry_date, "%d/%m/%Y").date() if expiry_date else None

        if freshness_score < 0.5:
            feedback = f"Warning: {product_info['name']} is not fresh, reject the item."
        elif expiry_date and expiry_date < current_date:
            feedback = f"Warning: {product_info['name']} has expired, reject the item."
        else:
            feedback = f"{product_info['name']} is fresh and good for shipment."
        
        logging.info(f"Product: {product_info['name']}, Freshness Score: {freshness_score}, Expiry Date: {expiry_date}, Feedback: {feedback}")
        return feedback

    def process_image(self, image_path):
        preprocessed_image, original_image = self.preprocess_image(image_path)
        if preprocessed_image is None:
            return "Failed to process image"

        extracted_text = self.extract_text_from_image(preprocessed_image)
        recognized_object = self.recognize_object(original_image)
        freshness_score = self.detect_freshness(original_image)
        expiry_date = self.extract_expiry_date(extracted_text)

        product_info = {
            'name': recognized_object,
            'details': extracted_text
        }

        feedback = self.provide_feedback(product_info, freshness_score, expiry_date)

        result = {
            'extracted_text': extracted_text,
            'recognized_object': recognized_object,
            'freshness_score': freshness_score,
            'expiry_date': expiry_date,
            'feedback': feedback
        }

        return result

def main():
    quality_control_system = SmartVisionQualityControl()

    # Process all images in a folder
    folder_path = 'datasets/test'
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, filename)
                print(f"Processing {image_path}")
                result = quality_control_system.process_image(image_path)
                print(result)
                print("---")

if __name__ == "__main__":
    main()