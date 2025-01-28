import easyocr
import cv2
import numpy as np
import os
from PIL import Image

class OCRExtractor:
    def __init__(self):
        # Initialize the OCR reader
        self.reader = easyocr.Reader(['en'])  # Use English language

    def has_text(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get image with only black and white
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Calculate percentage of white pixels
        text_pixel_ratio = np.sum(thresh == 255) / thresh.size

        # If the image has more than 1% white pixels, consider it as having text
        return text_pixel_ratio > 0.01

    def extract_text_from_image(self, image_path):
        # Read the image
        image = cv2.imread(image_path)
        
        # Check if the image has text
        if not self.has_text(image):
            print(f"Skipping {image_path} - No text detected")
            return None

        # Perform OCR
        results = self.reader.readtext(image)
        
        # Extract and join the detected text
        extracted_text = ' '.join([text for _, text, conf in results if conf > 0.5])
        
        return extracted_text.strip() if extracted_text else None

    def process_all_images_in_folder(self, folder_path, output_file=None):
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_path = os.path.join(root, filename)
                    print(f"Processing {image_path}")
                    extracted_text = self.extract_text_from_image(image_path)
                    
                    if extracted_text:
                        print("Extracted Text:", extracted_text)
                        
                        # Optionally write the extracted text to a file
                        if output_file:
                            with open(output_file, 'a') as f:
                                f.write(f"Extracted text from {image_path}:\n{extracted_text}\n\n")
                    else:
                        print(f"No text extracted from {image_path}")
                else:
                    print(f"Skipping non-image file: {filename}")

if __name__ == "__main__":
    # Example usage
    ocr_extractor = OCRExtractor()
    processed_images_folder = 'datasets/processed'  # Folder where processed images are saved
    output_file = 'ocr_results.txt'  # Log the OCR results to this file
    ocr_extractor.process_all_images_in_folder(processed_images_folder, output_file)