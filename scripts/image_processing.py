import cv2
import numpy as np
import os

def preprocess_image(image_path, output_folder=None):
    # Load the image
    img = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if img is None:
        print(f"Error: Could not load image from path {image_path}")
        return None
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # Histogram Equalization for better contrast
    equalized_img = cv2.equalizeHist(blurred_img)
    
    # Adaptive Thresholding
    thresh_img = cv2.adaptiveThreshold(equalized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    
    # Optionally save the processed image
    if output_folder:
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, thresh_img)
        print(f"Processed image saved to {output_path}")
    
    return thresh_img

def process_all_images_in_folder(folder_path, output_folder=None):
    # Ensure output folder exists
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Walk through all folders and subfolders recursively
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # Full path to the image
            image_path = os.path.join(root, filename)
            
            # Only process files with valid image extensions
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(f"Processing {image_path}")

                # Generate the corresponding output folder for each input subfolder
                relative_path = os.path.relpath(root, folder_path)
                output_subfolder = os.path.join(output_folder, relative_path)
                
                # Ensure the output subfolder exists
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                
                # Process and save the image in the corresponding output subfolder
                preprocess_image(image_path, output_subfolder)
            else:
                print(f"Skipping non-image file: {filename}")

if __name__ == "__main__":
    # Example usage
    folder_path = 'datasets/test'
    output_folder = 'datasets/processed'
    process_all_images_in_folder(folder_path, output_folder)