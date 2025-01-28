# Smart Vision Quality Control System

This project implements an advanced quality control system using computer vision and machine learning techniques. It's designed to automate the process of inspecting products, particularly in the context of e-commerce and grocery items.

## Features

- **Image Processing**: Preprocesses images to enhance quality and prepare them for analysis.
- **Optical Character Recognition (OCR)**: Extracts text from product labels and packaging using EasyOCR.
- **Object Recognition**: Identifies and classifies products using a machine learning model.
- **Freshness Detection**: Assesses the freshness of produce and perishable items.
- **Quality Control Feedback**: Provides automated feedback on product quality and suitability for shipment.

## Key Components

1. **Image Preprocessing**: Enhances image quality for better analysis.
2. **OCR Module**: Extracts text information from product labels.
3. **Object Recognition Model**: Classifies products based on visual features.
4. **Freshness Detection Algorithm**: Evaluates the freshness of products.
5. **Feedback System**: Generates quality control reports based on the analysis.

## Technologies Used

- Python
- OpenCV
- EasyOCR
- scikit-learn
- PIL (Python Imaging Library)
- NumPy

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare your dataset in the `datasets/train` folder
4. Run the main script: `python scripts/smart_vision_quality_control.py`

## Project Structure

- `datasets/`: Contains training and test data
- `scripts/`: Python scripts for each module
- `models/`: Saved machine learning models
- `requirements.txt`: List of project dependencies

## Contributing

Contributions to improve the system are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

[MIT License](LICENSE)