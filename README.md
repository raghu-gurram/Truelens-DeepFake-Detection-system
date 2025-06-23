# TrueLens DeepFake Video Detection

TrueLens is a Flask-based web application for detecting deepfake images using a pre-trained Xception model. The app provides a REST API endpoint for image prediction and can be easily extended for video or batch processing.

## Features
- Upload an image and get a prediction (real or fake)
- Uses TensorFlow and a pre-trained Xception model
- Simple REST API for integration
- Ready for deployment or further development

## Requirements
- Python 3.7+
- Flask
- TensorFlow
- Pillow
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TrueLens-DeepFakeVideoDetection.git
   cd TrueLens-DeepFakeVideoDetection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your trained model file (`xception_model.h5`) in the `models/` directory.

## Usage
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Send a POST request to `/predict` with an image file:
   - Endpoint: `http://localhost:5000/predict`
   - Form field: `file` (the image to check)

   Example using `curl`:
   ```bash
   curl -X POST -F "file=@path_to_image.jpg" http://localhost:5000/predict
   ```
   Response:
   ```json
   { "prediction": 1 }
   ```
   Where `1` means deepfake and `0` means real.

## File Structure
- `app.py` - Main Flask application
- `models/` - Folder for the trained model file
- `assets/` - Static files (CSS, JS, images)
- `index.html`, `front.html`, `1.html` - Frontend HTML files

## Notes
- The model file (`xception_model.h5`) is not included due to size limits. Download or train your own and place it in the `models/` folder.
- For large model files, consider using [Git LFS](https://git-lfs.github.com/) or provide a download link.

## License
MIT License
