import os
import cv2
import matplotlib.pyplot as plt
import base64
from ultralytics import YOLO
from flask import Flask, request, send_file, jsonify
from PIL import Image
import io
from flask_cors import CORS  # Import CORS
import requests  # To fetch the image from the URL
from urllib.parse import urlparse, unquote

# Initialize the Flask app
app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST"]}})


# Load the YOLO model
# model = YOLO("/home/adli/projects/brand_logo/best.pt")
model = YOLO("./runs/detect/train13/weights/best.pt")

# Directory for processed images
output_dir = 'IMAGES/processed_images'
os.makedirs(output_dir, exist_ok=True)

def predict_and_save(src, filename):
    result_predict = model.predict(source=src)

    # The result's plot will already be in BGR format, no need to convert it
    plot = result_predict[0].plot()  # Keep the BGR format

    # Save the processed image
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, plot)  # Save the image in BGR format

    return output_path


def image_to_base64(image_path):
    # Convert the image to base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # Check if an image file was sent
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Get the image from the request
    image_file = request.files['image']

    # Read the image
    image = Image.open(image_file.stream)

    # Save the uploaded image temporarily
    temp_image_path = os.path.join(output_dir, image_file.filename)
    image.save(temp_image_path)

    # Perform the prediction and save the processed image
    processed_image_path = predict_and_save(temp_image_path, image_file.filename)

    # Convert the processed image to base64
    image_base64 = image_to_base64(processed_image_path)

    # Remove the temporary saved file (optional cleanup)
    os.remove(temp_image_path)

    # Return the base64 result
    return jsonify({"image_base64": image_base64})

# New endpoint to process images from a URL
# New endpoint to process images from a URL
@app.route('/predict_url', methods=['POST'])
def predict_from_url():
    # Check if an image URL is provided
    data = request.json
    if 'image_url' not in data:
        return jsonify({"error": "No image URL provided"}), 400

    image_url = data['image_url']

    # Fetch the image from the URL
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Ensure the request was successful
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 400

    # Extract the filename and remove query parameters
    parsed_url = urlparse(image_url)
    filename = os.path.basename(parsed_url.path)  # Extracts the file name without query parameters
    filename = unquote(filename)  # Decode any percent-encoded characters

    # Convert the image from the URL into an Image object
    try:
        image = Image.open(io.BytesIO(response.content))
    except IOError:
        return jsonify({"error": "Invalid image format"}), 400

    # Save the fetched image temporarily
    temp_image_path = os.path.join(output_dir, filename)
    image.save(temp_image_path)

    # Perform the prediction and save the processed image
    processed_image_path = predict_and_save(temp_image_path, filename)

    # Convert the processed image to base64
    image_base64 = image_to_base64(processed_image_path)

    # Clean up the temporary file (optional)
    os.remove(temp_image_path)

    # Return the base64 result
    return jsonify({"image_base64": image_base64})
# @app.route('/predict', methods=['POST'])
# def predict_endpoint():
#     # Check if an image file was sent
#     if 'image' not in request.files:
#         return jsonify({"error": "No image provided"}), 400

#     # Get the image from the request
#     image_file = request.files['image']

#     # Read the image
#     image = Image.open(image_file.stream)

#     # Save the uploaded image temporarily
#     temp_image_path = os.path.join(output_dir, image_file.filename)
#     image.save(temp_image_path)

#     # Perform the prediction and save the processed image
#     processed_image_path = predict_and_save(temp_image_path, image_file.filename)

#     # Instead of converting to base64, directly send the file
#     response = send_file(processed_image_path, mimetype='image/jpeg', as_attachment=True, download_name=image_file.filename)

#     # Remove the temporary saved file (optional cleanup)
#     # os.remove(temp_image_path)

#     # Return the file response
#     return response

@app.route('/hello', methods=['GET'])
def hello_endpoint():
    return jsonify({"message": "Hello, World!"})
# Run the Flask app and specify the port
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)  # Specify the host and port here

# python flask_app.py
