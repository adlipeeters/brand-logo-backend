import cv2
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

# Load the model and prepare the data
model = YOLO("./runs/detect/train13/weights/best.pt")

# Create a directory for processed images if it doesn't exist
output_dir = 'IMAGES/processed_images'
os.makedirs(output_dir, exist_ok=True)

def predict(src):
    result_predict = model.predict(source=src)
    plot = result_predict[0].plot()

    # Convert the image to RGB for proper color display
    plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)

    # Save the processed image
    filename = os.path.basename(src)  # Get the image filename
    output_path = os.path.join(output_dir, filename)  # Construct the output path
    cv2.imwrite(output_path, plot)  # Save the image

    # Optionally display the image using matplotlib
    plt.imshow(plot)
    plt.axis('off')  # Hide axes for a cleaner display
    plt.show()

    print(f"Image saved at: {output_path}")

# Test with an image
path = 'IMAGES/test_images/porsche.jpg'
predict(path)
