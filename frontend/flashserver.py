import sys
from flask import Flask, request, jsonify, send_file
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.draw
import skimage.io
from io import BytesIO
from werkzeug.utils import secure_filename

from mrcnn import model as modellib, utils
from mrcnn.visualize import display_instances
from mrcnn.config import Config

from flask_cors import CORS

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Configurations
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/processed_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Model Configuration
class CustomConfig(Config):
    NAME = "object"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # Background + 2 classes (burger, chai)
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.9

# Load Model
MODEL_DIR = "C:\\PROJECTS\\CALORIFY V2\\main\\logs"
WEIGHTS_PATH = "C:\\PROJECTS\\CALORIFY V2\\main\\logs\\object20250323T1044\\mask_rcnn_object_0014.h5"

config = CustomConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(WEIGHTS_PATH, by_name=True)

# Class names
class_names = ["BG", "burger", "chai"]
OUTPUT_FOLDER = "static/processed_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)



@app.route("/detect", methods=["POST"])
def detect_objects():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400


    file = request.files["image"]
    print("received image")
    sys.stdout.flush()
    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # Load Image
    image = skimage.io.imread(image_path)

    # Run object detection
    results = model.detect([image], verbose=1)
    r = results[0]

    # Random calorie estimation for now

    # Extract detected class names
    # detected_classes = [class_names[i] for i in r["class_ids"]]

    # burgercalorie = np.random.randint(450,600)
    # chaicalorie = np.random.randint(90,150)

    # estimated_calories = 0  # Initialize the calorie counter
    # if len(detected_classes) > 0:
    #     for i in range(0, len(detected_classes)):
    #         if(detected_classes[i] == 'burger'):
    #             estimated_calories += burgercalorie
    #         else:
    #             estimated_calories += chaicalorie
                
    # estimated_calories = np.random.randint(100, 500)

    detected_classes = [class_names[i] for i in r["class_ids"]]

# Define approximate calorie density (cal/pixel) for each food item
    calorie_density = {
        "burger": np.random.uniform(0.005, 0.008),  # Calories per pixel
        "chai": np.random.uniform(0.0002, 0.0004)
    }

    # Calculate mask areas for detected objects
    mask_areas = [np.sum(r["masks"][:, :, i]) for i in range(r["masks"].shape[-1])]

    # Initialize calorie counter
    estimated_calories = 0  

    # Calculate calories based on detected objects
    for i in range(len(detected_classes)):
        food_item = detected_classes[i]
        if food_item in calorie_density:  # Check if food type exists
            temp = calorie_density[food_item] * mask_areas[i]
            if  temp > 400 and temp < 800:
                estimated_calories += calorie_density[food_item] * mask_areas[i]
            elif temp > 800:
                estimated_calories += np.random.randint(700,800)
            elif temp < 400 and food_item == "burger":
                estimated_calories += np.random.randint(300,400)
            elif temp < 400 and food_item == "chai":
                estimated_calories += np.random.randint(80,120)
            


    # Ensure output directory exists
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    # Plot detected objects
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)  # Use the actual image
    display_instances(image, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"], ax=ax)
    plt.savefig(output_path, bbox_inches="tight")

    

    plt.close(fig)

    processed_image_url = f"http://127.0.0.1:5000/{OUTPUT_FOLDER}/{filename}"
    if not os.path.exists(output_path):
        return jsonify({"error": "Processed image not found"}), 500


    

    return jsonify({
        "image_url": processed_image_url,
        "calories": estimated_calories
    })

if __name__ == "__main__":
    app.run(debug=True)
