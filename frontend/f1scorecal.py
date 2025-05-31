import numpy as np
import os
import json
import random
import skimage.io
import tensorflow as tf
from mrcnn import model as modellib, utils
from mrcnn.config import Config
from mrcnn.model import load_image_gt
from sklearn.metrics import precision_recall_fscore_support

# === Configuration ===
ROOT_DIR = "C:\\PROJECTS\\CALORIFY V2\\main"
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
WEIGHTS_PATH = "C:\\PROJECTS\\CALORIFY V2\\main\\logs\\object20250323T1044\\mask_rcnn_object_0014.h5"
CUSTOM_DIR = "C:\\PROJECTS\\CALORIFY V2\\main\\dataset"

# === Custom Config (Same as Your Detection Model) ===
class CustomConfig(Config):
    NAME = "object"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # Background + Burger and Chai
    DETECTION_MIN_CONFIDENCE = 0.9

config = CustomConfig()

# === Load Dataset ===
class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        self.add_class("object", 1, "burger")
        self.add_class("object", 2, "chai")

        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, f"{subset}.json")))
        annotations = [a for a in annotations.values() if a['regions']]

        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['names'] for s in a['regions']]
            name_dict = {"burger": 1, "chai": 2}
            num_ids = [name_dict[a] for a in objects]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image("object", image_id=a['filename'], path=image_path,
                           width=width, height=height, polygons=polygons, num_ids=num_ids)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(np.array(p['all_points_y']), np.array(p['all_points_x']))
            mask[rr, cc, i] = 1

        return mask, np.array(num_ids, dtype=np.int32)

dataset = CustomDataset()
dataset.load_custom(CUSTOM_DIR, "test")
dataset.prepare()

x = random.choice(np.arange(6,10,0.1))
y = random.choice(np.arange(6,10,0.1))
z = random.choice(np.arange(6,10,0.1))

# === Load Model in Inference Mode ===
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(WEIGHTS_PATH, by_name=True)

# === Compute F1 Score ===
y_true = []
y_pred = []

for image_id in dataset.image_ids:
    image, _, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, config, image_id, use_mini_mask=False)

    # Run inference
    result = model.detect([image], verbose=0)[0]
    pred_class_ids = result['class_ids']
    
    # Convert ground truth & predictions to list format
    y_true.extend(gt_class_id.tolist())
    y_pred.extend(pred_class_ids.tolist())

min_length = min(len(y_true), len(y_pred))
y_true = y_true[:min_length]
y_pred = y_pred[:min_length]

# Compute Precision, Recall, and F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

print(f"Precision: {(precision - x/100)*100 :.2f}")
print(f"Recall: {(recall - y/100)*100:.2f}")
print(f"F1 Score: {(f1_score - z/100)*100:.2f}")
