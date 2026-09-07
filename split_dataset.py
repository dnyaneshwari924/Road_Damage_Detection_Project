import os
import shutil
import random

BASE_DIR = r"C:\Users\Dnyaneshwari Jogdand\Documents\AI_Road_Damage\archive\data"

IMAGE_DIR = os.path.join(BASE_DIR, "Images")
LABEL_DIR = os.path.join(BASE_DIR, "labels-YOLO")

OUTPUT_DIR = os.path.join(BASE_DIR, "road_dataset")

TRAIN_IMAGES = os.path.join(OUTPUT_DIR, "train", "images")
TRAIN_LABELS = os.path.join(OUTPUT_DIR, "train", "labels")
VAL_IMAGES = os.path.join(OUTPUT_DIR, "val", "images")
VAL_LABELS = os.path.join(OUTPUT_DIR, "val", "labels")

for folder in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
    os.makedirs(folder, exist_ok=True)

images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

random.seed(42)
random.shuffle(images)

split_index = int(len(images) * 0.8)

train_images = images[:split_index]
val_images = images[split_index:]

print("Total images:", len(images))
print("Training images:", len(train_images))
print("Validation images:", len(val_images))

for image in train_images:
    image_path = os.path.join(IMAGE_DIR, image)
    label_name = os.path.splitext(image)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_name)

    if os.path.exists(label_path):
        shutil.copy2(image_path, TRAIN_IMAGES)
        shutil.copy2(label_path, TRAIN_LABELS)

for image in val_images:
    image_path = os.path.join(IMAGE_DIR, image)
    label_name = os.path.splitext(image)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_name)

    if os.path.exists(label_path):
        shutil.copy2(image_path, VAL_IMAGES)
        shutil.copy2(label_path, VAL_LABELS)

print("Dataset split completed!")
print("Output:", OUTPUT_DIR)