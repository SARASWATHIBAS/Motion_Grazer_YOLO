import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# === Configuration ===
base_dir = os.path.expanduser('~/Motion_Grazer_YOLO')
input_dir = base_dir
model_path = os.path.join(base_dir, 'best.pt')
output_base = os.path.join(base_dir, 'output_images')

bright_dir = os.path.join(output_base, "bright")
pigs_dir = os.path.join(output_base, "pigs")
no_pigs_dir = os.path.join(output_base, "no_pigs")

alpha, beta = 1.5, 30  # contrast and brightness

# === Load YOLO model
model = YOLO(model_path)

# === Create required folders
os.makedirs(bright_dir, exist_ok=True)
os.makedirs(pigs_dir, exist_ok=True)
os.makedirs(no_pigs_dir, exist_ok=True)

# === Get image files (exclude non-image files)
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    print("❗ No images found in the input directory.")
    exit()

# === Process each image
for img_file in tqdm(image_files, desc="Processing Images"):
    input_path = os.path.join(input_dir, img_file)
    bright_path = os.path.join(bright_dir, img_file)

    # Read and brighten
    img = cv2.imread(input_path)
    if img is None:
        print(f"⚠️ Skipping unreadable file: {img_file}")
        continue

    bright = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imwrite(bright_path, bright)

    # Run YOLO detection on bright image
    results = model.predict(source=bright_path, conf=0.25, save=False, verbose=False)[0]

    # Check if pig is detected (class 0)
    has_pig = any(int(cls) == 0 for cls in results.boxes.cls.cpu().numpy())

    # Move to appropriate folder
    out_path = os.path.join(pigs_dir if has_pig else no_pigs_dir, img_file)
    cv2.imwrite(out_path, bright)

print("✅ Sorting complete!")
print(f"🐷 Images with pigs: {len(os.listdir(pigs_dir))}")
print(f"🚫 Images without pigs: {len(os.listdir(no_pigs_dir))}")
