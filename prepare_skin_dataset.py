import os
import shutil
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

skin_dir = os.path.join(BASE_DIR, "dataset", "skin")

meta_path = os.path.join(skin_dir, "HAM10000_metadata.csv")

img_part1 = os.path.join(skin_dir, "HAM10000_images_part_1")
img_part2 = os.path.join(skin_dir, "HAM10000_images_part_2")

benign_dir = os.path.join(skin_dir, "benign")
malignant_dir = os.path.join(skin_dir, "malignant")

os.makedirs(benign_dir, exist_ok=True)
os.makedirs(malignant_dir, exist_ok=True)

# ✅ Read CSV (NOT excel)
df = pd.read_csv(meta_path)

benign_count = 0
malignant_count = 0
missing_count = 0

# Benign classes in HAM10000
BENIGN = {"nv", "bkl", "df", "vasc"}   # melanocytic nevus, benign keratosis, dermatofibroma, vascular
# Malignant classes
MALIGNANT = {"mel", "bcc", "akiec"}    # melanoma, basal cell carcinoma, actinic keratoses/intraepithelial carcinoma

for _, row in df.iterrows():
    image_id = str(row["image_id"])
    dx = str(row["dx"]).strip().lower()

    # Find image in both folders
    img_file = os.path.join(img_part1, image_id + ".jpg")
    if not os.path.exists(img_file):
        img_file = os.path.join(img_part2, image_id + ".jpg")

    if not os.path.exists(img_file):
        missing_count += 1
        continue

    # Copy into binary classes
    if dx in BENIGN:
        shutil.copy(img_file, os.path.join(benign_dir, image_id + ".jpg"))
        benign_count += 1
    elif dx in MALIGNANT:
        shutil.copy(img_file, os.path.join(malignant_dir, image_id + ".jpg"))
        malignant_count += 1
    else:
        # If any unexpected label appears, skip it safely
        pass

print("✅ Skin dataset prepared!")
print("Benign images:", benign_count)
print("Malignant images:", malignant_count)
print("Missing images:", missing_count)
print("Saved to:")
print(" -", benign_dir)
print(" -", malignant_dir)