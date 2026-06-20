from PIL import Image
import os

dataset_path = "dataset/bone"

deleted = 0

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        path = os.path.join(root, file)

        try:
            img = Image.open(path)
            img.verify()

        except Exception:

            print("❌ Removing corrupted image:", path)

            os.remove(path)

            deleted += 1

print(f"\n✅ Finished. Removed {deleted} corrupted images.")