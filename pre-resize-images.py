from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import os

# Config
ORIGINAL_DIR = Path("../Datasets/statefarm/imgs")
FAST_DIR = Path("../Datasets/statefarm/imgs_fast_256") # New folder
TARGET_SIZE = 256

FAST_DIR.mkdir(parents=True, exist_ok=True)

# Get all jpgs
all_imgs = list(ORIGINAL_DIR.glob("**/*.jpg"))

print(f"Resizing {len(all_imgs)} images to {TARGET_SIZE}px...")

for img_path in tqdm(all_imgs):
    # Create corresponding class folder in new dir
    rel_path = img_path.relative_to(ORIGINAL_DIR)
    new_path = FAST_DIR / rel_path
    new_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open, Resize, Save
    if not new_path.exists():
        with Image.open(img_path) as im:
            # Resize preserving aspect ratio roughly or just square if you prefer
            # Here we resize so the smaller edge is 256
            im = im.resize((TARGET_SIZE, TARGET_SIZE)) 
            im.save(new_path, quality=90)

print("Done! Update your DATA_ROOT to point to:", FAST_DIR)