import os, shutil, random

SRC = r"COVID-19_Radiography_Dataset"
DST = r"data"

MAPPING = {
    "Normal":          "NORMAL",
    "COVID":           "COVID19",
    "Viral Pneumonia": "PNEUMONIA",
}

random.seed(42)
SPLIT = 0.85

for src_name, dst_name in MAPPING.items():
    src_images_path = os.path.join(SRC, src_name, "images")
    folder = src_images_path if os.path.exists(src_images_path) else os.path.join(SRC, src_name)
    
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    
    split_idx = int(len(images) * SPLIT)
    splits = {"train": images[:split_idx], "val": images[split_idx:]}
    
    for phase, files in splits.items():
        out_dir = os.path.join(DST, phase, dst_name)
        os.makedirs(out_dir, exist_ok=True)
        for fname in files:
            shutil.copy(os.path.join(folder, fname), os.path.join(out_dir, fname))
        print(f"✓ {phase}/{dst_name}: {len(files)} صورة")

print("\n✅ الداتا جاهزة!")