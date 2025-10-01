"""
Particles classification using DINO.

Copyright 2025 Nikolai Andrianov, nia@geus.dk

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import shutil
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.cluster import KMeans
import pandas as pd
from transformers import AutoFeatureExtractor, AutoModel

# -------------------
# Config
# -------------------
DATA_DIR = "../data/test_level_1_below_thresh_200_embed_class_filter"   # path to images
OUTPUT_DIR = "../data/test_level_1_below_thresh_200_embed_class_filter_clusters_dino"  # Directory to store clustered images
BATCH_SIZE = 32
N_CLUSTERS = 8   # Number of categories (unsupervised assumption)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# Custom Dataset
# -------------------
class PalynoDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".png")]
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_path  # Return path instead of label

# -------------------
# Load DINOv2 Model
# -------------------
model_name = "facebook/dino-vitb16"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(DEVICE)
model.eval()

# -------------------
# Feature Extraction
# -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std)
])

dataset = PalynoDataset(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

features, paths = [], []

with torch.no_grad():
    for imgs, batch_paths in loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs).last_hidden_state
        pooled_features = outputs.mean(dim=1)  # Global average pooling
        features.append(pooled_features.cpu())
        paths.extend(batch_paths)

features = torch.cat(features).numpy()

# -------------------
# K-Means Clustering
# -------------------
print("Running k-means clustering...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(features)
pseudo_labels = kmeans.labels_

# -------------------
# Save Results
# -------------------
results = pd.DataFrame({"image_path": paths, "pseudo_label": pseudo_labels})
results.to_csv("../data/test_level_1_below_thresh_200_embed_class_filter_labels_dino.csv", index=False)
print("✅ Pseudo-labels saved to ../data/test_level_1_below_thresh_200_embed_class_filter_labels_dino.csv")
print(results.head())

# -------------------
# Organize Images into Folders
# -------------------
print(f"Creating {N_CLUSTERS} cluster folders in '{OUTPUT_DIR}'...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create cluster folders
cluster_dirs = []
for i in range(N_CLUSTERS):
    cluster_folder = os.path.join(OUTPUT_DIR, f"cluster_{i}")
    os.makedirs(cluster_folder, exist_ok=True)
    cluster_dirs.append(cluster_folder)

# Copy images to corresponding cluster folder
for img_path, label in zip(paths, pseudo_labels):
    dst_path = os.path.join(cluster_dirs[label], os.path.basename(img_path))
    shutil.copy(img_path, dst_path)

print(f"✅ Images copied into {N_CLUSTERS} cluster folders in '{OUTPUT_DIR}'")
